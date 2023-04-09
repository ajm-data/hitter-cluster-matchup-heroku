import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN,SpectralClustering
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, r2_score
from sklearn import tree
from sklearn import metrics


from os import path

import json

import scipy.stats as stats
from scipy.stats import binom, binom_test
from scipy.stats import distributions as dist
from scipy.stats import beta

import pandasql as ps
from pandasql import sqldf

from st_aggrid import AgGrid



# Web-app title
st.markdown("<h1 style='text-align: center; color: red;'>Mike Trout vs Clustered Pitchers</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: black;'>Using advanced pitch metric data from BaseballSavant, a KMeans Clustering model is used to accurately group pitchers. Then we compare Mike Trout's statistics vs each cluster to gain insight into pitcher preferences and predict future outcomes.  </h4>", unsafe_allow_html=True)
##################################################
# Mike Trout vs Pitcher Career Numbers : df_trout
##################################################

def load_trout():
    # file_path = path.relpath("c:/Users/ajmme/app3/Mike_Trout.json")
    with open("Mike_Trout.json") as f:
        data_str = f.read()
        data_trout = json.loads(data_str)    
    
    trout = pd.DataFrame(data_trout)
    trout_cols = ['pitcher','pitcherFirstName', 'pitcherLastName',
            'ab', 'h', 'avg', 'hr', 'so', 'bb', '2b', '3b', 
            'obp', 'slg', 'ops']
    
    trout = trout[trout_cols]
    trout['full_name'] = trout['pitcherFirstName'] + ' ' + trout['pitcherLastName']
    trout['full_name'] = trout['full_name'].str.strip()
    trout['slg'] = trout['slg'].astype(float)
    trout['ops'] = trout['ops'].astype(float)

    
    return trout

df_trout = load_trout()

##################################################
##################################################
# Load KMeans Clustering Model : loaded_metric_model
##################################################

def load_metric_model():
    file_metrics = 'kmeans_metric.sav'
    lmm = pickle.load(open(file_metrics, 'rb'))
    return lmm

loaded_model = load_metric_model()

# ##################################################
# ##################################################
# # Read in clustered pitch metric csv: df_metric
# ##################################################

def get_metric_data():
     metric = pd.read_csv("Named_Clustered_Metric.csv")
     metric = metric.set_index("Unnamed: 0")
     metric_df = pd.DataFrame(metric)
     metric_df['full_name'] = metric_df['first_name'] + ' ' + metric_df['last_name']
     metric_df['full_name'] = metric_df['full_name'].str.strip()
     return metric_df


df_metric = get_metric_data()


# ##################################################
# ##################################################
# # Partition visualization header into 2 columns : col_viz
# ##################################################

# @st.cache_resource
def data_2_cols():
     col1, col2 = st.columns(2)
     
     metric_cols = df_metric.columns 

    #  with col1:
    #     # viz_cols = df_metric.columns
    #     # viz_cole = metric_cols[0:55]
    #     st.header('These are the predictors used for our KMeans model')
    #     st.write(metric_cols[0:55])

     with col1:
        st.header('Distribution of Clusters')
        cluster_distribution = plt.figure(figsize=(20, 20))
        sns.set(font_scale = 6)
        sns.countplot(x='Cluster', data=df_metric, palette ='deep').set(title='Pitcher Clusters')
        st.pyplot(cluster_distribution)
     
     with col2: 
        #  df_metric_col3_columns = df_metric.columns
         metric_col3_columns = metric_cols[0:56]
         df_col3 = df_metric[metric_col3_columns]
         X_metric = df_col3.drop(['Cluster'],axis=1)
         y_metric= df_col3[['Cluster']]
         X_train_metric, X_test_metric, y_train_metric, y_test_metric =train_test_split(X_metric, y_metric, test_size=0.3)
         
         kmeans_cluster_metric= DecisionTreeClassifier(criterion="entropy")
         kmeans_cluster_metric.fit(X_train_metric, y_train_metric)
         y_pred_metric = kmeans_cluster_metric.predict(X_test_metric)
         tested_metric = loaded_model.predict(X_test_metric)
         eval_metric = classification_report(y_test_metric, y_pred_metric, output_dict=True) 
         eval_metric = pd.DataFrame(eval_metric).T

         st.header('Classification Accuracy')
        
         st.dataframe(eval_metric[['precision','f1-score']],400,400)
         st.text('''Refresh page for different data splits 
        resulting in different f1-scores''')
     
     return col1, col2 

col_viz = data_2_cols()

# tvs_clusters = trout_view_sum.index

# cluster = st.selectbox('Choose Cluster', tvs_clusters)

def show_clusters():

    # df_trout = load_trout()
    # df_metric = get_metric_data()

    pyqldf = lambda q: sqldf(q, globals())

    cluster_query = """SELECT metric.Cluster, t.pitcher, ab, h, hr, avg, bb, so, obp 
        FROM df_trout AS t 
        LEFT JOIN df_metric AS metric 
        ON metric.full_name = t.pitcher 
        WHERE Cluster IS NOT NULL 
        ORDER BY ab DESC"""
    
    # trout_clusters = pd.DataFrame(ps.sqldf(cluster_query, locals()))

    trout_clusters = pyqldf(cluster_query)

    trout_agg_sum = trout_clusters.groupby('Cluster').sum()
    trout_agg_sum['avg'] = trout_agg_sum['h'] / trout_agg_sum['ab']

    # tvs_clusters = trout_agg_sum.index

    # cluster = st.selectbox('Choose Cluster', tvs_clusters)

    col21, col22 = st.columns(2)


    with col21:
        st.markdown("<h4 style='text-align: center; color: black;'>Trout's career statistics vs pitchers in each cluster</h4>", unsafe_allow_html=True)
        # tvs_clusters = trout_view_sum.index
        # tvs_clusters = trout_clusters['Cluster']

        # cluster = st.selectbox('Choose Cluster', tvs_clusters)
        # pitcher_cluster_view = trout_clusters.iloc[[tvs_clusters]]
        st.dataframe(trout_clusters)
        st.markdown("<p style='text-align: center; color: black;'>Full Screen option in top right of dataframe</p>", unsafe_allow_html=True)

        # AgGrid(trout_clusters)
        # cluster = st.selectbox('Choose Cluster', tvs_clusters)
    # df_trout = load_trout()
    # df_metric = get_metric_data()

        # cluster_query = "SELECT metric.Cluster, t.pitcher, ab, h, hr, avg, bb, so, obp FROM df_trout AS t LEFT JOIN df_metric AS metric ON metric.full_name = t.pitcher WHERE Cluster IS NOT NULL ORDER BY ab DESC"
        # trout_clusters = pd.DataFrame(ps.sqldf(cluster_query, locals()))
        # st.write(pitcher_cluster_view, 325, 400)

    
    with col22:
        st.markdown("<h4 style='text-align: center; color: black;'>Trout's career statistics vs each cluster</h4>", unsafe_allow_html=True)
        # trout_agg_sum = trout_clusters.groupby('Cluster').sum()
        # trout_agg_sum['avg'] = trout_agg_sum['h'] / trout_agg_sum['ab']


        # tr_slg = trout_clusters[['slg', 'ops']].mean()
        # st.dataframe(tr_slg, 325,400)
        # trout_agg_sum['ops'] = trout_agg_sum['ops'] / trout_agg_sum['ab']

        st.dataframe(trout_agg_sum)
        st.markdown("<p style='text-align: center; color: black;'>Sort columns by left-click</p>", unsafe_allow_html=True)


    return trout_agg_sum

# show_clusters()

# 329 hits in 1139 ab 


def density():

    trout_agg_sum = show_clusters()

    # trout_agg_sum = show_clusters()
    
    st.header('Bayesian Updated Batting Avg vs Cluster')

    clusters = trout_agg_sum.index

    clusters_choose = st.selectbox('Choose Cluster', clusters)

    cluster_spec_view = trout_agg_sum.iloc[[clusters_choose]]

    # dropped_agg_sum = trout_agg_sum.drop([[clusters_choose]])

    total_ab = trout_agg_sum['ab'].sum()
    total_hit = trout_agg_sum['h'].sum()

    cluster_ab = cluster_spec_view['ab']
    cluster_hit = cluster_spec_view['h']

    # prior a 
    # a = (total_hit - cluster_hit)
    # b = (total_ab - cluster_ab) - (total_hit - cluster_hit)
    a = 124
    b = 438 - 124
    # X_outcome = clustered_outcome_df.drop(['Cluster'],axis=1)
    theta_range = np.linspace(0, 1, 1000)
    theta_range_e = theta_range + 0.001 

    prior = stats.beta.cdf(x = theta_range_e, a=a, b=b) - stats.beta.cdf(x = theta_range, a=a, b=b) 
    likelihood = stats.binom.pmf(k = cluster_hit, n = cluster_ab, p = theta_range)
    posterior = likelihood * prior # element-wise multiplication
    small_size = 8
    med_size = 14
    large_size = 22
    # st.write(cluster_spec_view)
    # st.write(a, b)

    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    plt.rc('axes', labelsize=6)
    plt.rc('xtick', labelsize=med_size) 
    plt.rc('ytick', labelsize=med_size) 
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(20,7))
    plt.xlabel('θ', fontsize=24)
    axes[0].plot(theta_range, prior, label="Prior", linewidth=3, color='palegreen')
    axes[0].set_title("Prior", fontsize=22)
    axes[1].plot(theta_range, likelihood, label="Likelihood", linewidth=3, color='yellowgreen')
    axes[1].set_title("Sampling (Likelihood)", fontsize=22)
    axes[2].plot(theta_range, posterior, label='Posterior', linewidth=3, color='olivedrab')
    axes[2].set_title("Posterior", fontsize=22)
    
    st.pyplot(fig)




    st.text(f"His batting average estimate (season avg) prior is: {np.argmax(prior)}")
    st.text(f"The maximum likehood estimate for batting average vs the chosen cluster is: {np.argmax(likelihood)}")
    st.text(f"His batting average estimate vs this cluster given data is: {np.argmax(posterior)}")
    # st.pyplot(fig)
    # st.pyplot(axes)

    # likelihood = stats.binom.pmf(k = cluster_hit, n = cluster_ab, p = theta_range)

    a1 = a
    b1 = b + a
    p1 = a1/b1

    pval = binom_test(cluster_hit[clusters_choose], cluster_ab[clusters_choose], p=p1, alternative='less')
    pval300 = binom_test(cluster_hit[clusters_choose], cluster_ab[clusters_choose], p=.3, alternative='less')
    pval333 = binom_test(cluster_hit[clusters_choose], cluster_ab[clusters_choose], p=.333, alternative='less')
    st.write(f"The probability that Trout's avg vs this cluster is greater than his season avg of .281 is:  {round(pval, 4)*100}%")
    st.write(f"The probability that Trout's avg vs this cluster is greater than 333 is: {round(pval333, 4)*100}%")
    # st.write(cluster_hit[0], cluster_ab[0])
density()


