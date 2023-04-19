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

import statsapi

from os import path

import json

import scipy.stats as stats
from scipy.stats import binom, binom_test, binomtest
from scipy.stats import distributions as dist
from scipy.stats import beta

import pandasql as ps
from pandasql import sqldf

from st_aggrid import AgGrid


#################
# Web-app title #
#################
st.markdown("<h1 style='text-align: center; color: red;'>Batter vs Clustered Pitchers</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: black;'>Using advanced pitch metric data from BaseballSavant, a KMeans Clustering model is used to accurately group pitchers. Then we compare Batter statistics vs each cluster to gain insight into pitcher preferences and predict future outcomes.  </h4>", unsafe_allow_html=True)


#####################
# Hitter Select Box #
#####################
hitters = ['Mike_Trout', 'Shohei_Ohtani',  'Gio_Urshela', 'Hunter_Renfroe', 'Taylor_Ward', 'Anthony_Rendon', 'Luis_Rengifo', 'Brandon_Drury']
hitter_choose = st.selectbox('Choose Batter', hitters)

########################
# Open Selected Hitter #
########################
def open_hitter():
    with open(f"{hitter_choose}.json") as f:
        data_str = f.read()
        data_hitter = json.loads(data_str)  
    return data_hitter

########################
# Load Selected Hitter #
########################
def load_hitter():
    
    data_hitter = open_hitter()
    hitter = pd.DataFrame(data_hitter)

    hitter_cols = ['pitcher','pitcherFirstName', 'pitcherLastName',
            'ab', 'h', 'avg', 'hr', 'so', 'bb', '2b', '3b', 
            'obp', 'slg', 'ops']
    
    hitter = hitter[hitter_cols]
    hitter['full_name'] = hitter['pitcherFirstName'] + ' ' + hitter['pitcherLastName']
    hitter['full_name'] = hitter['full_name'].str.strip()
    hitter['slg'] = hitter['slg'].astype(float)
    hitter['ops'] = hitter['ops'].astype(float)

    
    return hitter

df_hitter = load_hitter()

####################################################
# Load KMeans Clustering Model : loaded_metric_model
####################################################

def load_metric_model():
    file_metrics = 'kmeans_metric.sav'
    lmm = pickle.load(open(file_metrics, 'rb'))
    return lmm

loaded_model = load_metric_model()

# ##################################################
# Read in clustered pitch metric csv: df_metric ####
# ##################################################

def get_metric_data():
     metric = pd.read_csv("Named_Clustered_Metric.csv")
     metric = metric.set_index("Unnamed: 0")
     metric_df = pd.DataFrame(metric)
     metric_df['full_name'] = metric_df['first_name'] + ' ' + metric_df['last_name']
     metric_df['full_name'] = metric_df['full_name'].str.strip()
     return metric_df


df_metric = get_metric_data()


# ##################################
# Aggregate hitter vs cluster data #
# ##################################
def show_clusters():

    probable_cluster = 4

    pyqldf = lambda q: sqldf(q, globals())

    cluster_query = """SELECT metric.Cluster, t.pitcher, ab, h, hr, avg, bb, so, obp 
        FROM df_hitter AS t 
        LEFT JOIN df_metric AS metric 
        ON metric.full_name = t.pitcher 
        WHERE Cluster IS NOT NULL 
        ORDER BY ab DESC"""
    

    hitter_clusters = pyqldf(cluster_query)

    hitter_agg_sum = hitter_clusters.groupby('Cluster').sum()
    hitter_agg_sum['avg'] = hitter_agg_sum['h'] / hitter_agg_sum['ab']


    tab21, tab22, tab23, tab24, tab25 = st.tabs(['Career vs Pitcher', 'Career vs Cluster', 'Pitcher Search', 'Probable Pitcher', 'Live'])


    with tab21:
        st.markdown("<h4 style='text-align: center; color: black;'>Hitter's career statistics vs pitchers in each cluster</h4>", unsafe_allow_html=True)
        st.text("Click the 'hamburger' icon at the top of a column to search, select, and filter.")

        AgGrid(hitter_clusters, height=350, fit_columns_on_grid_load=True)
        
        st.markdown("<pstyl e='text-align: center; color: black;'>Full Screen option in top right of dataframe</p>", unsafe_allow_html=True)
    
    with tab22:
        st.markdown("<h4 style='text-align: center; color: black;'>Hitter's career statistics vs each cluster</h4>", unsafe_allow_html=True)

        st.dataframe(hitter_agg_sum)
        st.markdown("<p style='text-align: center; color: black;'>Sort columns by left-click</p>", unsafe_allow_html=True)

    with tab23:
        # icon("search")
        # selected = st.text_input("", "Search...")
        # button_clicked = st.button("OK")
        # pitcher_last_name = st.text_input('Search by last name', 'pitcher')
        
        all_pitchers = pd.read_csv('Named_Clustered_Metric.csv')
        all_pitchers_df = pd.DataFrame(all_pitchers)
        all_pitchers_df['full_name'] = all_pitchers_df['first_name'] + ' ' + all_pitchers_df['last_name']
        all_pitchers_df['full_name'] = all_pitchers_df['full_name'].str.strip()

        all_pitchers_df = all_pitchers_df[['Cluster', 'full_name']]
        
        st.text("Click the 'hamburger' icon at the top of a column to search, select, and filter.")
        AgGrid(all_pitchers_df, height=350, fit_columns_on_grid_load=True)

    with tab24: 
        st.write("These are upcoming pitchers, their cluster, and the hitter's stats vs their cluster")
        from new import prob_pitch_df
        pp = prob_pitch_df()
        
        xtry = pd.merge(pp, hitter_agg_sum, left_on='Cluster', right_index=True)
        st.dataframe(xtry)

    with tab25:
        from new import live_box
        lb = live_box()
        st.text('If box-score is empty, no game is currently underway')
        st.dataframe(lb)


    return hitter_agg_sum



# ##################################################
#       Plot Batting Average Distributions      ####
# ##################################################

def density():

    hitter_agg = show_clusters()

    
    st.header('Bayesian Updated Batting Avg vs Cluster')

    clusters = hitter_agg.index

    clusters_choose = st.selectbox('Choose Cluster', clusters)

    cluster_spec_view = hitter_agg.iloc[[clusters_choose]]

    # dropped_agg_sum = hitter_agg.drop([[clusters_choose]])

    total_ab = hitter_agg['ab'].sum()
    total_hit = hitter_agg['h'].sum()

    cluster_ab = cluster_spec_view['ab']
    cluster_hit = cluster_spec_view['h']

    # prior a 
    a = (total_hit - cluster_hit)
    b = (total_ab - cluster_ab) - (total_hit - cluster_hit)


    # X_outcome = clustered_outcome_df.drop(['Cluster'],axis=1)
    theta_range = np.linspace(0, 1, 1000)
    theta_range_e = theta_range + 0.001 

    prior = stats.beta.cdf(x = theta_range_e, a=a, b=b) - stats.beta.cdf(x = theta_range, a=a, b=b) 
    likelihood = stats.binom.pmf(k = cluster_hit, n = cluster_ab, p = theta_range)
    posterior = likelihood * prior # element-wise multiplication
    small_size = 8
    med_size = 14
    large_size = 22

    
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

    a1 = a
    b1 = b + a
    p1 = a1/b1

    p25 = (cluster_hit[clusters_choose] / cluster_ab[clusters_choose]) * 1.1
    p50 = (cluster_hit[clusters_choose] / cluster_ab[clusters_choose]) * 1.2




    pval300 = binom_test(cluster_hit[clusters_choose], cluster_ab[clusters_choose], p=p25, alternative='less')
    pval333 = binom_test(cluster_hit[clusters_choose], cluster_ab[clusters_choose], p=p50, alternative='less')
    st.write(f"The probability that {hitter_choose}'s avg vs this cluster is greater than {round(p25*1000,0)} is:  {round(pval300, 4)*100}%")
    st.write(f"The probability that {hitter_choose}'s avg vs this cluster is greater than {round(p50*1000,0)} is: {round(pval333, 4)*100}%")
density()



# def data_2_cols():
#      tab1, tab2 = st.tabs(['Cluster Distribution', 'Model Precision'])
     
#      metric_cols = df_metric.columns 

#     #  with col1:
#     #     # viz_cols = df_metric.columns
#     #     # viz_cole = metric_cols[0:55]
#     #     st.header('These are the predictors used for our KMeans model')
#     #     st.write(metric_cols[0:55])

#      with tab1:
#         st.header('Distribution of Clusters')
#         cluster_distribution = plt.figure(figsize=(10, 10))
#         sns.set(font_scale = 2)
#         sns.countplot(x='Cluster', data=df_metric, palette ='deep').set(title='Pitcher Clusters')
#         st.pyplot(cluster_distribution)
     
#      with tab2: 
#         #  df_metric_tab3_columns = df_metric.columns
#          metric_col3_columns = metric_cols[0:56]
#          df_col3 = df_metric[metric_col3_columns]
#          X_metric = df_col3.drop(['Cluster'],axis=1)
#          y_metric= df_col3[['Cluster']]
#          X_train_metric, X_test_metric, y_train_metric, y_test_metric =train_test_split(X_metric, y_metric, test_size=0.3)
         
#          kmeans_cluster_metric= DecisionTreeClassifier(criterion="entropy")
#          kmeans_cluster_metric.fit(X_train_metric, y_train_metric)
#          y_pred_metric = kmeans_cluster_metric.predict(X_test_metric)
#          tested_metric = loaded_model.predict(X_test_metric)
#          eval_metric = classification_report(y_test_metric, y_pred_metric, output_dict=True) 
#          eval_metric = pd.DataFrame(eval_metric).T

#          st.header('Classification Accuracy')
        
#          st.dataframe(eval_metric[['precision','f1-score']],400,400)
#          st.text('''Refresh page for different data splits 
#         resulting in different f1-scores''')
     
#      return tab1, tab2 

# col_viz = data_2_cols()


# from new import prob_pitch

# prob_pitch()

# st.write(probable_pitcher)
