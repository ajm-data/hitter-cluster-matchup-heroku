##################################################
##################################################
# Partition visualization header into 2 columns : col_viz
##################################################

# @st.cache_resource
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

# data_2_cols()