from teradataml import (
    DataFrame,
    ScaleFit,
    ScaleTransform,
    execute_sql,
    db_drop_table
)
from teradataml import KMeans 
# from lime.lime_tabular import LimeTabularExplainer
from aoa import (
    record_training_stats,
    aoa_create_context,
    ModelContext
)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

   

def plot_cluster_size(pd_cluster, img_filename):
    pd_cluster.plot(x='td_clusterid_kmeans', y='td_size_kmeans',kind='bar').set_title('Cluster Size')
    plt.title('# of Customers per Cluster')
    plt.xlabel('Cluster ID')
    plt.ylabel('Count of Customers')
    plt.legend('', frameon=False)
    fig = plt.gcf()
    fig.savefig(img_filename, dpi=500)
    plt.clf()
    
def train(context: ModelContext, **kwargs):
    aoa_create_context()
    
    # Extracting feature names, target name, and entity key from the context
    feature_names = context.dataset_info.feature_names
    # target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    # Load the training data from Teradata
    train_df = DataFrame.from_query(context.dataset_info.sql)

#     print ("Scaling using InDB Functions...")
#     # X_train = train_df.drop(['anomaly_int','WELDING_ID'], axis = 1)
#     # y_train = train_df.select(["anomaly_int"])
#     # Scale the training data using the ScaleFit and ScaleTransform functions
#     scaler = ScaleFit(
#         data=train_df,
#         target_columns = feature_names,
#         scale_method="RANGE",
#         global_scale=False
#     )

#     scaled_train = ScaleTransform(
#         data=train_df,
#         object=scaler.output,
#         accumulate = [entity_key]
#     )
    
#     scaler.output.to_sql(f"scaler_${context.model_version}", if_exists="replace")
#     scaled_train.result.to_sql("scale_transform_data",if_exists="replace" )
#     print("Saved scaler")
    
         
    print("Starting training using teradata kmeans...")
    
    qry = f'''SELECT * from TD_KMeans (
    ON Transformed_Customer_Data AS InputTable
    OUT TABLE ModelTable(KMeans_Model)
    USING
        IdColumn('CustomerID')
        TargetColumns('TotalQuantity','TotalPrice','TotalItems','TotalSales','SalesPerItem')
        StopThreshold({context.hyperparams["StopThreshold"]})---(0.001)---(0.0275)---(0.0395)
        NumClusters({context.hyperparams["NumClusters"]}) ---(5)
        Seed({context.hyperparams["Seed"]})---(42)
        MaxIterNum({context.hyperparams["MaxIterNum"]})---(25) --400
        NumInit({context.hyperparams["NumInit"]})---(1)
        InitialCentroidsMethod({context.hyperparams["InitialCentroidsMethod"]}) ---('KMEANS++')
        
    ) AS dt;'''
    
    
# {'target_columns': ['TotalQuantity', 'TotalPrice', 'TotalItems', 'TotalSales', 'SalesPerItem'], 
#  'num_clusters': 5, 'seed': 42, 'threshold': 0.001, 'iter_max': 25, 'num_init': 1, 
#  'id_column': 'CustomerID', 'initialcentroids_method': 'KMEANS++', 'data': None},    
    try:
        execute_sql(qry)
    except:
        db_drop_table("KMeans_Model")
        execute_sql(qry)
    # RF_classifier = osml.RandomForestClassifier(n_estimators=10,max_leaf_nodes=2,max_features='auto',max_depth=2)
    # RF_classifier.fit(X_train, y_train)
    # RF_classifier.deploy(model_name="RF_classifier", replace_if_exists=True)
#     kmeans_out = KMeans(
#     id_column="CustomerID",
#     data=scaled_train.result,
#     target_columns=['TotalQuantity','TotalPrice','TotalItems','TotalSales','SalesPerItem'],
#     output_cluster_assignment=context.hyperparams["output_cluster_assignment"],
#     threshold=context.hyperparams["threshold"],
#     iter_max=context.hyperparams["iter_max"],
#     num_clusters=context.hyperparams["num_clusters"])
      
#     kmeans_out.to_sql("kmeans_model", if_exists="replace")    
    print("Complete kmeans training...")
    
    # Plot cluster size
    df_cluster = DataFrame("KMeans_Model")
    df_cluster = df_cluster[df_cluster.td_clusterid_kmeans.isnull()==0]
    plot_cluster_size(df_cluster.to_pandas(), f"{context.artifact_output_path}/cluster_size")
    
#     record_training_stats(
#         train_df,
#         features=feature_names,
#         targets=[target_name],
#         categorical=[target_name],
#         feature_importance=feature_importance,
#         context=context
#     )
    
    print("All done!")
