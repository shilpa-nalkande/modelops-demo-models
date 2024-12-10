# from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from teradataml import KMeansPredict
# from lime.lime_tabular import LimeTabularExplainer
from teradataml import(
    DataFrame, 
    copy_to_sql, 
    get_context, 
    get_connection, 
    ScaleTransform, 
    ConvertTo, 
    Silhouette,
    ROC,
    execute_sql,
    db_drop_table
)
from aoa import (
    record_evaluation_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)

import joblib
import json
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

        
# Compute feature importance based on tree traversal
def compute_feature_importance(model,X_train):
    feat_dict= {}
    for col, val in sorted(zip(X_train.columns, model.feature_importances_),key=lambda x:x[1],reverse=True):
        feat_dict[col]=val
    feat_df = pd.DataFrame({'Feature':feat_dict.keys(),'Importance':feat_dict.values()})
    # print(feat_df)
    return feat_df

def plot_feature_importance(fi, img_filename):
    feat_importances = fi.sort_values(['Importance'],ascending = False).head(10)
    feat_importances.plot(kind='barh').set_title('Feature Importance')
    fig = plt.gcf()
    fig.savefig(img_filename, dpi=500)
    plt.clf()


# Define function to plot a confusion matrix from given data
def plot_clusters(cf, img_filename):
    import matplotlib.pyplot as plt
    plt.scatter(data=cf,x='CustomerID', y='td_distance_kmeans', c='td_clusterid_kmeans')
    plt.legend('td_clusterid_kmeans')
    fig = plt.gcf()
    fig.savefig(img_filename, dpi=500)
    plt.clf()


    
    

def evaluate(context: ModelContext, **kwargs):

    aoa_create_context()

    
    feature_names = context.dataset_info.feature_names
    # target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    # Load the test data from Teradata
    test_df = DataFrame.from_query(context.dataset_info.sql)
#     X_test = test_df.drop(['anomaly_int','WELDING_ID'], axis = 1)
#     y_test = test_df.select(["anomaly_int"])
    # Scaling the test set
#     print ("Loading scaler...")
#     scaler = DataFrame(f"scaler_${context.model_version}")

#     scaled_test = ScaleTransform(
#         data=test_df,
#         object=scaler,
#         accumulate = [entity_key]
#     )
    
    print("Evaluating kmeans...")
#     kmeans_model = DataFrame("kmeans_model")
    
#     KMeansPredict_out = KMeansPredict(object=kmeans_model,
#                                       data=test_df)
 
#     # Print the result DataFrames.
#     print(KMeansPredict_out.result)

    qry = '''Create table KMeans_Predict_Output AS (
    SELECT * FROM TD_KMeansPredict (
        ON Transformed_Customer_Data AS InputTable
        ON KMeans_Model as ModelTable DIMENSION
        USING
            OutputDistance('true')
            Accumulate('[1:5]')
    ) AS dt ) with data;'''
    
    try:
        execute_sql(qry)
    except:
        db_drop_table("KMeans_Predict_Output")
        execute_sql(qry)
    
    predict_df = DataFrame("KMeans_Predict_Output")
    qry = '''create table Silhouette_tb as (SELECT * FROM TD_Silhouette(
    ON KMeans_Predict_Output AS inputTable
    USING
        IdColumn('CustomerID')
        ClusterIdColumn('td_clusterid_kmeans')
        TargetColumns('[3:7]')
        OutputType('SCORE')
    ) AS dt) with data;'''
    
    try:
        execute_sql(qry)
    except:
        db_drop_table("Silhouette_tb")
        execute_sql(qry)
    
#      # Extract and store evaluation metrics
    metrics_pd = DataFrame("Silhouette_tb").to_pandas()
    print(metrics_pd)
      
    evaluation = {
        'Score': '{:.8f}'.format(metrics_pd.silhouette_score[0])
    }

     # Save evaluation metrics to a JSON file
    with open(f"{context.artifact_output_path}/metrics.json", "w+") as f:
        json.dump(evaluation, f)     
    

     
    # Generate and save cluster plot

    plot_clusters(predict_df.to_pandas(), f"{context.artifact_output_path}/clusters_plots")
    
#     # Generate and save ROC curve plot
#     roc_out = ROC(
#         data=predict_df,
#         probability_column='randomforestclassifier_predict_1',
#         observation_column=target_name,
#         positive_class='1',
#         num_thresholds=1000
#     )
    
#     plot_roc_curve(predict_df.to_pandas(), f"{context.artifact_output_path}/roc_curve")
    
#     feature_importance = compute_feature_importance(RF_classifier.modelObj,X_test)
#     plot_feature_importance(feature_importance, f"{context.artifact_output_path}/feature_importance")
    
    
#     predictions_table = "predictions_tmp"
#     copy_to_sql(df=predict_df, table_name=predictions_table, index=False, if_exists="replace", temporary=True)
    
    
#     # calculate stats if training stats exist
#     if os.path.exists(f"{context.artifact_input_path}/data_stats.json"):
#         record_evaluation_stats(
#             features_df=test_df,
#             predicted_df=DataFrame.from_query(f"SELECT * FROM {predictions_table}"),
#             feature_importance=feature_importance,
#             context=context
#         )

    print("All done!")
