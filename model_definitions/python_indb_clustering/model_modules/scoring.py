from teradataml import td_sklearn as osml
from teradataml import (
    copy_to_sql,
    DataFrame,
    ScaleTransform,
    execute_sql,
    db_drop_table
)
from aoa import (
    record_scoring_stats,
    aoa_create_context,
    ModelContext
)
import pandas as pd
import dill
import json
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def score(context: ModelContext, **kwargs):
    
    aoa_create_context()

    
    # Extract feature names, target name, and entity key from the context
    feature_names = context.dataset_info.feature_names
    # target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key
    
    # Load the test dataset
    # test_df = DataFrame.from_query(context.dataset_info.sql)
    # copy_to_sql(
    #     df=test_df,
    #     schema_name=context.dataset_info.predictions_database,
    #     table_name='test_df',
    #     index=False,
    #     if_exists="replace"
    # )
    # X_test = test_df.drop(['WELDING_ID'], axis = 1)
    # # y_test = test_df.select(["anomaly_int"])
    # features_tdf = DataFrame.from_query(context.dataset_info.sql)
    # features_pdf = features_tdf.to_pandas(all_rows=True)
    # test_df.set_index("WELDING_ID")
#     # Scaling the test set
#     print ("Loading scaler...")
#     scaler = DataFrame(f"scaler_${context.model_version}")

#     # Scale the test dataset using the trained scaler
#     scaled_test = ScaleTransform(
#         data=test_df,
#         object=scaler,
#         accumulate = entity_key
#     )
    
      
    # print(predictions_pdf)
    print("Scoring using osml...")
    
    qry = '''Create table KMeans_TestPredict_Output AS (
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
        db_drop_table("KMeans_TestPredict_Output")
        execute_sql(qry)
    
    print("Finished Scoring")
    # print(predictions_pdf.columns)
   
     # with open(f"{context.artifact_input_path}/exp_obj", 'rb') as f:
#         explainer = dill.load(f)
#     pred_df = pd.DataFrame(columns=["WELDING_ID","json_report"])
#     convert_dict = {'WELDING_ID': int,
#                 'json_report': str
#                 }
 
#     pred_df = pred_df.astype(convert_dict)
#     print("Starting prediction explainer for all rows...")
#     for i in range(len(test_df.to_pandas())):
#         df = test_df.iloc[i, :]
#         welding_id = df.select(["WELDING_ID"]).get_values().flatten()
#         # print("WELDING_ID", welding_id)
#         df = df.drop(columns=["WELDING_ID"])
        
#         exp = explainer.explain_instance(df.get_values().flatten(), RF_classifier.modelObj.predict_proba, num_features=9)
#         # print("explisttype",type(json.dumps(exp.as_list())), json.dumps(exp.as_list()))
#         new_row = pd.DataFrame({"WELDING_ID": welding_id,"json_report":json.dumps(exp.as_list())})
#         # print("new_row", new_row)
#         pred_df = pd.concat([pred_df, new_row], ignore_index=True, axis=0)
   

    # store the predictions
   
    prediction_df = DataFrame("KMeans_TestPredict_Output")
    predictions_pdf = prediction_df.assign(job_id = context.job_id)
    print(predictions_pdf)

        
    copy_to_sql(
        df=predictions_pdf,
        schema_name=context.dataset_info.predictions_database,
        table_name=context.dataset_info.predictions_table,
        index=False,
        if_exists="append"
    )
#     final_pred_df = pred_df.merge(right=predictions_pdf, how = 'inner', on="WELDING_ID")
    
#     final_pred_df.rename(columns={'json_report_x':'explainer_variables'},inplace = True)
#     final_pred_df.drop(['json_report_y'], axis=1, inplace=True)
#     # print(final_pred_df)
#     copy_to_sql(
#         df=final_pred_df,
#         schema_name=context.dataset_info.predictions_database,
#         table_name=context.dataset_info.predictions_table,
#         index=False,
#         if_exists="replace"
#     )
        
    print("Saved predictions in Teradata")

    # calculate stats
    predictions_df = DataFrame.from_query(f"""
        SELECT 
            * 
        FROM {context.dataset_info.get_predictions_metadata_fqtn()} 
            WHERE job_id = '{context.job_id}'
    """)
    
    # record_scoring_stats(features_df=features_tdf, predicted_df=predictions_df, context=context)

    print("All done!")
