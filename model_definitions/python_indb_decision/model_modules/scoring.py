from teradataml import td_sklearn as osml
from teradataml import (
    copy_to_sql,
    DataFrame,
    TDDecisionForestPredict,
    ScaleTransform
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

    # Load the trained model from SQL
    # model = DataFrame(f"model_${context.model_version}")

    # Extract feature names, target name, and entity key from the context
    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key
    
    # Load the test dataset
    test_df = DataFrame.from_query(context.dataset_info.sql)
    copy_to_sql(
        df=test_df,
        schema_name=context.dataset_info.predictions_database,
        table_name='test_df',
        index=False,
        if_exists="replace"
    )
    X_test = test_df.drop(['WELDING_ID'], axis = 1)
    # y_test = test_df.select(["anomaly_int"])
    features_tdf = DataFrame.from_query(context.dataset_info.sql)
    features_pdf = features_tdf.to_pandas(all_rows=True)
    test_df.set_index("WELDING_ID")
    # Scaling the test set
    print ("Loading scaler...")
    scaler = DataFrame(f"scaler_${context.model_version}")

    # Scale the test dataset using the trained scaler
    scaled_test = ScaleTransform(
        data=test_df,
        object=scaler,
        accumulate = entity_key
    )
    
      
    # print(predictions_pdf)
    print("Scoring using osml...")
    DT_classifier = osml.load(model_name="DT_classifier")
    predict_DT =DT_classifier.predict(X_test)
    # Convert predictions to pandas DataFrame and process
    # predictions_pdf = predict_DT.to_pandas(all_rows=True)
    df_pred = predict_DT.to_pandas(all_rows=True)
    
    predictions_pdf = predict_DT.to_pandas(all_rows=True).rename(columns={"decisiontreeclassifier_predict_1": target_name})
    print("Finished Scoring")
    # print(predictions_pdf.columns)
   
    with open(f"{context.artifact_input_path}/exp_obj", 'rb') as f:
        explainer = dill.load(f)
    pred_df = pd.DataFrame(columns=["WELDING_ID","json_report"])
    convert_dict = {'WELDING_ID': int,
                'json_report': str
                }
 
    pred_df = pred_df.astype(convert_dict)
    print("Starting prediction explainer for all rows...")
    for i in range(len(test_df.to_pandas())):
        df = test_df.iloc[i, :]
        welding_id = df.select(["WELDING_ID"]).get_values().flatten()
        # print("WELDING_ID", welding_id)
        df = df.drop(columns=["WELDING_ID"])
        
        exp = explainer.explain_instance(df.get_values().flatten(), DT_classifier.modelObj.predict_proba, num_features=9)
        # print("explisttype",type(json.dumps(exp.as_list())), json.dumps(exp.as_list()))
        new_row = pd.DataFrame({"WELDING_ID": welding_id,"json_report":json.dumps(exp.as_list())})
        # print("new_row", new_row)
        pred_df = pd.concat([pred_df, new_row], ignore_index=True, axis=0)
        # pred_df.append({"json_report":json.dumps(explainer.explain_instance(df.get_values().flatten(), DT_classifier.modelObj.predict_proba, num_features=9).as_list())})
    # store the predictions
   
    predictions_pdf = pd.DataFrame(predictions_pdf, columns=[target_name])
    # predictions_pdf[entity_key] = features_pdf.index.values
    predictions_pdf[entity_key] = test_df.select(["WELDING_ID"]).get_values()
    # add job_id column so we know which execution this is from if appended to predictions table
    # print(predictions_pdf)
    predictions_pdf["job_id"] = context.job_id

    # teradataml doesn't match column names on append.. and so to match / use same table schema as for byom predict
    # example (see README.md), we must add empty json_report column and change column order manually (v17.0.0.4)
    # CREATE MULTISET TABLE pima_patient_predictions
    # (
    #     job_id VARCHAR(255), -- comes from airflow on job execution
    #     PatientId BIGINT,    -- entity key as it is in the source data
    #     HasDiabetes BIGINT,   -- if model automatically extracts target
    #     json_report CLOB(1048544000) CHARACTER SET UNICODE  -- output of
    # )
    # PRIMARY INDEX ( job_id );
    
    predictions_pdf["json_report"] = ""
    predictions_pdf = predictions_pdf[["job_id", entity_key, target_name, "json_report"]]
    
    copy_to_sql(
        df=predictions_pdf,
        schema_name=context.dataset_info.predictions_database,
        table_name='predictions_pdf',
        index=False,
        if_exists="replace"
    )
    final_pred_df = pred_df.merge(right=predictions_pdf, how = 'inner', on="WELDING_ID")
    
    final_pred_df.rename(columns={'json_report_x':'explainer_variables'},inplace = True)
    final_pred_df.drop(['json_report_y'], axis=1, inplace=True)
    # print(final_pred_df)
    copy_to_sql(
        df=final_pred_df,
        schema_name=context.dataset_info.predictions_database,
        table_name=context.dataset_info.predictions_table,
        index=False,
        if_exists="replace"
    )
        
    print("Saved predictions in Teradata")

    # calculate stats
    predictions_df = DataFrame.from_query(f"""
        SELECT 
            * 
        FROM {context.dataset_info.get_predictions_metadata_fqtn()} 
            WHERE job_id = '{context.job_id}'
    """)
    
    record_scoring_stats(features_df=features_tdf, predicted_df=predictions_df, context=context)

    print("All done!")
