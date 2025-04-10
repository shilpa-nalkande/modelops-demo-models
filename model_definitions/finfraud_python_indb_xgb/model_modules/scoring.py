from teradataml import (
    copy_to_sql,
    DataFrame,
    XGBoostPredict,
    ConvertTo,
    translate
)
from aoa import (
    record_scoring_stats,
    aoa_create_context,
    ModelContext
)
import pandas as pd
from teradatasqlalchemy import INTEGER


def score(context: ModelContext, **kwargs):
    
    aoa_create_context()

    # Load the trained model from SQL
    model = DataFrame(f"model_${context.model_version}")

    # Extract feature names, target name, and entity key from the context
    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    # Load the test dataset
    test_df = DataFrame.from_query(context.dataset_info.sql)
    features_tdf = DataFrame.from_query(context.dataset_info.sql)

    print("Scoring...")
    # Make predictions using the XGBoostPredict function
    predictions = XGBoostPredict(
                            newdata=test_df,
                            object=model,
                            model_type='Classification',
                            id_column='txn_id',
                            object_order_column=['task_index', 'tree_num',
                                               'iter', 'tree_order'],
                            # accumulate='isFraud',
                            output_prob=True,
                            output_responses=['0', '1']
                        )
    
    # Convert predictions to pandas DataFrame and process
    # predictions_pdf = predictions.result.to_pandas(all_rows=True).rename(columns={"Prediction": target_name}).astype(int)
    predictions_df = predictions.result
    # print(predictions_df)
    predictions_pdf = predictions_df.assign(drop_columns=True,
                                             job_id=translate(context.job_id),
                                             txn_id=predictions_df.txn_id,
                                             isFraud=predictions_df.Prediction.cast(type_=INTEGER),
                                             json_report=translate("  "))
                                             
    
    
    # converted_data = ConvertTo(data = predictions_pdf,
    #                            target_columns = ['job_id','PatientId', 'HasDiabetes','json_report'],
    #                            target_datatype = ["VARCHAR(charlen=255,charset=LATIN,casespecific=NO)"
    #                                               ,"integer","integer","VARCHAR(charlen=5000,charset=LATIN)"])
    # df=converted_data.result
    
    # print(predictions_pdf)
    print("Finished Scoring")
    # print(predictions_pdf)

    # store the predictions

#     # teradataml doesn't match column names on append.. and so to match / use same table schema as for byom predict
#     # example (see README.md), we must add empty json_report column and change column order manually (v17.0.0.4)
#     # CREATE MULTISET TABLE pima_patient_predictions
#     # (
#     #     job_id VARCHAR(255), -- comes from airflow on job execution
#     #     PatientId BIGINT,    -- entity key as it is in the source data
#     #     HasDiabetes BIGINT,   -- if model automatically extracts target
#     #     json_report CLOB(1048544000) CHARACTER SET UNICODE  -- output of
#     # )
#     # PRIMARY INDEX ( job_id );

    copy_to_sql(
        df=predictions_pdf,
        schema_name=context.dataset_info.predictions_database,
        table_name=context.dataset_info.predictions_table,
        index=False,
        if_exists="append"
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
