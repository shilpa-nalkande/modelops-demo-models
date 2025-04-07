from teradataml import (
    copy_to_sql,
    DataFrame,
    TDDecisionForestPredict,
    OrdinalEncodingFit,
    ScaleFit,
    ColumnTransformer,
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
    predictions = TDDecisionForestPredict(object = model,
                                        newdata = test_df,
                                        id_column = "CustomerID",
                                        detailed = False,
                                        output_prob = True,
                                        output_responses = ['0','1'])
    
    # Convert predictions to pandas DataFrame and process
    # predictions_pdf = predictions.result.to_pandas(all_rows=True).rename(columns={"Prediction": target_name}).astype(int)
    predictions_df = predictions.result
    # print(predictions_df)
    predictions_pdf = predictions_df.assign(drop_columns=True,
                                             job_id=translate(context.job_id),
                                             CustomerID=predictions_df.CustomerID,
                                             Churn=predictions_df.prediction.cast(type_=INTEGER),
                                             json_report=translate("  "))
                                             
    
    
    # print(predictions_pdf)
    print("Finished Scoring")
    # print(predictions_pdf)

    # store the predictions

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

    record_scoring_stats(features_df=features_tdf, predicted_df=predictions_pdf, context=context)

    print("All done!")
