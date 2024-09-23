from teradataml import (
    copy_to_sql,
    DataFrame,
    ArimaForecast,
    TDAnalyticResult,
    TDSeries,
    ArimaEstimate
)
from aoa import (
    record_scoring_stats,
    aoa_create_context,
    ModelContext
)
import pandas as pd

def plot_forecast(forecast_result, img_filename):
    from teradataml import Figure
    forecast_result_plot = forecast_result.groupby('ROW_I').avg()
    figure = Figure(width=1000, height=700, heading="Forecast Sales")

    plot = forecast_result_plot.plot(x=forecast_result_plot.ROW_I, y=forecast_result_plot.avg_FORECAST_VALUE, 
              xlabel='Forecast Period', ylabel='Forecast Sales',  grid_color='black',
                   grid_linewidth=0.5, grid_linestyle="-", figure=figure)
    plot.save(img_filename)

def score(context: ModelContext, **kwargs):
    
    aoa_create_context()

    # Load the trained model from SQL
    # model = DataFrame(f"model_${context.model_version}")
    df1 = DataFrame('arima_data')
    data_series_df_1 = TDSeries(data=df1,
                              id="Sales_Date",
                              row_index=("idcols"),
                              row_index_style= "SEQUENCE",
                              payload_field="Weekly_Sales",
                              payload_content="REAL")
    arima_est_out = DataFrame('arima_est_tb')

      
    print("Forecasting...")
    # Make forecast using ArimaForecast function
    arima_estimate_op = ArimaEstimate(data1=data_series_df_1,
                                      nonseasonal_model_order=[2,1,1],
                                      constant=False,
                                      algorithm="MLE",
                                      coeff_stats=True,
                                      fit_metrics=True,
                                      residuals=True,
                                      fit_percentage=100)
    data_art_df = TDAnalyticResult(data=arima_estimate_op.result)
 
    arima_forcast_out = ArimaForecast(data=data_art_df, forecast_periods=7)
    forecast_result=arima_forcast_out.result
    # forecast_result = forecast_result.groupby('ROW_I').avg()
    print("Finished Forecasting")

    # store the Forecast
    copy_to_sql(
        df=forecast_result,
        # schema_name=context.dataset_info.predictions_database,
        # table_name=context.dataset_info.predictions_table,
        table_name='arima_forecast',
        index=False,
        if_exists="replace"
    )

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
    plot_forecast(forecast_result, f"{context.artifact_output_path}/forecast_sales")
    
    
    print("Saved Forecasts in Teradata")

    # calculate stats
#     predictions_df = DataFrame.from_query(f"""
#         SELECT 
#             * 
#         FROM {context.dataset_info.get_predictions_metadata_fqtn()} 
#             WHERE job_id = '{context.job_id}'
#     """)

#     record_scoring_stats(features_df=features_tdf, predicted_df=predictions_df, context=context)

    print("All done!")
