from teradataml import (
    copy_to_sql,
    DataFrame,
    ArimaForecast,
    TDAnalyticResult,
    TDSeries,
    ArimaEstimate,
    Unnormalize
)
from aoa import (
    record_scoring_stats,
    aoa_create_context,
    ModelContext
)
import pandas as pd

def plot_sales(arima_val_out, img_filename):
    from teradataml import Figure
    val_result=arima_val_out.fitresiduals
    # val_result = val_result.groupby('ROW_I').avg()
    figure = Figure(width=1000, height=700, heading="Comparison of Actual vs Predicted")
    plot = val_result.plot(x=val_result.ROW_I, y=[val_result.ACTUAL_VALUE, val_result.CALC_VALUE], 
             style=['dark orange', 'green'], xlabel='Sales Date', ylabel='Sales',  grid_color='black',
            xtick_format='YYYY-MM',grid_linewidth=0.5, grid_linestyle="-", legend=['Actual Value','Predicted Value'],
                           figure=figure)
    plot.save(img_filename)


def plot_forecast(forecast_result, img_filename):
    from teradataml import Figure
    # forecast_result_plot = forecast_result.groupby('ROW_I').avg()
    forecast_result_plot = forecast_result
    figure = Figure(width=1000, height=700, heading="Forecast Sales")

    plot = forecast_result_plot.plot(x=forecast_result_plot.ROW_I, y=forecast_result_plot.FORECAST_VALUE, 
              xlabel='Forecast Period', ylabel='Forecast Sales',  grid_color='black',xtick_format='YYYY-MM',
                   grid_linewidth=0.5, grid_linestyle="-", figure=figure)
    plot.save(img_filename)

def score(context: ModelContext, **kwargs):
    
    aoa_create_context()

    # Load the trained model from SQL
    # model = DataFrame(f"model_${context.model_version}")
    df1 = DataFrame('normalize_data')
    data_series_df_1 = TDSeries(data=df1,
                              id="idcols",
                              row_index=("ROW_I"),
                              row_index_style= "TIMECODE",
                              payload_field="Weekly_Sales",
                              payload_content="REAL")
    arima_est_out = DataFrame('arima_est_tb')

      
    print("Forecasting...")
    # Make forecast using ArimaForecast function
    arima_estimate_op = ArimaEstimate(data1=data_series_df_1,
                                      nonseasonal_model_order=[context.hyperparams["p"],context.hyperparams["d"],context.hyperparams["q"]],
                                      constant=False,
                                      algorithm="MLE",
                                      coeff_stats=True,
                                      fit_metrics=True,
                                      residuals=True,
                                      fit_percentage=100,
                                output_fmt_index_style="FLOW_THROUGH")
    print(arima_estimate_op.fitresiduals)
    plot_sales(arima_estimate_op, f"{context.artifact_output_path}/actual_cal_tot_sales")
    data_art_df = TDAnalyticResult(data=arima_estimate_op.result)
 
    arima_forcast_out = ArimaForecast(data=data_art_df, forecast_periods=7,
                                output_fmt_index_style="FLOW_THROUGH")
    # forecast_result=arima_forcast_out.result
    # forecast_result = forecast_result.groupby('ROW_I').avg()
    print("Finished Forecasting")
    print("Unnormalize Data")
    
    df_metadata = DataFrame('normalize_metadata')
    df_metadata2 = df_metadata.assign(drop_columns = False,
                                           mean2 = df_metadata.MEAN_Weekly_Sales,
                                           sd2 = df_metadata.SD_Weekly_Sales,
                                           mean3 = df_metadata.MEAN_Weekly_Sales,
                                           sd3 = df_metadata.SD_Weekly_Sales,
                                           mean4 = df_metadata.MEAN_Weekly_Sales,
                                           sd4 = df_metadata.SD_Weekly_Sales,
                                           mean5 = df_metadata.MEAN_Weekly_Sales,
                                           sd5 = df_metadata.SD_Weekly_Sales)
    print(df_metadata2)
    #Create teradataml TDSeries objects.
    td_series_forecast = TDSeries(data=arima_forcast_out.result,
                          id="idcols",
                          row_index="ROW_I",
                          row_index_style="TIMECODE",
                          payload_field=["FORECAST_VALUE","LO_80","HI_80","LO_95","HI_95"],
                          payload_content="MULTIVAR_REAL",
                          interval="WEEKS(1)"
                          )
    
    td_series_metadata_forecast = TDSeries(data=df_metadata2, #from the seasonlized series
                          id="idcols",
                          row_index="ROW_I",
                          row_index_style="SEQUENCE",
                          payload_field=["MEAN_Weekly_Sales", "SD_Weekly_Sales","mean2","sd2",
                                        "mean3","sd3","mean4","sd4","mean5","sd5"],
                          payload_content="MULTIVAR_REAL"                  
                          )
    
    forecast_unnormalize = Unnormalize(data1=td_series_forecast,
                      data2=td_series_metadata_forecast,
                         input_fmt_input_mode="MATCH",
                                   output_fmt_index_style="FLOW_THROUGH")
    df_forecast_un = forecast_unnormalize.result
    
    print(df_forecast_un)
    # store the Forecast
    copy_to_sql(
        df=df_forecast_un,
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
    plot_forecast(df_forecast_un, f"{context.artifact_output_path}/forecast_sales")
    
    
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
