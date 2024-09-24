from teradataml import (
    DataFrame,
    ArimaEstimate,
    ArimaValidate,
    ArimaForecast,
    TDAnalyticResult,
    OutlierFilterFit,
    OutlierFilterTransform,
    Resample,
    DickeyFuller,
    copy_to_sql,
    execute_sql,
    ACF,PACF, TDSeries, db_drop_table
    
    
)

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
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# Compute acf and pacf 
def compute_acf_pacf(data_series_df):
    print("inside acf pacf...")
    print(data_series_df)
    acf_out = ACF(data=data_series_df,
                  max_lags=12,
                  demean=True,
                  qstat = False,
                  alpha=0.05)
    df_acf_plot = acf_out.result
    pacf_out = PACF(data=data_series_df,
                    algorithm='LEVINSON_DURBIN',
                    max_lags=12,
                    alpha=0.05)
    df_pacf_plot = pacf_out.result
    return df_acf_plot, df_pacf_plot


def plot_acf_fun(img_filename):
    from teradataml import Figure
    figure = Figure(width=800, height=900, image_type="png", heading="Auto Correlation")
    print("ACF plots...")
    df_acf_plot=DataFrame('acf_data')
    print(df_acf_plot)
    plot = df_acf_plot.plot(x=df_acf_plot.ROW_I, 
        y=(df_acf_plot.OUT_Weekly_Sales, df_acf_plot.CONF_OFF_Weekly_Sales),
        kind='corr', ylabel = " ", color="blue", figure=figure)
    plot.save(img_filename)
    # plot.clf()
    
# def plot_acf_fun(data_series_df, img_filename):
#     from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#     df_plot = data_series_df.to_pandas()
#     plot_acf(df_plot["Weekly_Sales"], lags=12)
#     plt.gcf()
#     plt.savefig(img_filename, dpi=500)
#     plt.clf()
        
def plot_pacf_fun(img_filename):
    from teradataml import Figure
    figure = Figure(width=800, height=900, image_type="png", heading="Auto Correlation")
    print("PACF plots...")
    df_pacf_plot=DataFrame('pacf_data')
    print(df_pacf_plot)
    plot = df_pacf_plot.plot(x=df_pacf_plot.ROW_I, 
        y=(df_pacf_plot.OUT_Weekly_Sales, df_pacf_plot.CONF_OFF_Weekly_Sales),
        kind='corr',figsize=(600,400),ylabel = " ",
        color="blue",title="Partial Auto Correlation")
    # fig = plot.gcf()
    plot.save(img_filename)
    # plot.clf()    

# def plot_pacf_fun(data_series_df, img_filename):
#     from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#     df_plot = data_series_df.to_pandas()
#     plot_pacf(df_plot["Weekly_Sales"], lags=12)
#     plt.gcf()
#     plt.savefig(img_filename, dpi=500)
#     plt.clf()  


def train(context: ModelContext, **kwargs):
    aoa_create_context()
    
    # Extracting feature names, target name, and entity key from the context
    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    # Load the training data from Teradata
    train_df = DataFrame.from_query(context.dataset_info.sql)

    print ("Outliers using InDB Functions...")
    
    # Get the outliers training data using the OutlierFilterFit and Transform functions
    OutlierFilterFit_out = OutlierFilterFit(data = train_df,
                                            target_columns = "Weekly_Sales",
                                               )
    outlier_obj = OutlierFilterTransform(data=train_df,
                                 object=OutlierFilterFit_out.result)
    out_transform_df = outlier_obj.result
    
    OutlierFilterFit_out.result.to_sql(f"outlier_${context.model_version}", if_exists="replace")
    outlier_obj.result.to_sql('outlier_data', if_exists="replace")
    print("Saved Outliers")
    
    print("Starting training...")
#     data_series_df = TDSeries(data=outlier_obj.result,
#                               id="idcols",
#                               row_index=("Sales_Date"),
#                               row_index_style= "TIMECODE",
#                               payload_field="Weekly_Sales",
#                               payload_content="REAL")
    
#     print("Before Resample...")
#     print(data_series_df)
#     # data_series_df.to_sql('series_data', if_exists='replace')
#     uaf_out1 = Resample(data=data_series_df,
#                         interpolate='LINEAR',
#                         timecode_start_value="TIMESTAMP '2010-02-05 00:00:00'",
#                         timecode_duration="WEEKS(1)")
#     print(uaf_out1.result)
#     qry='''EXECUTE FUNCTION INTO ART(resample_art)
#             TD_RESAMPLE
#             (
#                 SERIES_SPEC(
#                     TABLE_NAME(outlier_data),
#                     SERIES_ID(idcols),
#                     ROW_AXIS(TIMECODE(Sales_Date)),
#                     PAYLOAD(
#                         FIELDS(Weekly_Sales),
#                         CONTENT(REAL)
#                     )
#                 ),
#                 FUNC_PARAMS(
#                     TIMECODE(
#                         START_VALUE(TIMESTAMP '2010-02-05 00:00:00'), 
#                         DURATION(WEEKS(1))
#                     ),
#                     INTERPOLATE(LINEAR)
#                 )
#             );'''
#     try:
#         execute_sql(qry)
#     except:
#         db_drop_table('resample_art')
#         execute_sql(qry)
#     print("After Resample...")
   
#     df1=DataFrame('resample_art')
#     df1=df1.select(['idcols','ROW_I', 'Weekly_Sales']).assign(Sales_Date=df1.ROW_I)
#     print(df1)
#     df1.to_sql('arima_data', if_exists="replace")
    outlier_obj.result.to_sql('arima_data', if_exists="replace")
    # Check if the series is stationary using DickeyFuller
    print("Before TDSeries...")
    data_series_df_1 = TDSeries(data=outlier_obj.result,
                              id="Sales_Date",
                              row_index=("idcols"),
                              row_index_style= "SEQUENCE",
                              payload_field="Weekly_Sales",
                              payload_content="REAL")
    data_series_df_1.to_sql('series_data_1', if_exists="replace")
    print(DataFrame('series_data_1'))
    print("Before DickeyFuller...")
    df_out = DickeyFuller(   data=data_series_df_1,
                           algorithm='NONE')
    
    print(df_out)
     # Calculate acf, pacf and generate plot
    print("Before acf pacf...")
    
    df_acf_plot, df_pacf_plot = compute_acf_pacf(data_series_df_1)
    # print(df_acf_plot, df_pacf_plot)
    # print(df_acf_plot)
    # print(df_pacf_plot)
    print("Before plots...")
    copy_to_sql(df=df_acf_plot, table_name='acf_data', if_exists='replace')
    copy_to_sql(df=df_pacf_plot, table_name='pacf_data', if_exists='replace')
    
    plot_acf_fun(f"{context.artifact_output_path}/acf_plot")
    plot_pacf_fun(f"{context.artifact_output_path}/pacf_plot")
    # Train the model using ARIMA
    try:
        print("Before Arimaestimate...")
        arima_est_out = ArimaEstimate(data1=data_series_df_1,
                                nonseasonal_model_order=[int(context.hyperparams["p"]),int(context.hyperparams["d"]),int(context.hyperparams["q"])],
                                constant=False,
                                algorithm="MLE",
                                coeff_stats=True,
                                fit_metrics=True,
                                residuals=True,
                                persist=True,
                                output_table_name='arima_est_tb',     
                                fit_percentage=80)
    except:
        print("Before Arimaestimate...")
        db_drop_table('arima_est_tb')
        arima_est_out = ArimaEstimate(data1=data_series_df_1,
                                nonseasonal_model_order=[int(context.hyperparams["p"]),int(context.hyperparams["d"]),int(context.hyperparams["q"])],
                                constant=False,
                                algorithm="MLE",
                                coeff_stats=True,
                                fit_metrics=True,
                                residuals=True,
                                persist=True,
                                output_table_name='arima_est_tb',     
                                fit_percentage=80)
        
    # data_art_df = TDAnalyticResult(data=arima_est_out.result)
    # Save the trained model to SQL
    # arima_est_out.result.to_sql(table_name='arima_est', if_exists="replace")  
    print("Saved trained model")

    # Calculate feature importance and generate plot
    # modeldata_art_df_pdf = model.result.to_pandas()['classification_tree']
    # feature_importance = compute_feature_importance(model_pdf)
    # plot_feature_importance(feature_importance, f"{context.artifact_output_path}/feature_importance")
    

    # record_training_stats(
    #     train_df,
    #     features=feature_names,
    #     targets=[target_name],
    #     categorical=[target_name],
    #     # feature_importance=feature_importance,
    #     context=context
    # )
    
    print("All done!")
