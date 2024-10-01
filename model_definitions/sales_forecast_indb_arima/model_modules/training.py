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
    SeasonalNormalize,
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

    print ("Make series stationary using seasonalnormalize...")
    
    # Get the outliers training data using the OutlierFilterFit and Transform functions
    from teradataml import DickeyFuller
    data_series_df_1 = TDSeries(data=train_df,
                              id="idcols",
                              row_index=("Sales_Date"),
                              row_index_style= "TIMECODE",
                              payload_field="Weekly_Sales",
                              payload_content="REAL")
    print ("Before DickeyFuller...")
    df_out = DickeyFuller(   data=data_series_df_1,
                           algorithm='NONE')
    
    data_series_df_norm = TDSeries(data=train_df,
                              id="idcols",
                              row_index="Sales_Date",
                              row_index_style="TIMECODE",
                              payload_field="Weekly_Sales",
                              payload_content="REAL",
                              interval="WEEKS(1)")
    
    print ("Before seasonalnormalize...")
    uaf_out = SeasonalNormalize(data=data_series_df_norm,
                                season_cycle="WEEKS",
                                cycle_duration=1,
                                output_fmt_index_style = 'FLOW_THROUGH')
    print ("After seasonalnormalize...")
    uaf_out.result.to_sql('normalize_data', if_exists="replace")
    uaf_out.metadata.to_sql('normalize_metadata', if_exists="replace")
    # outlier_obj.result.to_sql('outlier_data', if_exists="replace")
    print("Saved normalized series")
    
    
        
    print("Starting training...")
    print("Before TDSeries...")
    data_series_df_2 = TDSeries(data=uaf_out.result,
                              id="idcols",
                              row_index=("ROW_I"),
                              row_index_style= "TIMECODE",
                              payload_field="Weekly_Sales",
                              payload_content="REAL")
    
  
    print("Before 2nd DickeyFuller...")
    df_out_norm = DickeyFuller(data=data_series_df_2,
                           algorithm='NONE')
    
    print(df_out_norm)
     # Calculate acf, pacf and generate plot
    print("Before acf pacf...")
    
    df_acf_plot, df_pacf_plot = compute_acf_pacf(data_series_df_2)
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
        print(data_series_df_2)
        arima_est_out = ArimaEstimate(data1=data_series_df_2,
                                nonseasonal_model_order=[int(context.hyperparams["p"]),int(context.hyperparams["d"]),int(context.hyperparams["q"])],
                                constant=False,
                                algorithm="MLE",
                                coeff_stats=True,
                                fit_metrics=True,
                                residuals=True,
                                persist=True,
                                output_table_name='arima_est_tb',     
                                fit_percentage=80,
                                output_fmt_index_style="FLOW_THROUGH")
    except:
        print("Before Arimaestimate...")
        print(data_series_df_2)
        db_drop_table('arima_est_tb')
        arima_est_out = ArimaEstimate(data1=data_series_df_2,
                                nonseasonal_model_order=[int(context.hyperparams["p"]),int(context.hyperparams["d"]),int(context.hyperparams["q"])],
                                constant=False,
                                algorithm="MLE",
                                coeff_stats=True,
                                fit_metrics=True,
                                residuals=True,
                                persist=True,
                                output_table_name='arima_est_tb',     
                                fit_percentage=80,
                                output_fmt_index_style="FLOW_THROUGH")
        
    # data_art_df = TDAnalyticResult(data=arima_est_out.result)
    # Save the trained model to SQL
    # arima_est_out.result.to_sql(table_name='arima_data', if_exists="replace")  
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
