from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from teradataml import(
    DataFrame, 
    copy_to_sql, 
    get_context, 
    get_connection, 
    ArimaValidate,
    TDAnalyticResult
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


    
# Define function to plot Actual vs Calculated Sales
def plot_sales(arima_val_out, img_filename):
    from teradataml import Figure
    val_result=arima_val_out.fitresiduals
    val_result = val_result.groupby('Sales_Date').avg()
    figure = Figure(width=1000, height=700, heading="Comparison of Actua vs Predicted")
    plot = val_result.plot(x=val_result.Sales_Date, y=[val_result.avg_ACTUAL_VALUE, val_result.avg_CALC_VALUE], 
             style=['dark orange', 'green'], xlabel='Sales Date', ylabel='Sales',  grid_color='black',
                   grid_linewidth=0.5, grid_linestyle="-", legend=['Actual Value','Predicted Value'],figure=figure)
    plot.save(img_filename)



def evaluate(context: ModelContext, **kwargs):

    aoa_create_context()

    # Load the ArimaEstima
    # arima_est_out = DataFrame(f"model_${context.model_version}")
    arima_est_out = DataFrame('arima_est_tb')

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    
    print("Validating...")
    # Make predictions using the XGBoostPredict function
    
    data_art_df = TDAnalyticResult(data=arima_est_out)
    
    arima_val_out = ArimaValidate(data=data_art_df, fit_metrics=True, residuals=True)
    plot_sales(arima_val_out, f"{context.artifact_output_path}/actual_cal_sales")

#     # calculate stats if training stats exist
#     if os.path.exists(f"{context.artifact_input_path}/data_stats.json"):
#         record_evaluation_stats(
#             features_df=test_df,
#             predicted_df=DataFrame.from_query(f"SELECT * FROM {predictions_table}"),
#             feature_importance=feature_importance,
#             context=context
#         )

    print("All done!")
