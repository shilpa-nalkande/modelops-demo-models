from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from teradataml import(
    DataFrame, 
    copy_to_sql, 
    get_context, 
    get_connection, 
    ScaleTransform, 
    XGBoostPredict, 
    ConvertTo, 
    ClassificationEvaluator,
    ROC,
    Shap
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


# Define function to plot a confusion matrix from given data
def plot_confusion_matrix(cf, img_filename):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cf, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cf.shape[0]):
        for j in range(cf.shape[1]):
            ax.text(x=j, y=i,s=cf[i, j], va='center', ha='center', size='xx-large')
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix');
    fig = plt.gcf()
    fig.savefig(img_filename, dpi=500)
    plt.clf()

    
def plot_roc_curve(roc_out, img_filename):
    from teradataml import Figure
    figure = Figure(width=500, height=400, heading="Receiver Operating Characteristic (ROC) Curve")
    auc = roc_out.result.get_values()[0][0]
    plot = roc_out.output_data.plot(
        x=roc_out.output_data.fpr,
        y=[roc_out.output_data.tpr, roc_out.output_data.fpr],
        xlabel='False Positive Rate',
        ylabel='True Positive Rate',
        color='carolina blue',
        figure=figure,
        legend=[f'XGBoost AUC = {round(auc, 4)}', 'AUC Baseline'],
        legend_style='lower right',
        grid_linestyle='--',
        grid_linewidth=0.5
    )
    plot.save(img_filename)
    # plot.show()
    # fig = plt.gcf()
    # fig.savefig(img_filename, dpi=500)
    # plt.clf()    

def evaluate(context: ModelContext, **kwargs):

    aoa_create_context()

    # Load the trained model from SQL
    model = DataFrame(f"model_${context.model_version}")

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    # Load the test data from Teradata
    test_df = DataFrame.from_query(context.dataset_info.sql)

    # Scaling the test set
    print ("Loading scaler...")
    scaler = DataFrame(f"scaler_${context.model_version}")

    scaled_test = ScaleTransform(
        data=test_df,
        object=scaler,
        accumulate = [target_name,entity_key]
    )
    
    print("Evaluating...")
    # Make predictions using the XGBoostPredict function
    predictions = XGBoostPredict(
        object=model,
        newdata=scaled_test.result,
        model_type = 'Classification',
        accumulate=target_name,
        id_column=entity_key,
        output_prob=True,
        output_responses=['0','1'],
        object_order_column=['task_index', 'tree_num', 'iter', 'class_num', 'tree_order']
    )

    # Convert the predicted data into the specified format
    predicted_data = ConvertTo(
        data = predictions.result,
        target_columns = [target_name,'Prediction'],
        target_datatype = ["INTEGER"]
    )

    # Evaluate classification metrics using ClassificationEvaluator
    ClassificationEvaluator_obj = ClassificationEvaluator(
        data=predicted_data.result,
        observation_column=target_name,
        prediction_column='Prediction',
        num_labels=2
    )

     # Extract and store evaluation metrics
        
    metrics_pd = ClassificationEvaluator_obj.output_data.to_pandas()

    evaluation = {
        'Accuracy': '{:.2f}'.format(metrics_pd.MetricValue[0]),
        'Micro-Precision': '{:.2f}'.format(metrics_pd.MetricValue[1]),
        'Micro-Recall': '{:.2f}'.format(metrics_pd.MetricValue[2]),
        'Micro-F1': '{:.2f}'.format(metrics_pd.MetricValue[3]),
        'Macro-Precision': '{:.2f}'.format(metrics_pd.MetricValue[4]),
        'Macro-Recall': '{:.2f}'.format(metrics_pd.MetricValue[5]),
        'Macro-F1': '{:.2f}'.format(metrics_pd.MetricValue[6]),
        'Weighted-Precision': '{:.2f}'.format(metrics_pd.MetricValue[7]),
        'Weighted-Recall': '{:.2f}'.format(metrics_pd.MetricValue[8]),
        'Weighted-F1': '{:.2f}'.format(metrics_pd.MetricValue[9]),
    }

     # Save evaluation metrics to a JSON file
    with open(f"{context.artifact_output_path}/metrics.json", "w+") as f:
        json.dump(evaluation, f)
        
    # Generate and save confusion matrix plot
    cm_df = ClassificationEvaluator_obj.result
    cm_df = cm_df.select(['CLASS_1','CLASS_2'])
    print(cm_df.get_values())
    # cm = confusion_matrix(predicted_data.result.to_pandas()['HasDiabetes'], predicted_data.result.to_pandas()['Prediction'])
    # print(cm)
    plot_confusion_matrix(cm_df.get_values(), f"{context.artifact_output_path}/confusion_matrix")

    # Generate and save ROC curve plot
    roc_out = ROC(
        data=predictions.result,
        probability_column='Prob_1',
        observation_column=target_name,
        positive_class='1',
        num_thresholds=1000
    )
    plot_roc_curve(roc_out, f"{context.artifact_output_path}/roc_curve")

    # Calculate feature importance and generate plot
    # try:
    #     model_pdf = model.result.to_pandas()['classification_tree']
    #     feature_importance = compute_feature_importance(model_pdf)
    #     feature_importance_df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance'])
    #     plot_feature_importance(feature_importance, f"{context.artifact_output_path}/feature_importance")
    # except:
    #     feature_importance = {}

    predictions_table = "predictions_tmp"
    copy_to_sql(df=predicted_data.result, table_name=predictions_table, index=False, if_exists="replace", temporary=True)

    # calculate stats if training stats exist
    if os.path.exists(f"{context.artifact_input_path}/data_stats.json"):
        record_evaluation_stats(
            features_df=test_df,
            predicted_df=DataFrame.from_query(f"SELECT * FROM {predictions_table}"),
            # feature_importance=feature_importance,
            context=context
        )

    print("All done!")
