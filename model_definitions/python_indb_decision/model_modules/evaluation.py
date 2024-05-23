from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from teradataml import td_sklearn as osml
from lime.lime_tabular import LimeTabularExplainer
from teradataml import(
    DataFrame, 
    copy_to_sql, 
    get_context, 
    get_connection, 
    ScaleTransform, 
    TDDecisionForestPredict, 
    ConvertTo, 
    ClassificationEvaluator,
    ROC
)
from aoa import (
    record_evaluation_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)

import joblib
import json
import dill
import numpy as np
import pandas as pd
import os


# Define function to traverse a decision tree and count feature occurrences
def traverse_tree(tree, feature_counter):
    if 'split_' in tree and 'attr_' in tree['split_']:
        feature_counter[tree['split_']['attr_']] += 1
    if 'leftChild_' in tree:
        traverse_tree(tree['leftChild_'], feature_counter)
    if 'rightChild_' in tree:
        traverse_tree(tree['rightChild_'], feature_counter)

        
# Define function to compute feature importance from tree structures
def compute_feature_importance(trees_json):
    feature_counter = Counter()
    for tree_json in trees_json:
        tree = json.loads(tree_json)
        traverse_tree(tree, feature_counter)
    total_splits = sum(feature_counter.values())
    feature_importance = {feature: count / total_splits for feature, count in feature_counter.items()}
    return feature_importance


# Define a function to plot feature importances
def plot_feature_importance(fi, img_filename):
    import pandas as pd
    import matplotlib.pyplot as plt
    feat_importances = pd.Series(fi)
    feat_importances.nlargest(10).plot(kind='barh').set_title('Feature Importance')
    fig = plt.gcf()
    fig.savefig(img_filename, dpi=500)
    plt.clf()


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


# Define function to plot ROC curve from ROC output data 
def plot_roc_curve(roc_out, img_filename):
    import matplotlib.pyplot as plt
    auc = roc_out.result.to_pandas().reset_index()['AUC'][0]
    roc_results = roc_out.output_data.to_pandas()
    plt.plot(roc_results['fpr'], roc_results['tpr'], color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % 0.27)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    fig = plt.gcf()
    fig.savefig(img_filename, dpi=500)
    plt.clf()
    
# Define function to plot ROC curve from ROC output data 
def dt_explain(exp_out, img_filename):
    # fig = exp_out.show_in_notebook(show_table=True)
    # fig.savefig(img_filename, dpi=500)
    # exp.as_list() 
    fig = exp_out.as_pyplot_figure().gcf()
    fig.savefig(img_filename, dpi=500)
    plt.clf()
    # Save evaluation metrics to a JSON file
    

def evaluate(context: ModelContext, **kwargs):

    aoa_create_context()

    # Load the trained model from SQL
    model = DataFrame(f"model_${context.model_version}")

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    # Load the test data from Teradata
    test_df = DataFrame.from_query(context.dataset_info.sql)
    X_test = test_df.drop(['anomaly_int','WELDING_ID'], axis = 1)
    y_test = test_df.select(["anomaly_int"])
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
    # predictions = XGBoostPredict(
    #     object=model,
    #     newdata=scaled_test.result,
    #     model_type = 'Classification',
    #     accumulate=target_name,
    #     id_column=entity_key,
    #     output_prob=True,
    #     output_responses=['0','1'],
    #     object_order_column=['task_index', 'tree_num', 'iter', 'class_num', 'tree_order']
    # )
    
    predictions = TDDecisionForestPredict(object = model,
                                        newdata = scaled_test.result,
                                        id_column = "WELDING_ID",
                                        detailed = False,
                                        output_prob = True,
                                        output_responses = ['0','1'],
                                        accumulate = 'anomaly_int')

    # Convert the predicted data into the specified format
    predicted_data = ConvertTo(
        data = predictions.result,
        target_columns = [target_name,'prediction'],
        target_datatype = ["INTEGER"]
    )

    # Evaluate classification metrics using ClassificationEvaluator
    ClassificationEvaluator_obj = ClassificationEvaluator(
        data=predicted_data.result,
        observation_column=target_name,
        prediction_column='prediction',
        num_labels=2
    )

     # Extract and store evaluation metrics
    metrics_pd = ClassificationEvaluator_obj.output_data.to_pandas()
    print("Evaluating osml...")
    DT_classifier = osml.load(model_name="DT_classifier")
    predict_DT =DT_classifier.predict(X_test,y_test)
    accuracy_DT = DT_classifier.score(X_test, y_test)
    df = X_test.sample(n=1)
    df = df.drop(columns="sampleid")
    with open(f"{context.artifact_output_path}/exp_obj", 'rb') as f:
        explainer = dill.load(f)
       
    exp = explainer.explain_instance(df.get_values().flatten(), DT_classifier.modelObj.predict_proba, num_features=9)
    
    accuracy_osml = accuracy_DT.to_pandas()
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
        'Accuracy-osml': '{:.2f}'.format(accuracy_osml.score[0]),
    }

     # Save evaluation metrics to a JSON file
    with open(f"{context.artifact_output_path}/metrics.json", "w+") as f:
        json.dump(evaluation, f)
        
    # Generate and save confusion matrix plot
    cm = confusion_matrix(predicted_data.result.to_pandas()['anomaly_int'], predicted_data.result.to_pandas()['prediction'])
    plot_confusion_matrix(cm, f"{context.artifact_output_path}/confusion_matrix")

    # Generate and save ROC curve plot
    roc_out = ROC(
        data=predictions.result,
        probability_column='prob_1',
        observation_column=target_name,
        positive_class='1',
        num_thresholds=1000
    )
    plot_roc_curve(roc_out, f"{context.artifact_output_path}/roc_curve")
    # exp.show_in_notebook(show_table=True)
    # dt_explain(exp, f"{context.artifact_output_path}/explain_pred.html")
    exp.save_to_file(f"{context.artifact_output_path}/expimg.html")
    with open(f"{context.artifact_output_path}/exp_features.json", "w+") as f:
        json.dump(exp.as_list(), f)
    # Calculate feature importance and generate plot
    try:
        model_pdf = model.result.to_pandas()['classification_tree']
        feature_importance = compute_feature_importance(model_pdf)
        feature_importance_df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance'])
        plot_feature_importance(feature_importance, f"{context.artifact_output_path}/feature_importance")
    except:
        feature_importance = {}

    predictions_table = "predictions_tmp"
    copy_to_sql(df=predicted_data.result, table_name=predictions_table, index=False, if_exists="replace", temporary=True)
    
    
    # calculate stats if training stats exist
    if os.path.exists(f"{context.artifact_input_path}/data_stats.json"):
        record_evaluation_stats(
            features_df=test_df,
            predicted_df=DataFrame.from_query(f"SELECT * FROM {predictions_table}"),
            feature_importance=feature_importance,
            context=context
        )

    print("All done!")
