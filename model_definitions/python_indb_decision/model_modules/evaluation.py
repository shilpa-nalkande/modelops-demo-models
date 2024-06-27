from sklearn.metrics import confusion_matrix, roc_curve, auc
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
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

        
# Compute feature importance based on tree traversal
def compute_feature_importance(model,X_train):
    feat_dict= {}
    for col, val in sorted(zip(X_train.columns, model.feature_importances_),key=lambda x:x[1],reverse=True):
        feat_dict[col]=val
    feat_df = pd.DataFrame({'Feature':feat_dict.keys(),'Importance':feat_dict.values()})
    # print(feat_df)
    return feat_df

def plot_feature_importance(fi, img_filename):
    feat_importances = fi.sort_values(['Importance'],ascending = False).head(10)
    feat_importances.plot(kind='barh').set_title('Feature Importance')
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
    from sklearn import metrics
    # auc = roc_out.result.to_pandas().iloc[0,0]
    # roc_results = roc_out.output_data.to_pandas()
    fpr, tpr, thresholds = metrics.roc_curve(roc_out['anomaly_int'], roc_out['decisiontreeclassifier_predict_1'])
    auc = metrics.auc(fpr, tpr)
    # plt.plot(roc_results['fpr'], roc_results['tpr'], color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' %auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.plot(fpr,tpr,label="ROC curve AUC="+str(auc), color='darkorange')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--') 
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    fig = plt.gcf()
    fig.savefig(img_filename, dpi=200)
    plt.clf()
    
    

def evaluate(context: ModelContext, **kwargs):

    aoa_create_context()

    # Load the trained model from SQL
    # model = DataFrame(f"model_${context.model_version}")

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
    
    print("Evaluating osml...")
    DT_classifier = osml.load(model_name="DT_classifier")
    predict_DT =DT_classifier.predict(X_test,y_test)
    # predict_prob =DT_classifier.predict_proba(X_test)
    # print(predict_prob)
    accuracy_DT = DT_classifier.score(X_test, y_test)
    df = X_test.sample(n=1)
    df = df.drop(columns="sampleid")
    with open(f"{context.artifact_input_path}/exp_obj", 'rb') as f:
        explainer = dill.load(f)
       
    exp = explainer.explain_instance(df.get_values().flatten(), DT_classifier.modelObj.predict_proba, num_features=9)
    

    # Evaluate classification metrics using ClassificationEvaluator
    ClassificationEvaluator_obj = ClassificationEvaluator(
        data=predict_DT,
        observation_column=target_name,
        prediction_column='decisiontreeclassifier_predict_1',
        num_labels=2
    )

#      # Extract and store evaluation metrics
    metrics_pd = ClassificationEvaluator_obj.output_data.to_pandas()
    #print(metrics_pd)
    # y_true_df = predict_DT.select(['anomaly_int'])
    # y_pred_df = predict_DT.select(['decisiontreeclassifier_predict_1'])
    # opt = osml.classification_report(y_true=y_true_df, y_pred=y_pred_df, output_dict=True, target_names=["class 0", "class 1"])
    # report_df = pd.DataFrame(opt)
    # print(report_df)
     
         
    evaluation = {
        'Accuracy': '{:.4f}'.format(metrics_pd.MetricValue[0]),
        'Micro-Precision': '{:.4f}'.format(metrics_pd.MetricValue[1]),
        'Micro-Recall': '{:.4f}'.format(metrics_pd.MetricValue[2]),
        'Micro-F1': '{:.4f}'.format(metrics_pd.MetricValue[3]),
        'Macro-Precision': '{:.4f}'.format(metrics_pd.MetricValue[4]),
        'Macro-Recall': '{:.4f}'.format(metrics_pd.MetricValue[5]),
        'Macro-F1': '{:.4f}'.format(metrics_pd.MetricValue[6]),
        'Weighted-Precision': '{:.4f}'.format(metrics_pd.MetricValue[7]),
        'Weighted-Recall': '{:.4f}'.format(metrics_pd.MetricValue[8]),
        'Weighted-F1': '{:.4f}'.format(metrics_pd.MetricValue[9]),
        # 'Accuracy-osml': '{:.2f}'.format(accuracy_osml.score[0]),
    }

     # Save evaluation metrics to a JSON file
    with open(f"{context.artifact_output_path}/metrics.json", "w+") as f:
        json.dump(evaluation, f)
        
    # Generate and save confusion matrix plot
    cm = confusion_matrix(predict_DT.to_pandas()['anomaly_int'], predict_DT.to_pandas()['decisiontreeclassifier_predict_1'])
    plot_confusion_matrix(cm, f"{context.artifact_output_path}/confusion_matrix")
    # dt_explain(shap_values[0],X_test,f"{context.artifact_output_path}/shap_exp.png")
    # dt_explain(explainer_shap.expected_value[0], shap_values50 ,X_test.to_pandas().iloc[1:50, :],f"{context.artifact_output_path}/shap_exp.png")
    # Generate and save ROC curve plot
    roc_out = ROC(
        data=predict_DT,
        probability_column='decisiontreeclassifier_predict_1',
        observation_column=target_name,
        positive_class='1',
        num_thresholds=1000
    )
    
    # fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # roc_auc = auc(fpr, tpr)
    # plot_roc_curve(roc_out, f"{context.artifact_output_path}/roc_curve")
    plot_roc_curve(predict_DT.to_pandas(), f"{context.artifact_output_path}/roc_curve")
    # exp.show_in_notebook(show_table=True)
    # dt_explain(exp, f"{context.artifact_output_path}/explain_pred.html")
    exp.save_to_file(f"{context.artifact_output_path}/expimg.html")
    # with open(f"{context.artifact_output_path}/exp_features.json", "w+") as f:
    #     json.dump(exp.as_list(), f)
    # Calculate feature importance and generate plot
    # try:
        # model_pdf = model.result.to_pandas()['classification_tree']
    feature_importance = compute_feature_importance(DT_classifier.modelObj,X_test)
    plot_feature_importance(feature_importance, f"{context.artifact_output_path}/feature_importance")
    
    # except:
    #     feature_importance = {}

    predictions_table = "predictions_tmp"
    copy_to_sql(df=predict_DT, table_name=predictions_table, index=False, if_exists="replace", temporary=True)
    
    
    # calculate stats if training stats exist
    if os.path.exists(f"{context.artifact_input_path}/data_stats.json"):
        record_evaluation_stats(
            features_df=test_df,
            predicted_df=DataFrame.from_query(f"SELECT * FROM {predictions_table}"),
            feature_importance=feature_importance,
            context=context
        )

    print("All done!")
