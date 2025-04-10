from teradataml import (
    DataFrame,
    XGBoost,
    ScaleFit,
    ScaleTransform,
    OrdinalEncodingFit,
    ColumnTransformer,
    Shap
)

from aoa import (
    record_training_stats,
    aoa_create_context,
    ModelContext
)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from collections import Counter


def compute_feature_importance(feat_df):
    df = feat_df.to_pandas()
    df = df.T.reset_index()
    df=df.rename(columns={'index': 'Feature', 0: 'Importance'})
    df['Feature'] = df['Feature'].str.replace('TD_', '')
    df['Feature'] = df['Feature'].str.replace('_SHAP', '')
    return df


def compute_feature_explain(explain_df):
    explain_df = explain_df.drop(['txn_id','Label','tree_num'],axis=1)
    shap_mean = explain_df.agg(['min', 'max'])
    df = shap_mean.to_pandas()
    df = df.T.reset_index()
    df=df.rename(columns={'index': 'Feature', 0: 'Importance'})
    mean_positive = df[df['Importance'] > 0]
    mean_negative = df[df['Importance'] < 0]
    mean_positive['Feature'] = mean_positive.loc[:,'Feature'].str.replace('max_TD_', '')
    mean_positive['Feature'] = mean_positive.loc[:,'Feature'].str.replace('_SHAP', '')
    mean_negative['Feature'] = mean_negative.loc[:,'Feature'].str.replace('min_TD_', '')
    mean_negative['Feature'] = mean_negative.loc[:,'Feature'].str.replace('_SHAP', '')
    # mean_positive['Feature'] = mean_positive['Feature'].str.replace('max_TD_', '')
    # mean_positive['Feature'] = mean_positive['Feature'].str.replace('_SHAP', '')
    # mean_negative['Feature'] = mean_negative['Feature'].str.replace('min_TD_', '')
    # mean_negative['Feature'] = mean_negative['Feature'].str.replace('_SHAP', '')
    return mean_positive,mean_negative
    

def plot_feature_importance(df, img_filename):
    df = df.sort_values(by="Importance", ascending=False)
    # Plot the bar graph
    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance",y="Feature",data=df, palette="viridis")
    plt.title("Feature Importance")
    plt.xlabel("SHAP Importance Value")
    plt.ylabel("Features")
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(img_filename, dpi=500)
    plt.clf()
    
def plot_feature_explain(mean_positive,mean_negative, img_filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35

    ax.barh(mean_positive["Feature"], mean_positive["Importance"],color='salmon', label='-1 (positive)') 
    ax.barh(mean_negative["Feature"], mean_negative["Importance"],color='cyan', label='1 (negative)')
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title("Mean shap for all samples")
    ax.legend(title="sign")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    # plt.show()
    fig = plt.gcf()
    fig.savefig(img_filename, dpi=500)
    plt.clf()    
    
def train(context: ModelContext, **kwargs):
    aoa_create_context()
    
    # Extracting feature names, target name, and entity key from the context
    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    # Load the training data from Teradata
    train_df = DataFrame.from_query(context.dataset_info.sql)
    
    print("Starting training...")

    # Train the model using XGBoost
    # model = DecisionForest(data = train_df,
    #             input_columns = ["Tenure", "InternetService", "OnlineSecurity", "SeniorCitizen",
    #                                 "PaymentMethod", "OnlineBackup", "Dependents", "Partner", "MultipleLines", 
    #                                 "StreamingMovies", "Gender", "PhoneService", "TotalCharges", "Contract", 
    #                                 "MonthlyCharges", "DeviceProtection", "PaperlessBilling", "StreamingTV", 
    #                                 "TechSupport"],
    #             response_column = 'Churn',
    #             family = 'Binomial',
    #             min_impurity= 0.0,
    #             max_depth= 5,
    #             min_node_size= 1,
    #             num_trees= -1,
    #             seed= 42,
    #             tree_type = 'CLASSIFICATION')
    
    model = XGBoost(
                    data=train_df,
                    input_columns=["amount", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "oldbalanceOrig", "CASH_IN_type", 
                        "CASH_OUT_type", "DEBIT_type", "PAYMENT_type", "TRANSFER_type"],
                    response_column = 'isFraud',
                    lambda1 = 120.0,
                    model_type='Classification',
                    seed=42,
                    shrinkage_factor=0.1,
                    max_depth=7
                )
    # Save the trained model to SQL
    model.result.to_sql(f"model_${context.model_version}", if_exists="replace")  
    print("Saved trained model")
    
    #Shap explainer 
    Shap_out = Shap(data=train_df, 
                object=model.result, 
                id_column='txn_id',
                training_function="TD_XGBoost", 
                model_type="Classification",
                input_columns=feature_names, 
                detailed=True)
    
    feat_df = Shap_out.output_data
    explain_df = Shap_out.result
    # print(explain_df)

 
    df = compute_feature_importance(feat_df)
    plot_feature_importance(df, f"{context.artifact_output_path}/feature_importance")
    pos_expl_df, neg_expl_df = compute_feature_explain(explain_df)
    # print(pos_expl_df)
    # print(neg_expl_df)
    plot_feature_explain(pos_expl_df,neg_expl_df, f"{context.artifact_output_path}/feature_explainability")
    
    categorical=["CASH_IN_type","CASH_OUT_type", "DEBIT_type", "PAYMENT_type", "TRANSFER_type","isFraud"]

    record_training_stats(
        train_df,
        features=feature_names,
        targets=[target_name],
        categorical=categorical
        ,
        # feature_importance=feature_importance,
        context=context
    )
    
    print("All done!")
