from teradataml import (
    DataFrame,
    ScaleFit,
    ScaleTransform,
)
from teradataml import td_sklearn as osml
# from lime.lime_tabular import LimeTabularExplainer
from aoa import (
    record_training_stats,
    aoa_create_context,
    ModelContext
)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import dill
from collections import Counter
import shap
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Compute feature importance based on tree traversal
def compute_feature_importance(model,X_train):
    # from sklearn.inspection import permutation_importance
    feat_dict= {}
    for col, val in sorted(zip(X_train.columns, model.feature_importances_),key=lambda x:x[1],reverse=True):
        feat_dict[col]=val
    feat_df = pd.DataFrame({'Feature':feat_dict.keys(),'Importance':feat_dict.values()})
    
    return feat_df
    

def plot_feature_importance(fi, img_filename):
    feat_importances = fi.sort_values(['Importance'],ascending = False).head(10)
    feat_importances.plot(kind='barh').set_title('Feature Importance')
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

    print ("Scaling using InDB Functions...")
    X_train = train_df.drop(['anomaly_int','WELDING_ID'], axis = 1)
    y_train = train_df.select(["anomaly_int"])
    # Scale the training data using the ScaleFit and ScaleTransform functions
    scaler = ScaleFit(
        data=train_df,
        target_columns = feature_names,
        scale_method="STD",
        global_scale=False
    )

    scaled_train = ScaleTransform(
        data=train_df,
        object=scaler.output,
        accumulate = [target_name,entity_key]
    )
    
    scaler.output.to_sql(f"scaler_${context.model_version}", if_exists="replace")
    print("Saved scaler")
    
         
    print("Starting training using teradata osml...")

    RF_classifier = osml.RandomForestClassifier(n_estimators=10,max_leaf_nodes=2,max_features='auto',max_depth=2)
    RF_classifier.fit(X_train, y_train)
    RF_classifier.deploy(model_name="RF_classifier", replace_if_exists=True)
        
    print("Complete osml training...")
    
    # Calculate feature importance and generate plot
    feature_importance = compute_feature_importance(RF_classifier.modelObj,X_train)
    plot_feature_importance(feature_importance, f"{context.artifact_output_path}/feature_importance")
    
    record_training_stats(
        train_df,
        features=feature_names,
        targets=[target_name],
        categorical=[target_name],
        feature_importance=feature_importance,
        context=context
    )
    
    print("All done!")
