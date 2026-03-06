from teradataml import (
    DataFrame,
    ScaleFit,
    ScaleTransform,
)
from teradataml import td_sklearn as osml
from teradataml import td_lightgbm

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
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def compute_feature_importance(features, importances):
    feat_dict = {}
    for i in range(len(importances)):
        feat_dict[features[i]] = importances[i]
    feat_df = pd.DataFrame({'Feature':feat_dict.keys(),'Importance':feat_dict.values()})
    return feat_df

def plot_feature_importance(fi, img_filename):
    feat_importances = fi.sort_values(['Importance'], ascending = False)
    feat_importances.plot(kind='barh', x='Feature', y='Importance').set_title('Feature Importance')
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
    X_train = train_df.drop(['HasDiabetes','PatientId'], axis = 1)
    y_train = train_df.select(["HasDiabetes"])
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

    # Dataset creation.
    LightGBM_Classifier = td_lightgbm.Dataset(X_train, y_train) #, silent = True, free_raw_data=False)

    print("Starting training using teradata osml...")

    model = td_lightgbm.train(params={}, train_set = LightGBM_Classifier, num_boost_round=30)

    model.save_model(f"{context.artifact_output_path}/light_gbm")

    print("Complete osml training...")

    # Calculate feature importance and generate plot
    imp = model.feature_importance()
    feature_importance = compute_feature_importance(feature_names, imp)

    # Plot feature importance using Gain
    plot_feature_importance(feature_importance, f"{context.artifact_output_path}/feature_importance")
    # td_lightgbm.plot_importance(model, importance_type="gain", figsize=(7,6), title="LightGBM Feature Importance (Gain)")
    # plt.savefig(f"{context.artifact_output_path}/feature_importance")

    record_training_stats(
        train_df,
        features=feature_names,
        targets=[target_name],
        categorical=[target_name],
        feature_importance=feature_importance,
        context=context
    )

    print("All done!")
