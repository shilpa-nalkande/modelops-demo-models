from teradataml import (
    DataFrame,
    DecisionForest,
    ScaleFit,
    ScaleTransform,
)
from teradataml import td_sklearn as osml
from lime.lime_tabular import LimeTabularExplainer
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

# Define a function to recursively traverse a decision tree and count the usage of features
def traverse_tree(tree, feature_counter):
    if 'split_' in tree and 'attr_' in tree['split_']:
        feature_counter[tree['split_']['attr_']] += 1
    if 'leftChild_' in tree:
        traverse_tree(tree['leftChild_'], feature_counter)
    if 'rightChild_' in tree:
        traverse_tree(tree['rightChild_'], feature_counter)

# Compute feature importance based on tree traversal
def compute_feature_importance(trees_json):
    feature_counter = Counter()
    for tree_json in trees_json:
        tree = json.loads(tree_json)
        traverse_tree(tree, feature_counter)
    total_splits = sum(feature_counter.values())
    feature_importance = {feature: count / total_splits for feature, count in feature_counter.items()}
    return feature_importance


def plot_feature_importance(fi, img_filename):
    feat_importances = pd.Series(fi)
    feat_importances.nlargest(10).plot(kind='barh').set_title('Feature Importance')
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
    
    print("Starting training...")

    # Train the model using DecisionForest
    model = DecisionForest(data=scaled_train.result, 
                           input_columns = ['std_RESISTANCE','kurtosis_RESISTANCE','min_resistance_diff','skew_RESISTANCE',
                                            'sum_RESISTANCE','max_RESISTANCE','var_RESISTANCE','mean_RESISTANCE','min_RESISTANCE'], 
                            response_column = 'anomaly_int', 
                            max_depth = 16, 
                            num_trees = 8, 
                            min_node_size = 1, 
                            mtry = 1, 
                            mtry_seed = 3, 
                            seed = 3, 
                            tree_type = 'CLASSIFICATION')

    # Save the trained model to SQL
    model.result.to_sql(f"model_${context.model_version}", if_exists="replace")  
    print("Saved trained model")
    
    print("Starting osml training...")
    DT_classifier = osml.DecisionTreeClassifier(random_state=42)
    DT_classifier.fit(X_train, y_train)
    DT_classifier.deploy(model_name="DT_classifier", replace_if_exists=True)
    explainer = LimeTabularExplainer(X_train.get_values(), feature_names=X_train.columns,
                                            class_names=['Anomaly','NoAnomaly'], verbose=True, mode='classification')
    
    with open(f"{context.artifact_output_path}/exp_obj", 'wb') as f:
        dill.dump(explainer, f)
    print("Complete osml training...")
    
    # Calculate feature importance and generate plot
    model_pdf = model.result.to_pandas()['classification_tree']
    feature_importance = compute_feature_importance(model_pdf)
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
