import modules.data_wrangling.feature_selection as feature_selection
import modules.classification.classifiers as classifiers
import modules.classification.probabilities as probabilities

import json

# Get cols
def get_all_cols(row_1):
    
    features_classification_lists = feature_selection.features_classification_lists()
    
    unuseful_cols = ['ID', 'URL', 'DATE', 'YEAR', 'RQ_JOINT']
    remove_cols = ['JUDGE_ID', 'PLAIN_ML', 'DEFEN_ML']
    categorical_cols = ['AUT_COMM', 'HQ']
    court_decision_cols = features_classification_lists["Court decisions"]
   
    cols = list(set(row_1.to_frame().T.columns) - set(unuseful_cols))
    cols = list(set(cols) - set(remove_cols))
    cols = list(set(cols) - set(categorical_cols))
    cols = list(set(cols) - set(court_decision_cols))
    
    return cols


# Top 10 vars and weights
def get_top_10_vars_and_wights_winwin(df):
    
    # Train model with all variables
    features_classification_lists = feature_selection.features_classification_lists()
    unuseful_cols = ['ID', 'URL', 'DATE', 'YEAR']
    categorical_features = ['AUT_COMM', 'HQ']    
    hide_cols = unuseful_cols + categorical_features + features_classification_lists["Court decisions"] + ["DEFEN_ML"]

    feature_importances = classifiers.rf_classifier(df, "WINWIN", hide_cols, printInfo = False)
    feature_importances = classifiers.sort_feature_importances(feature_importances)
        
    # Get top 10 variables with more importance        
    feature_importances_top_10 = feature_importances.head(10)
    features_top_10 = list(feature_importances_top_10["features"])
    weights_top_10 = list(feature_importances_top_10["coefficients"])
    feature_weights = dict(zip(features_top_10, weights_top_10))
        
     # Remove judge id or plaintiff gender
    if 'JUDGE_ID' in feature_weights:
        print("Removing 'JUDGE_ID' from the dict")
        feature_weights.pop('JUDGE_ID')
    elif 'PLAIN_ML' in feature_weights:
        print("Removing 'PLAIN_ML' from the dict")
        feature_weights.pop('PLAIN_ML')
    elif 'DEFEN_ML' in feature_weights:
        print("Removing 'DEFEN_ML' from the dict")
        feature_weights.pop('DEFEN_ML')
    
    return feature_weights


# Probability of winning
def prob_winning(df):
    
    features_classification_lists = feature_selection.features_classification_lists()
    unuseful_cols = ['ID', 'URL', 'DATE', 'YEAR']
    categorical_features = ['AUT_COMM', 'HQ']    
    hide_cols = unuseful_cols + categorical_features + ["DEFEN_ML"] + features_classification_lists["Court decisions"]
    
    df_conf = probabilities.rf_classifier_conf(df, "WINWIN", hide_cols)
    
    return df_conf

