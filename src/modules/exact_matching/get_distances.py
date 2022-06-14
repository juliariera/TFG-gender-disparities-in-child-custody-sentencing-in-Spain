import modules.exact_matching.precomputations as precomputations

import numpy as np
import pandas as pd
import json


def euclidean_distance(row_1, row_2, cols):
    
    row_1 = row_1[cols]
    row_2 = row_2[cols]
    
    row_1 = row_1.to_numpy()
    row_2 = row_2.to_numpy()
    
    dist = np.linalg.norm(row_1-row_2, ord = 2)
    return dist

def weighted_euclidean_distance(row_1, row_2, cols, weights):
    
    row_1 = row_1[cols]
    row_2 = row_2[cols]
    
    row_1 = row_1.to_numpy()
    row_2 = row_2.to_numpy()
    
    s = row_1 - row_2

    dist = np.sqrt((weights*s*s).sum())
    return dist

# Distance 1: Eclidean distance for all variables except the plaintiff gender and the judge id
def euclidean_distance_all(row_1, row_2):

    cols = precomputations.get_all_cols(row_1)

    return euclidean_distance(row_1, row_2, cols)


# Distance 2: Euclidean distance on a selection of few variables (except the plaintiff gender and the judge id) which have more importance in the winning probability
def euclidean_distance_most_imp_vars(row_1, row_2):
    
    # Check request type
    request_type = row_1.to_frame().T["RQ_JOINT"].values
    
    if request_type == 0:
        with open("../output/feature_weights_top_10_sole.json") as json_file:
            feature_weights = json.load(json_file)
    else:
        with open("../output/feature_weights_top_10_joint.json") as json_file:
            feature_weights = json.load(json_file)
    
    # Get the list of most important vars
    most_important_vars = list(feature_weights.keys())
        
    return euclidean_distance(row_1, row_2, most_important_vars)


# Distance 3: Euclidean distance (except the plaintiff gender and the judge id) with weights proportional to the importance of each variable in the model that predicts the probability of winning
def euclidean_distance_with_weights(row_1, row_2):
    
    # Check request type
    request_type = row_1.to_frame().T["RQ_JOINT"].values
    
    if request_type == 0:
        with open("../output/feature_weights_top_10_sole.json") as json_file:
            feature_weights = json.load(json_file)
    else:
        with open("../output/feature_weights_top_10_joint.json") as json_file:
            feature_weights = json.load(json_file)

    cols = list(feature_weights) # keys in the weights dictionary
    weights = list(feature_weights.values())

    return weighted_euclidean_distance(row_1, row_2, cols, weights)
                   

# Distance 4: Probability of winning, using all variables except the plaintiff gender and the judge id
def distance_probability_of_winning(row_1, row_2):
    
    prob_winning = pd.read_csv("..\output\prob_winning.csv", sep=";")
 
    prob_row_1 = prob_winning.loc[prob_winning['test_index'] == row_1.name, 'conf_1'] # row.name gives the index of the row
    prob_row_2 = prob_winning.loc[prob_winning['test_index'] == row_2.name, 'conf_1']
    
    return float(abs(prob_row_1.values-prob_row_2.values)) # absolute value of the difference
