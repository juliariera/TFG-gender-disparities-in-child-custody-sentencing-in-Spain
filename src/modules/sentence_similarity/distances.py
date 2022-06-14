import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import modules.utils.utils as utils
import modules.data_wrangling.feature_selection as feature_selection

def euclidean_distance(a, b):
    dist = np.linalg.norm(a-b)
    return dist


def similarity_cols(features_classification_lists, show_removed_cols = False, show_cols_similarity = False):
        
    non_info_cols = features_classification_lists["Judicial resolution"]
    court_decicion_cols = features_classification_lists["Court decisions"]
    
    cols = utils.flatten_list_of_lists(features_classification_lists)
    
    cols_similarity = [i for i in cols if i not in (non_info_cols)]
    cols_similarity = [i for i in cols_similarity if i not in (court_decicion_cols)]
    
    if(show_removed_cols):
        removed_cols = [i for i in cols if i not in cols_similarity]
        print("The removed columns are:")
        print(removed_cols)
        
    if(show_cols_similarity):
        print("\nThe columns used for similarity are:")
        print(cols_similarity)
        
    return cols_similarity


def similarity_df(df, cols_similarity):
    df_similarity = df[df.columns.intersection(cols_similarity)]
    return df_similarity


def create_distance_matrix(df, cols_similarity):
    
    df = similarity_df(df, cols_similarity)
    
    distance_matrix = pd.DataFrame(columns = np.arange(df.shape[0]), index = np.arange(df.shape[0]))
    
    for index_1, row in df.iterrows():

        for index_2, row in df.iterrows():

            row_1 = df.iloc[[index_1]].to_numpy()
            row_2 = df.iloc[[index_2]].to_numpy()

            dist = euclidean_distance(row_1, row_2)

            distance_matrix.loc[index_1,index_2] = dist
            
    return distance_matrix


def get_all_distances(distance_matrix):
    
    distances_df = pd.DataFrame(columns = ["sentence_1", "sentence_2", "distance"])

    for index_1, row in distance_matrix.iterrows():
        for index_2 in row.index:
            if(index_1 != index_2 and index_1 < index_2):
                dist = distance_matrix.loc[index_1, index_2]
                new_row = {'sentence_1':index_1, 'sentence_2':index_2, 'distance':dist}
                distances_df = distances_df.append(new_row, ignore_index=True)

    distances_df = distances_df.astype({"sentence_1": int, "sentence_2": int, "distance": float})
    
    return distances_df


def distances_plot(distances_df):
    sns.distplot(distances_df["distance"], label="distance", color = "#e2b0a6")
    plt.title('Distances distribution')
    plt.legend()
    plt.show()
