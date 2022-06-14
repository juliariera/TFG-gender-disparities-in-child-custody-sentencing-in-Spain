import pandas as pd

def get_sorted_distances_below_threshold(distances_df, threshold):
    
    # Sort by ascending order
    distances_df = distances_df.sort_values(by = "distance", ascending = True)
    distances_df = distances_df.reset_index(drop=True)
    
    # Select only rows below threshold
    subset_distances_df = distances_df[(distances_df["distance"] < threshold)]

    return subset_distances_df


def similar_pairs_different_label(df, subset_distances_df):
    
    df_similar_different_label = pd.DataFrame(columns = ["sentence_1", "sentence_2", "distance"])
    
    for index, row in subset_distances_df.iterrows():
            
            label_1 = df.loc[row["sentence_1"],"WINWIN"]
            label_2 = df.loc[row["sentence_2"],"WINWIN"]
            
            if(label_1 != label_2):
                df_similar_different_label = df_similar_different_label.append(row, ignore_index=True)
    
    df_similar_different_label = df_similar_different_label.astype({"sentence_1": int, "sentence_2": int, "distance": float})
    
    return df_similar_different_label