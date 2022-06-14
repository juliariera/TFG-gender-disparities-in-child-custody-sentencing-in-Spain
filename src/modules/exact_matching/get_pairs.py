import pandas as pd


# - For each x, with male plaintiff, in A:
    # - Find element y, with female plaintiff, in B that minimizes d(x, y) --> closest element in other group with different plaintiff gender. d(x,y) cannot consider neither the gender nor the judge id
    # - If d(x,y) < epsilon, accept the pair (x,y), if not, reject

def get_pairs(df_A, df_B, get_distance, epsilon = 1):
    pairs = pd.DataFrame(columns = ['index_1', 'index_2', 'dist'])

    # For each x, with male plaintiff, in A
    for index_A, row_A in df_A.iterrows():
        if(row_A["PLAIN_ML"] == 1):

            distances = {}

            # Find element y, with female plaintiff, in B
            for index_B, row_B in df_B.iterrows():
                if(row_B["PLAIN_ML"] == 0):
                    # Get distance d(x, y): element x and all elements y
                    dist = get_distance(row_A, row_B)
                    distances[index_B] = dist

            # Get the index of the element y that minimizes d(x, y)
            index_min_dist_B = min(distances, key=distances.get)

            # If d(x,y) < epsilon, accept the pair (x,y)
            if distances[index_min_dist_B] < epsilon:
                new_row = {'index_1': index_A, 'index_2': index_min_dist_B, 'dist': round(distances[index_min_dist_B], 3)}
                pairs = pairs.append(new_row, ignore_index=True)
                pairs['index_1'] = pairs['index_1'].astype(int)
                pairs['index_2'] = pairs['index_2'].astype(int)
                
    return pairs
