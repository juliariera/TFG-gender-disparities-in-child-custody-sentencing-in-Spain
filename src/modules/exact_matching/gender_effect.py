

def gender_effect(df, pairs):
    
    # Mean A: males
    A_values = [df.iloc[index_1]["WINWIN"] for index_1 in pairs["index_1"]] # create a list of the values "WINWIN" for each index
    mean_A = sum(A_values)/len(A_values)
    
    # Mean B: females
    B_values = [df.iloc[index_2]["WINWIN"] for index_2 in pairs["index_2"]] # create a list of the values "WINWIN" for each index
    mean_B = sum(B_values)/len(B_values)
    
    return round(mean_A - mean_B, 5)