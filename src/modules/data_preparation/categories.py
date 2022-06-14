import pandas as pd

def get_df_categories():
    return pd.read_csv(r"..\data\feature_category.csv", sep=";")

def get_list_categories():
    df_categories = get_df_categories()
    list_categories = list(pd.unique(df_categories['category']))
    return list_categories

def categories_list_and_dict():
    
    # Get data 
    df_feature_category = get_df_categories()

    # Get the categories list
    categories_list = get_list_categories()
    
    # Convert to dict
    dict_feature_category = {}

    for index, row in df_feature_category.iterrows():
        dict_feature_category[row["feature"]] = row["category"]

    return categories_list, dict_feature_category