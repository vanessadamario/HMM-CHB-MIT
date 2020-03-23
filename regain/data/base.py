"""Utility to load datasets."""
import pandas as pd
import pickle as pkl
import numpy as np
import os


def load_food_search_trends(groups=False):
    """
    Data from: http://foodb.ca/
    https://www.google.com/intl/es419/search/about/
    """
    filename = os.path.join(os.path.dirname(__file__),
                            "food_search_trends/food.csv")
    data = pd.read_csv(filename,
                       index_col=0)
    if groups:
        filename = os.path.join(os.path.dirname(__file__),
                                "food_search_trends/groups_of_food.pkl")
        with open(filename, "rb") as f:
            groups_dict = pkl.load(f)
        connections = np.zeros((len(groups_dict), data.shape[1]))
        connections = pd.DataFrame(connections, index=list(groups_dict.keys()),
                                   columns=data.columns)

        for g, v in groups_dict.items():
            connections.loc[g, v] = 1

        return data, connections, groups_dict
    else:
        return data


def load_finance(condensed=True, sectors=False):
    """
    Data from:  https://quantquote.com/historical-stock-data
    https://datahub.io/core/s-and-p-500-companies
    """
    if condensed:
        filename = os.path.join(os.path.dirname(__file__),
                                "finance/finance.csv")
        data = pd.read_csv(filename, index_col=0)
    else:
        filename = os.path.join(os.path.dirname(__file__),
                                "finance/not_collapsed.csv")
        data = pd.read_csv(filename, index_col=0)

    if sectors:
        filename = os.path.join(os.path.dirname(__file__),
                                "finance/industries_sector.csv")
        sectors = pd.read_csv(filename, index_col=0)
        
        
def load_commodities_data(type_data = 1):
    
    filename = os.path.join(os.path.dirname(__file__),
                                "Commodities/Dataset"+str(type_data)+".pickle")
    
    with open(filename ,'rb') as f:
            [Dataset] = pkl.load(f)
            
    return Dataset


def load_ETF_data(file = 'ETF_data'):
    filename = os.path.join(os.path.dirname(__file__),
                            "ETF/" + str(file) + ".pickle")

    with open(filename, 'rb') as f:
        [Dataset] = pkl.load(f)

    return Dataset
    
    
def load_Petrobras():
    """
    Data from:  https://quantquote.com/historical-stock-data
    https://datahub.io/core/s-and-p-500-companies
    """

    filename1 = os.path.join(os.path.dirname(__file__),
                            "Petro_BrazCurr/PETR3.csv")

    filename2 = os.path.join(os.path.dirname(__file__),
                            "Petro_BrazCurr/USD_BRL.csv")
    filename3 = os.path.join(os.path.dirname(__file__),
                            "Petro_BrazCurr/Petrolio.csv")
    dat1 = pd.read_csv(filename1, index_col=0)
    dat2 = pd.read_csv(filename2, index_col=0)
    dat3 = pd.read_csv(filename3, index_col=0)

    return dat1,dat2,dat3

