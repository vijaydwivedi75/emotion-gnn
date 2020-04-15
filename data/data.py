"""
    File to load dataset based on user control from main file
"""
from data.iemocap import IEMOCAP_KNN_Graph_Dataset

def LoadData(DATASET_NAME):
    """
        This function is called in the main.py file 
        returns:
        ; dataset object
    """
    # handling for IEMOCAP k-NN graph dataset
    if DATASET_NAME == 'IEMOCAP':
        return IEMOCAP_KNN_Graph_Dataset(DATASET_NAME)