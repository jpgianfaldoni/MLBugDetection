import pandas as pd
import pkg_resources


def load_dataset():
    """
        Loads the Breast Cancer classification Dataset

        Returns
        -------
        Pandas dataframe containing the Breast Cancer classification dataset
    """
    stream = pkg_resources.resource_stream(__name__, 'data/breast-cancer.csv')
    return pd.read_csv(stream)
