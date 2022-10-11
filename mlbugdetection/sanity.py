import pickle
import pandas as pd

def sanity_check(model, samples, target):
    '''Sanity Test
        Analyzes the sanity of a model with samples and 
        return a bool that represents if the tests passed or not.

    Parameters
    ----------
    model : sklearn model
        The model to be used for prediction.

    samples : pandas DataFrame
        The samples to be used for prediction, which the model
        need to predict correctly. 

    target : str
        The name of the column containing the target variable.

    Returns
    -------
    bool True if the model is sane, False otherwise.
    '''
    if type(model) == str:
        with open(model, 'rb') as f:
            model = pickle.load(f)

    result = model.predict(samples.drop(target, axis=1))
    original = samples[target]
    values = (pd.Series(result) == original).value_counts().index
    if len(values) == 2:
        return False
    return values[0]
