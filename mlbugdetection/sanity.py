import pandas as pd

def sanity_check(model, examples, target_column):
    '''
    Parameters
    ----------
    model : sklearn model
        The model to be used for prediction.

    examples : pandas DataFrame
        The examples to be used for prediction.

    target_column : str
        The name of the column containing the target variable.

    Returns
    -------
    True if the model is sane, False otherwise.
    '''
    result = model.predict(examples.drop(target_column, axis=1))
    original = examples[target_column]
    values = (pd.Series(result) == original).value_counts().index
    if len(values) == 2:
        return False
    return values[0]