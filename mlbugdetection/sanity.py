import pandas as pd

def sanity_check(model, examples, target_column):
    result = model.predict(examples.drop(target_column, axis=1))
    original = examples[target_column]
    values = (pd.Series(result) == original).value_counts().index
    if len(values) == 2:
        return False
    return values[0]