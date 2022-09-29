import pytest
import pandas as pd
import pickle

with open('models/LogisticRegression/LogisticRegression.pkl', 'rb') as f:
    LR = pickle.load(f)
with open('models/RandomForest/RandomForest.pkl', 'rb') as f:
    RF = pickle.load(f)

full_df = pd.read_csv('../datasets/fraud_new.csv')
sample = full_df.sample(1,ignore_index=True)

def sanity_check(model, examples, target_column):
    result = model.predict(examples.drop(target_column, axis=1))
    original = examples[target_column]
    values = (pd.Series(result) == original).value_counts().index
    if len(values) == 2:
        return False
    return values[0]

@pytest.mark.parametrize("model, examples, target_column", [
    (LR, sample, "isFraud"),
    (RF, sample, "isFraud")
])
def test_sanity(model, examples, target_column):
    result = model.predict(examples.drop(target_column, axis=1))
    original = examples[target_column]
    values = (pd.Series(result) == original).value_counts().index
    if len(values) == 2:
        assert values[0] == True
        assert values[1] == True
    else:
        assert values[0] == True
