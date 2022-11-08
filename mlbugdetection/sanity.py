import pickle
import pandas as pd
from .analysis_report import AnalysisReport

def check_type_input_model(model):
    ''' Check the type of the input model and returns the model object '''
    if type(model) == str:
        with open(model, 'rb') as f:
            model = pickle.load(f)
    return model

def sanity_check(model, samples, target):
    '''Sanity Test
        Analyzes the sanity of a model with samples and 
        return a bool that represents if the tests passed or not.

    Parameters
    ----------
    model : sklearn model or str
        The model to be used for prediction. Could be a model object or a path to a model file.

    samples : pandas DataFrame
        The samples to be used for prediction, which the model
        need to predict correctly. 

    target : str
        The name of the column containing the target variable.

    Returns
    -------
    bool True if the model is sane, False otherwise.
    '''
    model = check_type_input_model(model)

    result = model.predict(samples.drop(target, axis=1))
    original = samples[target]

    result = pd.Series(result).reset_index(drop=True)
    origin = original.reset_index(drop=True)
    values = result == origin

    if len(values.value_counts().index) == 2:
        return False

    return values[0]

def sanity_check_with_indexes(model, samples, target):
    '''Sanity Test With Indexes
        Analyzes the sanity of a model with samples and 
        shows a Analysis Report that shows if the tests passed or not.
        If the tests failed, it will show the indexes of the samples that were misclassified.

    Parameters
    ----------
    model : sklearn model or str
        The model to be used for prediction. Could be a model object or a path to a model file.

    samples : pandas DataFrame
        The samples to be used for prediction, which the model
        need to predict correctly. 

    target : str
        The name of the column containing the target variable.

    Returns
    -------
     AnalysisReport object with following attributes:
        For more information:
        >>> from mlbugdetection.analysis_report import AnalysisReport
        >>> help(AnalysisReport)

    model_name : str
        Name of the model being analysed.
    
    analysed_feature : str
        Name of the feature being analysed.

    metrics : dictionary
        Dictionary with all the calculated metrics, such as:
        
        'sanity' : bool
            If the model is sane or not.

        'sanity_indexes': List
            List of indexes of the samples that were misclassified.

    '''
    model = check_type_input_model(model)
    report = AnalysisReport()

    result = model.predict(samples.drop(target, axis=1))
    original = samples[target]

    result = pd.Series(result).reset_index(drop=True)
    origin = original.reset_index(drop=True)
    values = result == origin
    
    report.model_name = type(model).__name__
    report.analysed_feature = target

    if len(values.value_counts().index) == 2:
        report.metrics["sanity"] = False
        report.metrics["sanity_indexes"] = values[values==False].index.to_list()
        return report

    report.metrics["sanity"] = True
    report.metrics["sanity_indexes"] = []
    return report
