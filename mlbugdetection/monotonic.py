import pickle
import numpy as np
from matplotlib import pyplot as plt
from .analysis_report import AnalysisReport

def monotonicity_mse(predictions):
    """Monotonicity Mean Square Error

        Calculates the MSE between a list of prediction brobabilities and the closest monotonic version
        of this list.

    Parameters
    ----------

    predictions : List
        List of prediction probabilities calculated on the check_monotonicity function.
    
    Returns
    -------
        desc | asc : List 
            List of closest monotonic version of "predictions".

        mse_desc | mse_as : int
            MSE between "predictions" and desc/asc.
    """

    desc = np.minimum.accumulate(predictions)
    asc = np.maximum.accumulate(predictions)
    mse_desc = (np.square(predictions - desc)).mean(axis=0)
    mse_asc = (np.square(predictions - asc)).mean(axis=0)
    if min(mse_asc,mse_desc) == mse_desc:
        return desc, min(mse_asc,mse_desc)
    else:
        return asc, min(mse_asc,mse_desc)

def check_monotonicity_single_sample(model, sample, feature, start, stop, step=1):
    '''Monotonicity Analysis for a single example

    Parameters
    ----------
    model : sklearn model or str
        Model that will be used to make predictions. Could be a model object or a path to a model file.

    sample : pandas.DataFrame
        Pandas DataFrame containing one row that will be used as base point.

    feature : str
        Name of the feature being analysed.

    start : int
        The starting value of the feature's interval.

    stop : int
        The end value of the feature's interval.

    step : float, default=1
        Size of the step between ranges "start" and "stop".
        Ex: step = 0.1 between ranges 0 and 1 will result in [0  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]

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
    
    feature_range : tuple
        Range of values of the feature being analysed: (start, stop).
    
    metrics : dictionary
        Dictionary with all the calculated metrics, such as:
        
        'monotonic' : bool
             If the list of values is monotonic.

        'monotonic_mse': float
            MSE between the list of values and it`s closest monotonic aproximation. 

    graphs : List
            List of all the figures created.
    '''

    if len(sample) > 1:
        raise Exception("Sample must have only one example, please use 'check_monotonicity_multiple_samples' for multiple samples")
    if type(model) == str:
        with open(model, 'rb') as f:
            model = pickle.load(f)
            
    report = AnalysisReport()
    colValues = []
    predictions = []


    for i in np.arange(start,stop,step):
        colValues.append(i)
        sample[feature] = i
        prediction = model.predict_proba(sample)
        predictions.append(prediction[0][1])
    
    monotonic =  (all(predictions[i] <= predictions[i + 1] for i in range(len(predictions) - 1)) or all(predictions[i] >= predictions[i + 1] for i in range(len(predictions) - 1)))
    report.model_name = type(model).__name__
    report.analysed_feature = feature
    report.feature_range = (start, stop)
    fig = plt.figure(figsize=(6, 3), dpi=150)
    report.graphs.append(fig)
    if monotonic:
        report.metrics["monotonic"] = True
        report.metrics["monotonic_mse"] = 0
    else:
        report.metrics["monotonic"] = False
        monotonic_curve, m_mse_score = monotonicity_mse(predictions)
        report.metrics["monotonic_mse"] = m_mse_score
        plt.plot(colValues, monotonic_curve, linestyle='dashed', color='red', alpha=0.7, label="Monotonic Approximation")
    plt.plot(colValues, predictions, color='blue', alpha=0.7, label="Predictions Curve")
    plt.title(f"Model: {type(model).__name__}")
    plt.xlabel(f'Feature {feature} value')
    plt.ylabel('Predict proba')
    plt.legend(loc="lower right")
    return report

def check_monotonicity_multiple_samples(model, sample, feature, start, stop, step=1):
    '''Monotonicity Analysis for multiple examples

    Parameters
    ----------
    model : sklearn model or str
        Model that will be used to make predictions. Could be a model object or a path to a model file.

    sample : pandas.DataFrame
        Pandas DataFrame containing two or more rows that will be used as base point.

    feature : str
        Name of the feature being analysed.

    start : int
        The starting value of the feature's interval.

    stop : int
        The end value of the feature's interval.

    step : float, default=1
            Size of the step between ranges "start" and "stop".
            Ex: step = 0.1 between ranges 0 and 1 will result in [0  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]

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
    
    feature_range : tuple
        Range of values of the feature being analysed: (start, stop).
    
    metrics : dictionary
        Dictionary with all the calculated metrics, such as:
        
        'monotonic' : bool
             If the list of values is monotonic.

        'monotonic_mse': float
            MSE between the list of values and it`s closest monotonic aproximation. 
        
        'monotonic_means_std': float
            Standard deviation of the means of the predictions probabilities.

    graphs : List
            List of all the figures created.
    '''
    if len(sample) < 2:
        raise Exception("Sample must have multiple examples, please use 'check_monotonicity_single_sample' for single example")
    if type(model) == str:
        with open(model, 'rb') as f:
            model = pickle.load(f)
            
    report = AnalysisReport()
    colValues = []
    predictions = []

    for i in np.arange(start,stop,step):
        colValues.append(i)
        sample[feature] = i
        prediction = model.predict_proba(sample)
        predictions.append(np.mean(prediction[0][1]))

    monotonic =  (all(predictions[i] <= predictions[i + 1] for i in range(len(predictions) - 1)) or all(predictions[i] >= predictions[i + 1] for i in range(len(predictions) - 1)))
    report.model_name = type(model).__name__
    report.analysed_feature = feature
    report.feature_range = (start, stop)
    fig = plt.figure(figsize=(6, 3), dpi=150)
    report.graphs.append(fig)
    if monotonic:
        report.metrics["monotonic"] = True
        report.metrics["monotonic_mse"] = 0
    else:
        report.metrics["monotonic"] = False
        monotonic_curve, m_mse_score = monotonicity_mse(predictions)
        report.metrics["monotonic_mse"] = m_mse_score
        plt.plot(colValues, monotonic_curve, linestyle='dashed', color='red', alpha=0.7, label="Monotonic Approximation")
    report.metrics["monotonic_means_std"] = np.nanstd(predictions)
    plt.plot(colValues, predictions, color='blue', alpha=0.7, label="Predictions Curve")
    plt.title(f"Model: {type(model).__name__}")
    plt.xlabel(f'Feature {feature} value')
    plt.ylabel('Predict proba')
    plt.legend(loc="lower right")
    return report
