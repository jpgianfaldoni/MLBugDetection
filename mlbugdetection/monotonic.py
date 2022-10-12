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

def check_monotonicity(model, sample, feature, start, stop, steps=100):
    '''Monotonicity Analysis

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

    steps : int, default=100
        Number of values that will be atributed to the analysed feature. Must be non-negative.

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

        'monotonic_score': float
            MSE between the list of values and it`s closest monotonic aproximation. 

    graphs : List
            List of all the figures created.
    '''
    if type(model) == str:
        with open(model, 'rb') as f:
            model = pickle.load(f)
            
    report = AnalysisReport()
    colValues = []
    predictions = []
    for i in np.linspace(start,stop,steps):
        colValues.append(i)
        sample[feature] = i
        prediction = model.predict_proba(sample)
        predictions.append(prediction[0][0])
    
    monotonic =  (all(predictions[i] <= predictions[i + 1] for i in range(len(predictions) - 1)) or all(predictions[i] >= predictions[i + 1] for i in range(len(predictions) - 1)))
    report.model_name = type(model).__name__
    report.analysed_feature = feature
    report.feature_range = (start, stop)
    fig = plt.figure(figsize=(6, 3), dpi=150)
    report.graphs.append(fig)
    if monotonic:
        report.metrics["monotonic"] = True
        report.metrics["monotonic_score"] = 1
    else:
        report.metrics["monotonic"] = False
        monotonic_curve, m_mse_score = monotonicity_mse(predictions)
        report.metrics["monotonic_score"] = m_mse_score
        plt.plot(colValues, monotonic_curve, linestyle='dashed', color='red', alpha=0.7, label="Monotonic Approximation")
    plt.plot(colValues, predictions, color='blue', alpha=0.7, label="Predictions Curve")
    plt.title(f"Model: {type(model).__name__}")
    plt.xlabel('Feature value')
    plt.ylabel('Predict proba')
    plt.legend(loc="lower right")
    return report
