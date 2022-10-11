from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss
from .analysis_report import AnalysisReport

def calibration_check(model, samples, target):
    '''Calibration check for a model
        Analyzes the calibration of a model with samples and uses the
        Brier score loss as a metric for the calibration.

    Parameters
    ----------
    model : sklearn model
        The model to be used for prediction.

    samples : pandas DataFrame
        The samples to be used for prediction.
    
    target : str
        The name of the column containing the target variable.

    Returns
    -------
    AnalysisReport object with following attributes:
        For more information:
        >>> from mlbugdetection.analysis_report import AnalysisReport
        >>> help(AnalysisReport)

    model_name : string
        Name of the model being analysed.
    
    analysed_feature : string
        Name of the feature being analysed.
        For the calibration, we don't need this, so it's always empty.
    
    feature_range : tuple
        Range of values of the feature being analysed: (start, stop).
        For the calibration, we don't need this, so it's always empty.
    
    metrics : dictionary
        Dictionary with all the calculated metrics, such as:
        
        'brier_score' : float
            Brier score loss of the calibration curve. It is a MSE 
            between the perfect calibration and the model's calibration. 

    graphs : List
            List of all the figures created.

    '''
    report = AnalysisReport()
    X = samples.drop([target], axis=1)
    y_true = samples[target]
    y_pred = model.predict_proba(X)[:,1]
    brier_score = brier_score_loss(y_true, y_pred)
    fig = CalibrationDisplay.from_estimator(model, X, y_true).figure_
    report.graphs.append(fig)
    report.model_name = type(model).__name__
    report.metrics["brier_score"] = brier_score
    
    return report
