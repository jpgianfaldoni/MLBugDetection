from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss
from .analysis_report import AnalysisReport

def calibration_check(target_col, model, df):

    report = AnalysisReport()
    X = df.drop([target_col], axis=1)
    y_true = df[target_col]
    y_pred = model.predict_proba(X)[:,1]
    brier_score = brier_score_loss(y_true, y_pred)
    fig = CalibrationDisplay.from_estimator(model, X, y_true).figure_
    report.graphs.append(fig)
    report.model_name = type(model).__name__
    report.metrics["brier_score"] = brier_score
    
    return report
