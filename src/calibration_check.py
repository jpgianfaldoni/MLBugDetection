from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss
import numpy as np
from matplotlib import pyplot as plt


def calibration_check(target_col, model, df, plot_graph = False):
    #target_col: Df column with target labels

    X = df.drop([target_col], axis=1)
    y_true = df[target_col]
    y_pred = model.predict_proba(X)[:,1]
    brier_score = brier_score_loss(y_true, y_pred)
    if plot_graph:
        disp = CalibrationDisplay.from_estimator(model, X, y_true)
        plt.title(f"Calibration curve")
        plt.show()
    print(f"Brier Score Loss: {brier_score} for model {(type(model).__name__)}")
    return brier_score