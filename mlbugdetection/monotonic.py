import numpy as np
from matplotlib import pyplot as plt
from analysis_report import AnalysisReport

def monotonicity_mse(predictions):
    desc = np.minimum.accumulate(predictions)
    asc = np.maximum.accumulate(predictions)
    mse_desc = (np.square(predictions - desc)).mean(axis=0)
    mse_asc = (np.square(predictions - asc)).mean(axis=0)
    if min(mse_asc,mse_desc) == mse_desc:
        return desc, min(mse_asc,mse_desc)
    else:
        return asc, min(mse_asc,mse_desc)

def check_monotonicity(feature, min, max, sample, model, steps=100):
    report = AnalysisReport()
    colValues = []
    predictions = []
    for i in np.linspace(min,max,steps):
        colValues.append(i)
        sample[feature] = i
        prediction = model.predict_proba(sample)
        predictions.append(prediction[0][0])
    
    monotonic =  (all(predictions[i] <= predictions[i + 1] for i in range(len(predictions) - 1)) or all(predictions[i] >= predictions[i + 1] for i in range(len(predictions) - 1)))
    report.model_info["model_name"] = type(model).__name__
    report.model_info["analysed_feature"] = feature
    report.model_info["feature_range"] = (min, max)
    fig = plt.figure(figsize=(6, 3), dpi=150)
    report.graphs.append(fig)
    if monotonic:
        report.metrics["monotonic"] = True
        report.metrics["monotonic_score"] = 1
    else:
        report.metrics["monotonic"] = False
        monotonic_curve, m_mse_score = monotonicity_mse(predictions)
        report.metrics["monotonic_score"] = m_mse_score
        report.warnings.append(f"Feature '{feature}' doesn`t have monotonic behavior between ranges {min} and {max}")
        plt.plot(colValues, monotonic_curve, linestyle='dashed', color='red', alpha=0.7, label="Monotonic Approximation")
    plt.plot(colValues, predictions, color='blue', alpha=0.7, label="Predictions Curve")
    plt.title(f"Model: {type(model).__name__}")
    plt.xlabel('Feature value')
    plt.ylabel('Predict proba')
    plt.legend(loc="lower right")
    return report
