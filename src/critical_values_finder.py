import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy import diff
from analysis_report import AnalysisReport



def highest_and_lowest_indexes(predictions):
    dy = list(diff(predictions))
    negatives = list(filter(lambda x: (x < 0), dy))
    positives = list(filter(lambda x: (x > 0), dy))
    
    highest_positives = sorted(positives, reverse=True)[:3]
    lowest_negatives = sorted(negatives)[:3]

    highest_indexes = [[dy.index(x), dy.index(x)+1] for x in highest_positives]
    lowest_indexes  = [[dy.index(x), dy.index(x)+1] for x in lowest_negatives]

    return highest_indexes, lowest_indexes

def find_critical_values(model, sample, feature, limit, border, step=100):
    report = AnalysisReport()
    column_values = []
    predictions = []
    range_ = np.linspace(limit, border, step)
    report.model_info["model_name"] = type(model).__name__
    report.model_info["analysed_feature"] = feature
    report.model_info["feature_range"] = (limit, border)
    fig = plt.figure(figsize=(6, 3), dpi=150)
    report.graphs.append(fig)

    for val in range_:
        column_values.append(val)
        sample[feature] = val
        predictions.append(model.predict_proba(sample)[0][0])

    highest_positives, lowest_negatives = highest_and_lowest_indexes(predictions)
    if len(highest_positives) > 0:
        print(f"\nHighest positives changes identified on feature '{feature}': ")
        for indexes in highest_positives:
            range0 = round(column_values[indexes[0]],3)
            range1 = round(column_values[indexes[1]],3)

            pred0 = predictions[indexes[0]]
            pred1 = predictions[indexes[1]]
            # print(f"\tFrom values {range0} to {range1} : diff = {pred1 - pred0}")
            report.metrics["positive_changes_ranges"].append((range0,range1))
            report.metrics["positive_changes_proba"].append((pred1,pred0))
            if(max(pred0, pred1) >= 0.5 and (min(pred0, pred1) < 0.5 )):
                report.metrics["classification_change_ranges"].append((range0,range1))
                report.metrics["classification_change_proba"].append((pred1,pred0))
                # print(f"\tWarning, prediction has changed")
                report.warnings.append(f"Warning, prediction has changed (0 to 1)")
                plt.axvline(x = range0, color = 'g', linestyle = '--', alpha = 0.5)
                plt.axvline(x = range1, color = 'g', linestyle = '--', alpha = 0.5)
    if len(lowest_negatives) > 0:
        # print(f"Lowest negatives identified on feature {feature}: ")
        for indexes in lowest_negatives:
            range0 = round(column_values[indexes[0]],3)
            range1 = round(column_values[indexes[1]],3)

            pred0 = round(predictions[indexes[0]], 3)
            pred1 = round(predictions[indexes[1]], 3)
            # print(f"\tFrom values {range0} to {range1} : diff = {pred1 - pred0}")
            report.metrics["negative_changes_ranges"].append((range0,range1))
            report.metrics["negative_changes_proba"].append((pred1,pred0))
            if(max(pred0, pred1) >= 0.5 and (min(pred0, pred1) < 0.5 )):
                # print(f"\tWarning, prediction has changed")
                report.warnings.append(f"Warning, prediction has changed (1 to 0)")
                plt.axvline(x = range0, color = 'r', linestyle = '--', alpha = 0.2)
                plt.axvline(x = range1, color = 'r', linestyle = '--', alpha = 0.2)
    if ((len(lowest_negatives) > 0) or (len(highest_positives) > 0)):
        plt.plot(column_values, predictions)
        plt.title(type(model).__name__)
        plt.xlabel(f'Feature {feature} value')
        plt.ylabel('Predict proba')
    return report
