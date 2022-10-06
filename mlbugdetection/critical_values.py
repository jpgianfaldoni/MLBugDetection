import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from .analysis_report import AnalysisReport


def highest_and_lowest_indexes(predictions):
    dy = list(np.diff(predictions))
    negatives = list(filter(lambda x: (x < 0), dy))
    positives = list(filter(lambda x: (x > 0), dy))
    
    highest_positives = sorted(positives, reverse=True)[:3]
    lowest_negatives = sorted(negatives)[:3]

    highest_indexes = [[dy.index(x), dy.index(x)+1] for x in highest_positives]
    lowest_indexes  = [[dy.index(x), dy.index(x)+1] for x in lowest_negatives]

    return highest_indexes, lowest_indexes

# def highest_and_lowest_values(predictions):
#     highest_indexes, lowest_indexes = highest_and_lowest_indexes(predictions)



    

def find_critical_values(model, sample, feature, limit, border, step=100):
    report = AnalysisReport()
    column_values = []
    predictions = []
    range_ = np.linspace(limit, border, step)
    report.model_name = type(model).__name__
    report.analysed_feature= feature
    report.feature_range = (limit, border)
    fig = plt.figure(figsize=(6, 3), dpi=150)
    report.graphs.append(fig)

    for val in range_:
        column_values.append(val)
        sample[feature] = val
        predictions.append(model.predict_proba(sample)[0][0])

    highest_positives, lowest_negatives = highest_and_lowest_indexes(predictions)
    if len(highest_positives) > 0:
        report.metrics["positive_changes_ranges"] = []
        report.metrics["positive_changes_proba"] = []
        for indexes in highest_positives:
            range0 = round(column_values[indexes[0]],3)
            range1 = round(column_values[indexes[1]],3)
            pred0 = predictions[indexes[0]]
            pred1 = predictions[indexes[1]]
            report.metrics["positive_changes_ranges"].append((range0,range1))
            report.metrics["positive_changes_proba"].append((pred1,pred0))
            if(max(pred0, pred1) >= 0.5 and (min(pred0, pred1) < 0.5 )):
                report.metrics["classification_change_ranges"].append((range0,range1))
                report.metrics["classification_change_proba"].append((pred1,pred0))
                plt.axvline(x = range0, color = 'g', linestyle = '--', alpha = 0.5)
                plt.axvline(x = range1, color = 'g', linestyle = '--', alpha = 0.5)
    if len(lowest_negatives) > 0:
        report.metrics["negative_changes_ranges"] = []
        report.metrics["negative_changes_proba"] = []
        for indexes in lowest_negatives:
            range0 = round(column_values[indexes[0]],3)
            range1 = round(column_values[indexes[1]],3)
            pred0 = round(predictions[indexes[0]], 3)
            pred1 = round(predictions[indexes[1]], 3)
            report.metrics["negative_changes_ranges"].append((range0,range1))
            report.metrics["negative_changes_proba"].append((pred1,pred0))
            if(max(pred0, pred1) >= 0.5 and (min(pred0, pred1) < 0.5 )):
                plt.axvline(x = range0, color = 'r', linestyle = '--', alpha = 0.2)
                plt.axvline(x = range1, color = 'r', linestyle = '--', alpha = 0.2)
    if ((len(lowest_negatives) > 0) or (len(highest_positives) > 0)):
        plt.plot(column_values, predictions)
        plt.title(type(model).__name__)
        plt.xlabel(f'Feature {feature} value')
        plt.ylabel('Predict proba')
    return report


def find_several_critical_values(model, samples, feature, limit, border, step=100):
    samples = samples.copy()
    report = AnalysisReport()
    column_values = []
    range_ = np.linspace(limit, border, step)
    
    report.model_name = type(model).__name__
    report.analysed_feature = feature
    report.feature_range = (limit, border)
    fig = plt.figure(figsize=(6, 3), dpi=150)
    report.graphs.append(fig)

    predictions_dict = {}
    for i in range(samples.shape[0]):
        predictions_dict[i] = {
            "preds" : []
        }
    for val in range_:
        column_values.append(val)
        samples.loc[:, feature] = val
        samples_predictions = model.predict_proba(samples)
        for i in range(len(samples_predictions)):
            predictions_dict[i]["preds"].append(samples_predictions[i][0])

    positive_means = []
    negative_means = []
    for key in predictions_dict.keys():
        predictions_dict[key]["diff"] = list(np.diff(predictions_dict[key]["preds"]))
        predictions_dict[key]["negative_diffs"] = list(filter(lambda x: (x < 0), predictions_dict[key]["diff"]))
        predictions_dict[key]["positive_diffs"] = list(filter(lambda x: (x > 0), predictions_dict[key]["diff"]))
        if len(predictions_dict[key]["positive_diffs"]) > 0:
            positive_means.append(np.mean(sorted(predictions_dict[key]["positive_diffs"], reverse=True)[:3]))
        else:
            positive_means.append(0)

        if len(predictions_dict[key]["negative_diffs"]) > 0:
            negative_means.append(np.mean(sorted(predictions_dict[key]["negative_diffs"])[:3]))
        else:
            negative_means.append(0)
    report.metrics["positive_means"] = {}
    report.metrics["negative_means"] = {}
    
    report.metrics['positive_means']['mean'] = np.mean(positive_means)
    report.metrics['positive_means']['median'] = np.median(positive_means)
    report.metrics['positive_means']['std'] = np.std(positive_means)
    report.metrics['positive_means']['var'] = np.var(positive_means)

    report.metrics['negative_means']['mean'] = np.mean(negative_means)
    report.metrics['negative_means']['median'] = np.median(negative_means)
    report.metrics['negative_means']['std'] = np.std(negative_means)
    report.metrics['negative_means']['var'] = np.var(negative_means)

    print("Positive means:")
    print(f"\tMean: {report.metrics['positive_means']['mean']}")
    print(f"\tMedian: {report.metrics['positive_means']['median']}")
    print(f"\tSandard Deviation: {report.metrics['positive_means']['std']}")
    print(f"\tVariance: {report.metrics['positive_means']['var']}")

    print("Negative means:")
    print(f"\tMean: {report.metrics['negative_means']['mean']}")
    print(f"\tMedian: {report.metrics['negative_means']['median']}")
    print(f"\tSandard Deviation: {report.metrics['negative_means']['std']}")
    print(f"\tVariance: {report.metrics['negative_means']['var']}")

    # plt.hist(positive_means, bins=10)
    # plt.hist(negative_means, bins=10)
    return report