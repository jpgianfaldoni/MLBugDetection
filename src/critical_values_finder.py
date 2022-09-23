import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy import diff


def highest_and_lowest_indexes(predictions):
    dy = list(diff(predictions))
    negatives = list(filter(lambda x: (x < 0), dy))
    positives = list(filter(lambda x: (x > 0), dy))
    
    highest_positives = sorted(positives, reverse=True)[:3]
    lowest_negatives = sorted(negatives)[:3]

    highest_indexes = [[dy.index(x), dy.index(x)+1] for x in highest_positives]
    lowest_indexes  = [[dy.index(x), dy.index(x)+1] for x in lowest_negatives]

    return highest_indexes, lowest_indexes

def find_critical_values(model, sample, feature, limit, border, step=100, plot_graph=True):
    #media e dp
    column_values = []
    predictions = []
    range_ = np.linspace(limit, border, step)

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
            print(f"\tFrom values {range0} to {range1} : Predict proba diff = {pred1 - pred0}")
            if(max(pred0, pred1) >= 0.5 and (min(pred0, pred1) < 0.5 )):
                print(f"\tWarning, prediction has changed")
                plt.axvline(x = range0, color = 'green', linestyle = 'dashed', label="Positive classification change")
                plt.axvline(x = range1, color = 'green', linestyle = 'dashed')
    if len(lowest_negatives) > 0:
        print(f"Highest negatives changes identified on feature '{feature}': ")
        for indexes in lowest_negatives:
            range0 = round(column_values[indexes[0]],3)
            range1 = round(column_values[indexes[1]],3)

            pred0 = round(predictions[indexes[0]], 3)
            pred1 = round(predictions[indexes[1]], 3)
            print(f"\tFrom values {range0} to {range1} : Predict proba diff = {pred1 - pred0}")
            if(max(pred0, pred1) >= 0.5 and (min(pred0, pred1) < 0.5 )):
                print(f"\tWarning, prediction has changed")
                plt.axvline(x = range0, color = 'red', linestyle = 'dashed', label="Negative classification change")
                plt.axvline(x = range1, color = 'red', linestyle = 'dashed')
    if ((len(lowest_negatives) > 0) or (len(highest_positives) > 0)) and plot_graph:
        plt.plot(column_values, predictions, label="Predictions curve")
        plt.title(f"Critical values analysis of '{feature}' in {type(model).__name__} Model")
        plt.xlabel(f"Feature '{feature}' value")
        plt.ylabel("Predict Probability")
        plt.legend(loc='center left', bbox_to_anchor=(0.7, 0.3),
          fancybox=True, shadow=True)
        plt.show()