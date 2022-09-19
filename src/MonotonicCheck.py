import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy import diff


def monotonicity_score(predictions):
    # predictions: Array of predictions made by the model

    dy = diff(predictions)
    neg_count = len(list(filter(lambda x: (x < 0), dy)))
    pos_count = len(list(filter(lambda x: (x > 0), dy)))
    zeros = len(list(filter(lambda x: (x == 0), dy)))+1
    if (len(predictions)-zeros) == 0:
        return 0
    return abs(pos_count-neg_count)/(len(predictions)-zeros)



def check_monotonicity(feature, min, max, sample, model, steps=100, plot_graph = False):
    # feature: Feature that wants to check monoticity
    # min, max: Minimum and maximum values that will be checked
    # sample: Pandas Series that will be used as a basis
    # steps: Number of iterations
    # plot_graph: Plots graph feature value x predict proba


    colValues = []
    predictions = []
    for i in np.linspace(min,max,steps):
        colValues.append(i)
        sample[feature] = i
        prediction = model.predict_proba(sample)
        predictions.append(prediction[0][0])
    
    monotonic =  (all(predictions[i] <= predictions[i + 1] for i in range(len(predictions) - 1)) or all(predictions[i] >= predictions[i + 1] for i in range(len(predictions) - 1)))
    m_score = monotonicity_score(predictions)

    print(f"Model: {(type(model).__name__)}")

    if monotonic:
        print(f"Feature {feature} has monotonic behavior between ranges {min} and {max}")
    else:
        print(f"Warning: Feature '{feature}' doesn`t have monotonic behavior between ranges {min} and {max}")
        print(f"Feature '{feature}' has a score of {m_score}")
    if plot_graph:
        plt.plot(colValues, predictions)
        plt.title(type(model).__name__)
        plt.xlabel('Feature value')
        plt.ylabel('Predict proba')
        plt.show()