import pickle
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from .analysis_report import AnalysisReport

def highest_and_lowest_indexes(predictions, keep_n=3):
    '''Return indexes of highest changes (positive or negative) 
       in predictions 

    Parameters
    ----------
    predictions : list
        Array that contains predictions to be analysed

    keep_n : int
        Number of values that are to be keeped in each list
    '''
    dy = list(np.diff(predictions))
    negatives = list(filter(lambda x: (x < 0), dy))
    positives = list(filter(lambda x: (x > 0), dy))
    
    highest_positives = sorted(positives, reverse=True)[:keep_n]
    lowest_negatives = sorted(negatives)[:keep_n]

    highest_indexes = [[dy.index(x), dy.index(x)+1] for x in highest_positives]
    lowest_indexes  = [[dy.index(x), dy.index(x)+1] for x in lowest_negatives]

    return highest_indexes, lowest_indexes

def find_critical_values(model, sample, feature : str, start, stop, steps=100, keep_n=3):
    '''Critical Values Finder
        Finds highest changes (positive or negative) in predict_proba 
        over an specified inteval [`start`, `stop`].

    Parameters
    ----------
    model : sklearn model
        Model already trained and tested from scikit-learn.

    sample : pandas DataFrame
        A single row of the dataframe that will be used for the analysis.

    feature : str
        Feature of dataframe that will be analysed.
    
    start : int
        The starting value of the feature's interval.
    
    stop : int
        The end value of the feature's interval.
    
    steps : int, default=100
        Number of values that will be atributed to the analysed feature. Must be non-negative.

    keep_n : int, default=3
        Number of values that are to be keeped in each list

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
        
        'positive_changes_proba' : List
            List of feature ranges that resulted in the biggest positive 
            changes in the model`s prediction probability.

         'positive_changes_proba' : List
            List of biggest positive variations in the model`s prediction 
            probability.
        
        'negative_changes_ranges' : List
            List of feature ranges that resulted in the biggest negative 
            changes in the model`s prediction probability.

        'negative_changes_proba' : List
            List of biggest negative variations in the model`s prediction 
            probability.
        
        'classification_change_ranges' : List
            List of feature ranges that resulted in a change of the model`s 
            classification.
        
        'classification_change_proba' : List
            List of prediction probability values before and after the 
            classification change.

    graphs : List
            List of all the figures created.
        
    '''
    if type(model) == str:
        with open(model, 'rb') as f:
            model = pickle.load(f)
            
    report = AnalysisReport()
    column_values = []
    predictions = []
    range_ = np.linspace(start, stop, steps)
    report.model_name = type(model).__name__
    report.analysed_feature= feature
    report.feature_range = (start, stop)
    fig = plt.figure(figsize=(6, 3), dpi=150)
    report.graphs.append(fig)

    for val in range_:
        column_values.append(val)
        sample[feature] = val
        predictions.append(model.predict_proba(sample)[0][0])

    highest_positives, lowest_negatives = highest_and_lowest_indexes(predictions, keep_n=keep_n)
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


def find_several_critical_values(model, samples, feature, start, stop, steps=100, bins=15, keep_n=5):
    '''Critical Values Finder in Several Samples
        Finds mean, median, standard deviation, variation of the critical values
        found in the samples over an specified inteval [`start`, `stop`].

    Parameters
    ----------
    model : sklearn model
        Model already trained and tested from scikit-learn.

    samples : pandas DataFrame
        One or more rows of the dataframe that will be used for the analysis.

    feature : str
        Feature of dataframe that will be analysed.
    
    start : int
        The starting value of the feature's interval.
    
    stop : int
        The end value of the feature's interval.
    
    steps : int, default=100
        Number of samples to generate. Must be non-negative.

    bins : int, default=15
        It defines the number of equal-width bins in the range.

    keep_n : int, default=5
        Number of the highest values to use for mean, median, std, var calculation.

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

        'positive_means' : dictionary
            Contains the following:

            'mean' : float
                Mean of the all the positive changes means
        
            'median' : float
                Median of the all the positive changes means
        
            'std' : float
                Standard Deviation of the all the positive changes means
        
            'var' : float
                Variation of the all the positive changes means
        
        'negative_means' : dictionary
            Contains the following:

            'mean' : float
                Mean of the all the negative changes means
        
            'median' : float
                Median of the all the negative changes means
        
            'std' : float
                Standard Deviation of the all the negative changes means
        
            'var' : float
                Variation of the all the negative changes means
        
    graphs : List
        List of all the figures created.
    '''
    samples = samples.copy()
    report = AnalysisReport()
    column_values = []
    predictions = []
    range_ = np.linspace(start, stop, steps)
    
    report.model_name = type(model).__name__
    report.analysed_feature = feature
    report.feature_range = (start, stop)

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
    means = []
    for key in predictions_dict.keys():
        predictions_dict[key]["diff"] = list(np.diff(predictions_dict[key]["preds"]))

        predictions_dict[key]["positive_diffs"] = list(filter(lambda x: (x > 0), predictions_dict[key]["diff"]))
        predictions_dict[key]["negative_diffs"] = list(filter(lambda x: (x <= 0), predictions_dict[key]["diff"]))
        
        highest_positive_diffs = sorted(predictions_dict[key]["positive_diffs"], reverse=True)[:keep_n]
        highest_negative_diffs = sorted(predictions_dict[key]["negative_diffs"])[:keep_n]

        positive_means.append(np.mean(highest_positive_diffs) if len(highest_positive_diffs) > 0 else 0)
        negative_means.append(np.mean(highest_negative_diffs) if len(highest_negative_diffs) > 0 else 0)
        
    report.metrics["positive_means"] = {}
    report.metrics["negative_means"] = {}
    
    report.metrics['positive_means']['mean'] = np.nanmean(positive_means)
    report.metrics['positive_means']['median'] = np.nanmedian(positive_means)
    report.metrics['positive_means']['std'] = np.nanstd(positive_means)
    report.metrics['positive_means']['var'] = np.nanvar(positive_means)

    report.metrics['negative_means']['mean'] = np.nanmean(negative_means)
    report.metrics['negative_means']['median'] = np.nanmedian(negative_means)
    report.metrics['negative_means']['std'] = np.nanstd(negative_means)
    report.metrics['negative_means']['var'] = np.nanvar(negative_means)

    print("Positive means:")
    print(f"\tMean: {report.metrics['positive_means']['mean']}")
    print(f"\tMedian: {report.metrics['positive_means']['median']}")
    print(f"\tStandard Deviation: {report.metrics['positive_means']['std']}")
    print(f"\tVariance: {report.metrics['positive_means']['var']}")

    print("Negative means:")
    print(f"\tMean: {report.metrics['negative_means']['mean']}")
    print(f"\tMedian: {report.metrics['negative_means']['median']}")
    print(f"\tStandard Deviation: {report.metrics['negative_means']['std']}")
    print(f"\tVariance: {report.metrics['negative_means']['var']}")


    fig, ax= plt.subplots(1,2, figsize=(16,4))
    ax[0].set(xlabel="Mean", ylabel="Frequency")
    ax[0].hist(positive_means, bins=bins)
    ax[0].set_title("Histogram of positive means")
    ax[1].set(xlabel="Mean", ylabel="Frequency")
    ax[1].hist(negative_means, bins=bins)
    ax[1].set_title("Histogram of negative means")
    report.graphs.append(fig)
    # report.save_graphs()
    return report
