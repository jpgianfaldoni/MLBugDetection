import os

class AnalysisReport:
    """Analysis Report Class
    All library functions returns a Analysis Report Object

    Parameters
    ----------

    model_name : str, default = ''
        Name of the model being analysed.
    
    analysed_feature : str, default = ''
        Name of the feature being analysed.
    
    feature_range : tuple, default = ()
        Range of values of the feature being analysed: (start, stop).
    
    metrics : dictionary, default = {}
        Dictionary with all the calculated metrics. 
        All the possible metrics that can be calculated are:

        'monotonic' : bool
             If the list of values is monotonic.

        'monotonic_score': float
            MSE between the list of values and it`s closest monotonic aproximation.

        'positive_changes_ranges' : List
            List of feature ranges that resulted in the biggest positive changes in the model`s prediction probability.
        
        'positive_changes_proba' : List
            List of biggest positive variations in the model`s prediction probability.
        
        'negative_changes_ranges' : List
            List of feature ranges that resulted in the biggest negative changes in the model`s prediction probability.

        'negative_changes_proba' : List
            List of biggest negative variations in the model`s prediction probability.
        
        'classification_change_ranges' : List
            List of feature ranges that resulted in a change of the model`s classification.
        
        'classification_change_proba' : List
            List of prediction probability values before and after the classification change.
        
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
            Same as "positive_means", but for negative variations in the prediction probabilities.
        
        'sanity' : bool
            If the model is sane or not.
    
        'sanity_indexes' : List
            List of indexes of the samples that were misclassified.
    
    graphs : List, default = []
        List of all the figures created.
    """

    def __init__(self):
        self.model_name = ""
        self.analysed_feature = ""
        self.feature_range = ()
        self.metrics = {}
        self.graphs = []

    def save_graphs(self):
        os.makedirs("imgs", exist_ok=True)
        if self.graphs:
            for graph in self.graphs:
                graph.savefig(f'imgs/{self.model_name}_{self.analysed_feature}_{self.feature_range} ', dpi=200) 

        """Saves all figures contained on the 'graphs' parameter on a folder called 'imgs'.
        If the folder does not exists, it will be created automatically.
        """
