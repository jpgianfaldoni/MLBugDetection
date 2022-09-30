from matplotlib import pyplot as plt

class AnalysisReport:
 
    def __init__(self):
        self.model_info = {"model_name": "", "analysed_feature": "", "feature_range": ""}
        self.warnings = []
        self.errors = []
        self.metrics = {"monotonic": "", "monotonic_score": "", "brier_score": "", 
        "positive_changes_ranges" : [], "negative_changes_ranges": [], "positive_changes_proba" : [],
        "negative_changes_proba" : [], "classification_change_ranges" : [], "classification_change_proba" : []}
        self.graphs = []

    def save_graphs(self):
    # how and where the file is saved will be changed
        if self.graphs:
            for graph in self.graphs:
                graph.savefig(f'{self.model_info["model_name"]}_{self.model_info["analysed_feature"]}_{self.model_info["feature_range"]} ', dpi=200) 



    
 

    
