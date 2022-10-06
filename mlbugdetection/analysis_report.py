import os

class AnalysisReport:
 
    def __init__(self):
        self.model_name = ""
        self.analysed_feature = ""
        self.feature_range = ""
        self.metrics = {}
        self.graphs = []

    def save_graphs(self):
    # how and where the file is saved will be changed
        os.makedirs("imgs", exist_ok=True)
        if self.graphs:
            for graph in self.graphs:
                graph.savefig(f'imgs/{self.model_info["model_name"]}_{self.model_info["analysed_feature"]}_{self.model_info["feature_range"]} ', dpi=200) 
