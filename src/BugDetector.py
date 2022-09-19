import pickle
from typing import List
import pandas as pd
import numpy as np

class BugDetector:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path
        self.data = self.load_data()
        self.model = self.load_model()
        self.examples = []

    def create_examples(self, columns: List, max_offset = 1.5):
        #creates examples to be used on other functions
        pass

    def check_upper_limit_data(self, upperlim, columns: List):
        pass

    def check_lower_limit_data(self, lowerlim, columns: List):
        #checks if there is a value lower than the limit in data
        pass

    def check_specific_values(self, output, values: List):
        #checks if specific values have specific output both in data and model
        pass


    def find_most_important_feature(min, max):
        #Manual
        #Lime?
        #Feature isolada
        pass

    def check_monotonic(self, column, min, max, sample, positive = True):
        #checks if feature has monotonic relationship with output (positive or negative)
        pass

    def check_instance():
        #check if instance has specific output
        pass

    def model_calibration():
        #quao calibrado
        #ponto de corte
        pass

class DataFrameBugDetector(BugDetector):

    def create_examples(self, columns: List, max_offset = 1.5):
        #creates examples to be used on other functions
        pass

    def check_upper_limit_data(self, upperlim, columns: List):
        pass

    def check_lower_limit_data(self, lowerlim, columns: List):
        #checks if there is a value lower than the limit in data
        pass

    def check_specific_values(self, output, values: List):
        #checks if specific values have specific output both in data and model
        pass

    def check_monotonic(self, column, positive = True):
        #checks if feature has monotonic relationship with output (positive or negative)
        pass

    def load_data(self):
        #loads data from path
        return pd.read_csv(self.data_path)


class ModelBugDetector(BugDetector):
    
    def create_examples(self, columns: List, max_offset = 1.5):
        #creates examples to be used on other functions
        pass

    def check_upper_limit_data(self, upperlim, columns: List):
        pass

    def check_lower_limit_data(self, lowerlim, columns: List):
        #checks if there is a value lower than the limit in data
        pass

    def check_specific_values(self, output, values: List):
        #checks if specific values have specific output both in data and model
        pass

    def check_monotonic(self, column, positive = True):
        #checks if feature has monotonic relationship with output (positive or negative)
        pass

    def load_model(self):
        #loads model from path
        with open(self.model_path, 'rb') as f:
            return pickle.load(f)
    


    

    



    