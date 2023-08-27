"""Model loading module."""

import pkg_resources


def load_models_path():
    """
        Loads MPLClassifer and the KNN models paths trained with the Breast Cancer classification Dataset

        Returns
        -------
        Sklearn MPLClassifier and KNN models
    """
    stream_knn = pkg_resources.resource_stream(__name__, 'models/KNNBreastCancer.pkl')
    stream_nn = pkg_resources.resource_stream(__name__, 'models/NNBreastCancer.pkl')
    
    return stream_knn.name.replace("\\", "/"), stream_nn.name.replace("\\", "/")