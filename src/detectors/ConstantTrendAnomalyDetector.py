import numpy as np

class ConstantTrendAnomalyDetector:
    def __init__(self):
        self.constant_value = np.NaN

    def __are_values_constant(self, values):
        return np.unique(values).size == 1

    def is_model_suitable(self, data):
        return self.__are_values_constant(data)

    def train(self, values):
        self.constant_value = values[0]

    def is_anomalous(self, value):
        return value != self.constant_value