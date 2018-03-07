import pandas as pd

class ModeAnomalyDetector:
    def __init__(self):
        self.constant_value = None

    def train(self, values):
        dataframe = pd.DataFrame(values)

        mode = dataframe.mode()[0][0]
        self.constant_value = mode

        elements_equal_to_mode = dataframe[dataframe[0] == mode]

        return len(elements_equal_to_mode) / len(dataframe)

    def is_anomalous(self, value):
        return value != self.constant_value