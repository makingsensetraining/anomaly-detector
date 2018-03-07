import pandas as pd

class LinearPearsonAnomalyDetector:
    def __init__(self):
        self.dataframe = None
        self.corr_threshold = None

    def __generate_dataframe(self, values):
        dict_from_values = {}
        dict_from_values['variable'] = values
        dict_from_values['index'] = range(len(values))
        return pd.DataFrame(dict_from_values)

    def __get_pearson(self, dataframe):
        return dataframe.corr()['variable']['index']

    def train(self, values):
        self.dataframe = self.__generate_dataframe(values)
        self.corr_threshold = abs(self.__get_pearson(self.dataframe))
        return self.corr_threshold

    def is_anomalous(self, value):
        new_value_row = pd.Series([ value, len(self.dataframe.index) ], ['variable', 'index'])
        new_dataframe = self.dataframe.append(new_value_row, ignore_index=True)
        return abs(self.__get_pearson(new_dataframe)) < self.corr_threshold