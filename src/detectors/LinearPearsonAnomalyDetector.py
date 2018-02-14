import pandas as pd

class LinearPearsonAnomalyDetector:
    def __init__(self, initial_corr_threshold = 0.9):
        self.initial_corr_threshold = initial_corr_threshold
        self.dataframe = None
        self.corr_threshold = None

    def __generate_dataframe(self, values):
        dict_from_values = {}
        dict_from_values['variable'] = values
        dict_from_values['index'] = range(len(values))
        return pd.DataFrame(dict_from_values)

    def __get_pearson(self, dataframe):
        return dataframe.corr()['variable']['index']

    def is_model_suitable(self, values):
        dataframe = self.__generate_dataframe(values)
        pearson_correlation = self.__get_pearson(dataframe)
        return abs(pearson_correlation) > self.initial_corr_threshold

    def train(self, values):
        self.dataframe = self.__generate_dataframe(values)
        self.corr_threshold = self.__get_pearson(self.dataframe)

    def is_anomalous(self, value):
        new_value_row = pd.Series([ value, len(self.dataframe.index) ], ['variable', 'index'])
        new_dataframe = self.dataframe.append(new_value_row, ignore_index=True)
        return self.__get_pearson(new_dataframe) < self.corr_threshold