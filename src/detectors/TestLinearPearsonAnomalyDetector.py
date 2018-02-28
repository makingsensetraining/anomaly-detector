import unittest
import numpy as np
import LinearPearsonAnomalyDetector as detector

class TestLinearPearsonAnomalyDetector(unittest.TestCase):
    def __get_detector(self):
        sut = detector.LinearPearsonAnomalyDetector()
        self.assertIsNotNone(sut)
        return sut

    def test_is_model_suitable_false_on_constant(self):
        sut = self.__get_detector()

        self.assertFalse(sut.is_model_suitable([0]))
        self.assertFalse(sut.is_model_suitable([1, 1]))

    def test_is_model_suitable_linear(self):
        sut = self.__get_detector()

        self.assertTrue(sut.is_model_suitable([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        self.assertTrue(sut.is_model_suitable([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]))
        self.assertTrue(sut.is_model_suitable([2, 4, 6, 8, 10]))

    def test_is_model_suitable_non_linear(self):
        sut = self.__get_detector()

        self.assertFalse(sut.is_model_suitable([2, 4, 8, 16, 32, 64, 128, 256]))
        self.assertFalse(sut.is_model_suitable([2, 4, 6, 8, 10, 8, 6, 4, 2]))
        self.assertFalse(sut.is_model_suitable([2, 2, 2, 2, 2, 2]))
        self.assertFalse(sut.is_model_suitable([2, 4, 6, 8, 10, 12, 5]))

    def __is_anomalous_from_dataset(self, dataset, new_value):
        sut = self.__get_detector()

        sut.train(dataset)
        return sut.is_anomalous(new_value)

    def test_is_anomalous_false_cases(self):
        self.assertFalse(self.__is_anomalous_from_dataset([1, 2, 3, 4, 5, 6], 7))
        self.assertFalse(self.__is_anomalous_from_dataset([2, 3, 4, 5, 6, 7], 8))
        self.assertFalse(self.__is_anomalous_from_dataset([2, 4, 6, 8], 10))
        self.assertFalse(self.__is_anomalous_from_dataset([2, 4, 5, 7, 9, 11, 12, 14], 16))

    def test_is_anomalous_true_cases(self):
        self.assertTrue(self.__is_anomalous_from_dataset([1, 2, 3, 4, 5, 6], 8))
        self.assertTrue(self.__is_anomalous_from_dataset([2, 3, 4, 5, 6, 7], 6))
        self.assertTrue(self.__is_anomalous_from_dataset([2, 4, 6, 8], 7))
        self.assertTrue(self.__is_anomalous_from_dataset([2, 4, 6, 8, 10, 12, 14], 25))
        self.assertTrue(self.__is_anomalous_from_dataset([2, 4, 6, 8, 10, 12, 14], 15))
        self.assertTrue(self.__is_anomalous_from_dataset([2, 4, 6, 8, 10, 12, 14], 6))
        self.assertTrue(self.__is_anomalous_from_dataset([2, 4, 5, 7, 9, 11, 12, 14], 17))
        self.assertTrue(self.__is_anomalous_from_dataset([2, 4, 5, 7, 9, 11, 12, 14], 14))
        self.assertTrue(self.__is_anomalous_from_dataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 15))
        self.assertTrue(self.__is_anomalous_from_dataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1))

if __name__ == '__main__':
    unittest.main()