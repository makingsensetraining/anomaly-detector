import unittest
import numpy as np
import ConstantTrendAnomalyDetector as detector

class TestConstantTrendAnomalyDetector(unittest.TestCase):
    def __get_detector(self):
        sut = detector.ConstantTrendAnomalyDetector()
        self.assertIsNotNone(sut)
        return sut

    def test_is_model_suitable_constant_arrays(self):
        sut = self.__get_detector()

        self.assertTrue(sut.is_model_suitable(np.repeat(5, 100)))
        self.assertTrue(sut.is_model_suitable(np.repeat(5, 1)))
        self.assertTrue(sut.is_model_suitable(np.repeat(1, 1)))
        self.assertTrue(sut.is_model_suitable([1.1, 1.1]))
        self.assertTrue(sut.is_model_suitable(['test string', 'test string']))

    def test_is_model_suitable_invalid_cases(self):
        sut = self.__get_detector()

        self.assertFalse(sut.is_model_suitable(['a', 'b']))
        self.assertFalse(sut.is_model_suitable([1.1, 1.2]))
        self.assertFalse(sut.is_model_suitable([1, 1, 2]))
        self.assertFalse(sut.is_model_suitable([1, 2]))
        self.assertFalse(sut.is_model_suitable([0, 2]))

    def __is_anomalous_from_dataset(self, dataset, test_value):
        sut = self.__get_detector()

        sut.train(dataset)
        return sut.is_anomalous(test_value)

    def test_is_anomalous_returns_false_on_same_value(self):
        self.assertFalse(self.__is_anomalous_from_dataset([1], 1))
        self.assertFalse(self.__is_anomalous_from_dataset([1, 1], 1))
        self.assertFalse(self.__is_anomalous_from_dataset(np.repeat(5, 100), 5))
        self.assertFalse(self.__is_anomalous_from_dataset(np.repeat('test string', 5), 'test string'))
        self.assertFalse(self.__is_anomalous_from_dataset([1.1, 1.1], 1.1))

    def __train_and_assert_is_anomalous_true(self, values, test_value):
        sut = self.__get_detector()

        sut.train(values)
        self.assertTrue(sut.is_anomalous(test_value))

    def test_is_anomalous_returns_true_on_different_value(self):
        self.assertTrue(self.__is_anomalous_from_dataset([1], 2))
        self.assertTrue(self.__is_anomalous_from_dataset([1, 1], 2))
        self.assertTrue(self.__is_anomalous_from_dataset(np.repeat(5, 100), 6))
        self.assertTrue(self.__is_anomalous_from_dataset(np.repeat('test string', 5), 'second test string'))
        self.assertTrue(self.__is_anomalous_from_dataset([1.1, 1.1], 1.2))

if __name__ == '__main__':
    unittest.main()