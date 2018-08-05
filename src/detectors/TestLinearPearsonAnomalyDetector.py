import unittest
import numpy as np
import LinearPearsonAnomalyDetector as detector

class TestLinearPearsonAnomalyDetector(unittest.TestCase):
    def __get_detector(self):
        sut = detector.LinearPearsonAnomalyDetector()
        self.assertIsNotNone(sut)
        return sut

    def test_train_returns_level_of_confidence(self):
        sut = self.__get_detector()

        self.assertEqual(1, sut.train([1, 2, 3, 4, 5, 6]))
        self.assertEqual(1, sut.train([6, 5, 4, 3, 2, 1]))
        self.assertEqual(0, sut.train([81, 27, 9, 3, 1, 3, 9, 27, 81]))

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