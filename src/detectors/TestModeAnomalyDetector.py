import unittest
import numpy as np
import ModeAnomalyDetector as detector

class TestModeAnomalyDetector(unittest.TestCase):
    def __get_detector(self):
        sut = detector.ModeAnomalyDetector()
        self.assertIsNotNone(sut)
        return sut

    def test_train_returns_level_of_confidence(self):
        sut = self.__get_detector()

        self.assertEqual(1, sut.train(np.repeat(5, 100)))
        self.assertEqual(1, sut.train([1.1, 1.1]))
        self.assertEqual(1, sut.train(['test string', 'test string']))
        self.assertEqual(0.9, sut.train([1, 1, 1, 1, 1, 1, 1, 1, 1, 2]))
        self.assertEqual(0.25, sut.train([1, 2, 3, 4]))

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
        self.assertFalse(self.__is_anomalous_from_dataset([1, 1, 2], 1))

    def test_is_anomalous_returns_true_on_different_value(self):
        self.assertTrue(self.__is_anomalous_from_dataset([1], 2))
        self.assertTrue(self.__is_anomalous_from_dataset([1, 1], 2))
        self.assertTrue(self.__is_anomalous_from_dataset(np.repeat(5, 100), 6))
        self.assertTrue(self.__is_anomalous_from_dataset(np.repeat('test string', 5), 'second test string'))
        self.assertTrue(self.__is_anomalous_from_dataset([1.1, 1.1], 1.2))
        self.assertTrue(self.__is_anomalous_from_dataset([1, 1, 2], 2))
        self.assertTrue(self.__is_anomalous_from_dataset([1, 1, 2], 3))

if __name__ == '__main__':
    unittest.main()