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

    def test_is_model_suitable_(self):
        sut = self.__get_detector()

        self.assertFalse(sut.is_model_suitable(np.array([1, 2])))
        self.assertFalse(sut.is_model_suitable(np.array([0, 2])))

    def test_train_handles_arrays(self):
        sut = self.__get_detector()

        sut.train([1, 1])
        sut.train([1])
        sut.train([0])
        sut.train(np.ones(100))

    def __train_and_assert_is_anomalous_false(self, values, test_value):
        sut = self.__get_detector()

        sut.train(values)
        self.assertFalse(sut.is_anomalous(test_value))

    def test_is_anomalous_returns_false_on_same_value(self):
        self.__train_and_assert_is_anomalous_false(np.repeat(1, 1), 1)
        self.__train_and_assert_is_anomalous_false(np.repeat(1, 2), 1)
        self.__train_and_assert_is_anomalous_false(np.repeat(5, 100), 5)

    def __train_and_assert_is_anomalous_true(self, values, test_value):
        sut = self.__get_detector()

        sut.train(values)
        self.assertTrue(sut.is_anomalous(test_value))

    def test_is_anomalous_returns_true_on_different_value(self):
        self.__train_and_assert_is_anomalous_true(np.repeat(1, 1), 2)
        self.__train_and_assert_is_anomalous_true(np.repeat(1, 2), 2)
        self.__train_and_assert_is_anomalous_true(np.repeat(5, 100), 6)

if __name__ == '__main__':
    unittest.main()