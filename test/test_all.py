import unittest
import os
import sys

path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(path, ".."))

from test_data import AseDataTestCase
from test_train import TrainTestCase
from test_predict import main as predict_main
from test_relax import main as relax_main


class PredictTestCase(unittest.TestCase):
    def test_predict(self):
        predict_main()


class RelaxTestCase(unittest.TestCase):
    def test_relax(self):
        relax_main()


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(AseDataTestCase))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TrainTestCase))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(PredictTestCase))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(RelaxTestCase))
    unittest.TextTestRunner(verbosity=2).run(suite)
