import unittest
import os

from adsnet.trainer import Trainer

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def main():
    trainer = Trainer(
        dataset=f'{path}/Pt_ads_forces.traj',
        config={
            "node_size": 128,
            "num_interactions": 3,
            "cutoff": 5.5,
            "batch_size": 12,
            "initial_lr": 1e-4,
            "max_steps": 1000000,
            "forces_weight": 0.05,
            "stop_patience": 200,
            "device": "cpu",
            "output_dir": f'{path}/model_output',
            "log_interval": 10,
        },
    )
    trainer.train()


class TrainTestCase(unittest.TestCase):

    def test_Train(self):
        main()


if __name__ == '__main__':
    unittest.main()
