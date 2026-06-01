import os
from adsnet.trainer import Trainer

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def main():
    trainer = Trainer(
        config={
            "node_size": 128,
            "num_interactions": 3,
            "cutoff": 5.5,
            "batch_size": 12,
            "initial_lr": 1e-4,
            "max_steps": 1000000,
            "forces_weight": 0.70,
            "stop_patience": 800,
            "device": "cpu",
            "output_dir": f'{path}/output',
            "output_log": "output.log",
            "output_model": "model.pt",
            "load_model": f'{path}/output/model.pt',
            "log_interval": 10,
            "val_ratio": 0.1,
        },
        distributed=False,
    )
    trainer.train(f'{path}/dataset/train.traj')


if __name__ == '__main__':
    main()
