import os
from adsnet.trainer import Trainer

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def main():
    trainer = Trainer(
        model="adsnet",
        config={
            "device": "cpu",
            "num_channels": 128,
            "num_layers": 3,
            "cutoff": 5.5,
            "batch_size": 12,
            "initial_lr": 1e-4,
            "max_steps": 1000000,
            "forces_weight": 0.90,
            "stop_patience": 800,
            "log_interval": 2000,
            "output_dir": f"{path}/output",
            "output_log": "output.log",
            "output_model": "model.pt",
        },
        distributed=False,
    )
    trainer.train(f"{path}/dataset/train.traj")


if __name__ == "__main__":
    main()
