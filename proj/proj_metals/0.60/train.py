import os
from adsnet.trainer import Trainer

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

trainer = Trainer(
    dataset=f'{path}/images_refined.traj',
    config={
        "node_size": 128,
        "num_interactions": 3,
        "cutoff": 5.5,
        "batch_size": 12,
        "initial_lr": 1e-4,
        "max_steps": 1000000,
        "forces_weight": 0.05,
        "stop_patience": 200,
        "log_interval": 2000,
        "device": "cpu",
        "output_dir": f'{path}/model_output',
    },
)

if __name__ == '__main__':
    trainer.train()
