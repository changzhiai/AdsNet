import os
from adsnet.trainer import Trainer

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

trainer = Trainer(
    model="adsnet",
    config={
        "device": "cpu", 
        "node_size": 128,
        "num_interactions": 3,
        "cutoff": 5.5,
        "batch_size": 12,
        "initial_lr": 1e-4,
        "max_steps": 1000000,
        "forces_weight": 0.90,
        "stop_patience": 800,
        "log_interval": 2000,
        'load_model': f'{path}/output/model.pt',
        "output_dir": f'{path}/output',
        "output_log": "output.log",
        "output_model": "model.pt",
    },
    distributed=False,
)

if __name__ == '__main__':
    trainer.train(f'{path}/images_refined.traj')
