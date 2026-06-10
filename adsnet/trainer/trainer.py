import torch
import numpy as np
import json
import logging
import math
import os
import platform
import socket
import time
import itertools

from adsnet.data import AseData, collate_atomsdata
from adsnet.models import AdsNet


DEFAULT_CONFIG = {
    # Model parameters
    "num_channels": 128,
    "num_layers": 3,
    "cutoff": 5.5,
    # Training parameters
    "device": None,
    "batch_size": 12,
    "initial_lr": 0.0001,
    "max_steps": 1000000,
    "forces_weight": 0.05,
    # Scheduler
    "plateau_scheduler": False,
    "lr_decay_factor": 0.5,
    "lr_patience": 10,
    # Early stopping
    "stop_patience": 200,
    # Data
    "val_ratio": 0.1,
    "split_file": None,
    "random_seed": 3407,
    "load_model": None,
    # Output
    "output_dir": "output",
    "output_log": "output.log",
    "output_model": "model.pt",
    "log_interval": 10,
    "save_config": False,
    "save_split": False,
}


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, best_loss):
        if val_loss - best_loss > self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def forces_criterion(predicted, target, reduction="mean"):
    diff = predicted - target
    total_squared_norm = torch.linalg.norm(diff, dim=1)
    if reduction == "mean":
        return torch.mean(total_squared_norm)
    elif reduction == "sum":
        return torch.sum(total_squared_norm)
    else:
        raise ValueError("Reduction must be 'mean' or 'sum'")



def split_data(dataset, config):
    if config["split_file"]:
        with open(config["split_file"], "r") as fp:
            splits = json.load(fp)
    else:
        datalen = len(dataset)
        num_validation = int(math.ceil(datalen * config["val_ratio"]))
        indices = np.random.permutation(len(dataset))
        splits = {
            "train": indices[num_validation:].tolist(),
            "validation": indices[:num_validation].tolist(),
        }

    if config.get("save_split"):
        with open(os.path.join(config["output_dir"], "datasplits.json"), "w") as f:
            json.dump(splits, f)

    datasplits = {}
    for key, indices in splits.items():
        datasplits[key] = torch.utils.data.Subset(dataset, indices)
    return datasplits


def eval_model(model, dataloader, device, forces_weight):
    energy_running_ae = 0
    energy_running_se = 0
    forces_running_c_ae = 0
    forces_running_c_se = 0
    running_loss = 0
    count = 0
    forces_count = 0
    criterion = torch.nn.MSELoss()

    for batch in dataloader:
        device_batch = {
            k: v.to(device=device, non_blocking=True) for k, v in batch.items()
        }
        out = model(device_batch.copy())

        count += batch["ads_energy"].shape[0]
        energy_loss = criterion(out["ads_energy"], device_batch["ads_energy"]).item()

        if bool(forces_weight):
            forces_count += batch['ads_forces'].shape[0]
            forces_loss = forces_criterion(out["ads_forces"], device_batch["ads_forces"]).item()
        else:
            forces_loss = 0

        total_loss = forces_weight * forces_loss + (1 - forces_weight) * energy_loss
        running_loss += total_loss * batch["ads_energy"].shape[0]

        outputs = {key: val.detach().cpu().numpy() for key, val in out.items()}
        energy_targets = batch["ads_energy"].detach().cpu().numpy()
        energy_running_ae += np.sum(np.abs(energy_targets - outputs["ads_energy"]), axis=0)
        energy_running_se += np.sum(
            np.square(energy_targets - outputs["ads_energy"]), axis=0
        )

        if bool(forces_weight):
            forces_targets = batch["ads_forces"].detach().cpu().numpy()
            forces_diff = forces_targets - outputs["ads_forces"]
            forces_running_c_ae += np.sum(np.abs(forces_diff))
            forces_running_c_se += np.sum(np.square(forces_diff))

    energy_mae = energy_running_ae / count
    energy_rmse = np.sqrt(energy_running_se / count)

    if bool(forces_weight):
        forces_c_mae = forces_running_c_ae / (forces_count * 3)
        forces_c_rmse = np.sqrt(forces_running_c_se / (forces_count * 3))
    else:
        forces_c_mae = forces_c_rmse = 0

    total_loss = running_loss / count

    return {
        "energy_mae": energy_mae,
        "energy_rmse": energy_rmse,
        "forces_mae": forces_c_mae,
        "forces_rmse": forces_c_rmse,
        "sqrt(total_loss)": np.sqrt(total_loss),
    }


class Trainer:
    """Trainer class for AdsNet models."""

    def __init__(self, model="adsnet", config=None, distributed=False):
        """
        Initialize the trainer.

        Args:
            model (str): Model type (currently only "adsnet")
            config (dict, optional): Configuration overrides for DEFAULT_CONFIG
            distributed (bool): Whether to use distributed training
        """
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        self.model_type = model.lower()
        self.distributed = distributed
        self.device = torch.device(self.config["device"]) if self.config.get("device") else None

    def train(self, dataset_path=None):
        """
        Train the model on a dataset.

        Args:
            dataset_path (str): Path to .traj file
        """
        if dataset_path is None:
            dataset_path = self.config.get("dataset")
            if not dataset_path:
                raise ValueError("Dataset path must be provided via train() argument or in config['dataset']")

        config = self.config
        setup_seed(config["random_seed"])

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(config["output_dir"], exist_ok=True)
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
            handlers=[
                logging.FileHandler(
                    os.path.join(config["output_dir"], config["output_log"]), mode="a"
                ),
                logging.StreamHandler(),
            ],
        )

        # System info
        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"Running on {socket.gethostname()}")
        logging.info(f"Hardware architecture: {platform.machine()}")

        if config["save_config"]:
            with open(os.path.join(config["output_dir"], "arguments.json"), "w") as f:
                json.dump(config, f, default=str)

        torch.tensor([0], device=self.device)

        # Data
        logging.info(f"Loading data from {dataset_path}")
        dataset = AseData(
            dataset_path,
            cutoff=config["cutoff"],
            compute_forces=bool(config["forces_weight"]),
        )

        datasplits = split_data(dataset, config)

        train_loader = torch.utils.data.DataLoader(
            datasplits["train"],
            config["batch_size"],
            sampler=torch.utils.data.RandomSampler(datasplits["train"]),
            collate_fn=collate_atomsdata,
            drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            datasplits["validation"],
            config["batch_size"],
            collate_fn=collate_atomsdata,
            drop_last=True,
        )

        logging.info(f"Dataset size: {len(dataset)}, training set size: {len(datasplits['train'])}, validation set size: {len(datasplits['validation'])}")
        logging.info(f"Compute forces: {bool(config['forces_weight'])}")

        # Model
        net = AdsNet(
            num_layers=config["num_layers"],
            num_channels=config["num_channels"],
            cutoff=config["cutoff"],
            compute_forces=bool(config["forces_weight"]),
        )
        net.to(self.device)
        total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        total_memory = sum(p.element_size() * p.numel() for p in net.parameters() if p.requires_grad)
        logging.info(f"Total trainable parameters: {total_params}")
        logging.info(f"Total memory of the model: {total_memory / 1024**2:.2f} MB")

        # Configuration summary
        logging.info("---------------- Configuration Settings ----------------")
        logging.info(f"Model Type (model): {self.model_type}")
        logging.info(f"Number of Channels (num_channels): {config['num_channels']}")
        logging.info(f"Number of Layers (num_layers): {config['num_layers']}")
        logging.info(f"Cutoff Radius (cutoff): {config['cutoff']}")
        logging.info(f"Batch Size (batch_size): {config['batch_size']}")
        logging.info(f"Learning Rate (initial_lr): {config['initial_lr']}")
        logging.info(f"Forces Weight (forces_weight): {config['forces_weight']}")
        logging.info(f"Max Steps (max_steps): {config['max_steps']}")
        logging.info(f"Log Interval (log_interval): {config['log_interval']}")
        logging.info(f"Early Stopping Patience (stop_patience): {config['stop_patience']}")
        logging.info(f"Plateau Scheduler (plateau_scheduler): {config['plateau_scheduler']}")
        logging.info(f"Random Seed (random_seed): {config['random_seed']}")
        logging.info(f"Distributed Training (distributed): {self.distributed}")
        logging.info(f"Output Directory (output_dir): {config['output_dir']}")
        logging.info(f"Output Log File (output_log): {config['output_log']}")
        logging.info(f"Output Model File (output_model): {config['output_model']}")


        optimizer = torch.optim.Adam(net.parameters(), lr=config["initial_lr"])
        criterion = torch.nn.HuberLoss(delta=1.0)
        if config["plateau_scheduler"]:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min',
                factor=config["lr_decay_factor"],
                patience=config["lr_patience"],
            )
        else:
            scheduler_fn = lambda step: 0.96 ** (step / 100000)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_fn)
        early_stop = EarlyStopping(patience=config["stop_patience"])

        running_loss = 0
        running_loss_count = 0
        prev_loss = None
        best_val_loss = np.inf
        step = 0
        training_time = 0

        if config["load_model"] and os.path.exists(config["load_model"]):
            logging.info(f"Load model from {config['load_model']}")
            state_dict = torch.load(config["load_model"], map_location=self.device)
            net.load_state_dict(state_dict["model"])
            step = state_dict["step"]
            best_val_loss = state_dict["best_val_loss"]
            optimizer.load_state_dict(state_dict["optimizer"])
            scheduler.load_state_dict(state_dict["scheduler"])

        logging.info("---------------------- Training ------------------------")
        for epoch in itertools.count():
            for batch_host in train_loader:
                start = time.time()
                batch = {
                    k: v.to(device=self.device, non_blocking=True)
                    for (k, v) in batch_host.items()
                }

                optimizer.zero_grad()

                outputs = net(batch.copy())
                energy_loss = criterion(outputs["ads_energy"], batch["ads_energy"])
                if config["forces_weight"]:
                    f_loss = forces_criterion(outputs['ads_forces'], batch['ads_forces'])
                else:
                    f_loss = 0.0
                total_loss = (
                    config["forces_weight"] * f_loss
                    + (1 - config["forces_weight"]) * energy_loss
                )
                total_loss.backward()
                optimizer.step()
                running_loss += total_loss.item() * batch["ads_energy"].shape[0]
                running_loss_count += batch["ads_energy"].shape[0]
                training_time += time.time() - start

                if (step % config["log_interval"] == 0) or ((step + 1) == config["max_steps"]):
                    eval_start = time.time()
                    train_loss = running_loss / running_loss_count
                    running_loss = running_loss_count = 0

                    eval_dict = eval_model(net, val_loader, self.device, config["forces_weight"])

                    if any(np.isnan(v) for v in eval_dict.values()):
                        logging.warning("NaN detected in evaluation metrics at step={}, stopping training model.".format(step))
                        return best_val_loss

                    eval_formatted = ", ".join(
                        ["{}={:.3f}".format(k, v) for (k, v) in eval_dict.items()]
                    )
                    eval_loss = np.square(eval_dict["sqrt(total_loss)"])
                    smooth_loss = eval_loss if prev_loss is None else 0.9 * eval_loss + 0.1 * prev_loss
                    prev_loss = smooth_loss

                    logging.info(
                        "step={}, {}, sqrt(train_loss)={:.3f}, sqrt(smooth_loss)={:.3f}, patience={:3d}, training time={:.3f} min, eval time={:.3f} min".format(
                            step,
                            eval_formatted,
                            math.sqrt(train_loss),
                            math.sqrt(smooth_loss),
                            early_stop.counter,
                            training_time / 60,
                            (time.time() - eval_start) / 60,
                        )
                    )
                    training_time = 0

                    if config["plateau_scheduler"]:
                        scheduler.step(smooth_loss)

                    if not early_stop(math.sqrt(smooth_loss), best_val_loss):
                        best_val_loss = math.sqrt(smooth_loss)
                        torch.save(
                            {
                                "model": net.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "step": step,
                                "best_val_loss": best_val_loss,
                                "num_channels": config["num_channels"],
                                "num_layers": config["num_layers"],
                                "cutoff": config["cutoff"],
                            },
                            os.path.join(config["output_dir"], config["output_model"]),
                        )
                    else:
                        logging.info("Early stop reached, exiting")
                        return best_val_loss

                step += 1

                if not config["plateau_scheduler"]:
                    scheduler.step()

                if step >= config["max_steps"]:
                    logging.info("Max steps reached, exiting")
                    torch.save(
                        {
                            "model": net.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "step": step,
                            "best_val_loss": best_val_loss,
                            "num_channels": config["num_channels"],
                            "num_layers": config["num_layers"],
                            "cutoff": config["cutoff"],
                        },
                        os.path.join(config["output_dir"], "exit_" + config["output_model"]),
                    )
                    return best_val_loss

    def evaluate(self, dataloader):
        return eval_model(self.model, dataloader, self.device, self.config["forces_weight"])
