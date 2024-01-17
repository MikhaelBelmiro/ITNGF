import torch

import numpy as np

from typing import Any
from utils import load_numpy
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


class TSDataset(Dataset):
    def __init__(
        self,
        ts_torch,
        lookback_period,
        pred_period,
        scaler,
    ) -> None:
        super().__init__()
        self.ts_torch = ts_torch

        self.lookback_period = lookback_period
        self.pred_period = pred_period
        self.scaler = scaler
        self.full_historical_shape = ts_torch.shape[0] - self.pred_period


    def __len__(self):
        return max(self.ts_torch.shape[0] - (self.lookback_period + self.pred_period) + 1, 0)
    
    def __getitem__(self, index):
        hist_idx_start, hist_idx_end = index, index + self.lookback_period
        fut_idx_start, fut_idx_end = (
            index + self.lookback_period,
            index + self.lookback_period + self.pred_period,
        )

        hist_ts = self.ts_torch[hist_idx_start:hist_idx_end, :]
        fut_ts = self.ts_torch[fut_idx_start:fut_idx_end, :]

        scaled_hist_ts = self.scaler.transform(hist_ts)
        scaled_fut_ts = self.scaler.transform(fut_ts)

        out_hist_ts = scaled_hist_ts.unsqueeze(-1)
        out_fut_ts = scaled_fut_ts.unsqueeze(-1)

        out_dict = {
            "hist_ts": out_hist_ts,
            "fut_ts": out_fut_ts,
        }
        return out_dict
    
def collate_fn(batch):
    hist_ts = torch.stack([b["hist_ts"] for b in batch], dim=0)
    fut_ts = torch.stack([b["fut_ts"] for b in batch], dim=0)

    out_dict = {
        "hist_ts": hist_ts,
        "fut_ts": fut_ts,
    }
    return out_dict

class TorchStandardScaler2D:
    def __init__(self, mean=None, std=None) -> None:
        self.mean = mean
        self.std = std

    def fit(self, ts):
        self.mean = torch.mean(ts)
        self.std = torch.std(ts)
        self.std[self.std==0] = 1e-5

    def transform(self, ts):
        return (ts - self.mean) / self.std

    def fit_transform(self, ts):
        self.fit(ts)
        return self.transform(ts)

    def inverse_transform(self, ts):
        return ts * self.std + self.mean

    def transfer_to_device(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)


class TorchStandardScaler1D:
    def __init__(self, mean=None, std=None) -> None:
        self.mean = mean
        self.std = std

    def fit(self, ts):
        self.mean = torch.mean(ts, dim=0)
        self.std = torch.std(ts, dim=0)
        self.std[self.std==0] = 1e-5

    def transform(self, ts):
        return (ts - self.mean) / self.std

    def fit_transform(self, ts):
        self.fit(ts)
        return self.transform(ts)

    def inverse_transform(self, ts):
        return ts * self.std + self.mean

    def transfer_to_device(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)


class LitDataModule(LightningDataModule):
    def __init__(self, config) -> None:
        super(LitDataModule, self).__init__()
        self.config = config
        self.full_ts_data = load_numpy(
            self.config["data_loading"]["preprocessed_ts_data_path"]
        ).transpose()

        train_split, val_split, test_split = self.config["data_prep"]["split"]
        train_size = round(len(self.full_ts_data) * train_split)
        val_size = round((len(self.full_ts_data) * val_split))

        train_data = torch.from_numpy(self.full_ts_data[:train_size]).float()
        val_data = torch.from_numpy(
            self.full_ts_data[
                train_size
                - self.config["data_prep"]["lookback_period"] : train_size
                + val_size
            ]
        ).float()

        test_data = torch.from_numpy(
            self.full_ts_data[
                train_size + val_size - self.config["data_prep"]["lookback_period"] :
            ]
        ).float()
        self.fit_scaler(train_data)

        self.train_ts_dataset = TSDataset(
            train_data,
            self.config["data_prep"]["lookback_period"],
            self.config["data_prep"]["pred_period"],
            self.data_scaler,
        )
        self.val_ts_dataset = TSDataset(
            val_data,
            self.config["data_prep"]["lookback_period"],
            self.config["data_prep"]["pred_period"],
            self.data_scaler,
        )
        self.test_ts_dataset = TSDataset(
            test_data,
            self.config["data_prep"]["lookback_period"],
            self.config["data_prep"]["pred_period"],
            self.data_scaler,
        )

        self.data_related_config = self.create_data_related_config()

    def fit_scaler(self, train):
        if self.config["data_prep"]["scaler"] == "1d_scaler":
            self.data_scaler = TorchStandardScaler1D()
            self.model_scaler = TorchStandardScaler1D()
        elif self.config["data_prep"]["scaler"] == "2d_scaler":
            self.data_scaler = TorchStandardScaler2D()
            self.model_scaler = TorchStandardScaler2D()
        else:
            raise ValueError(
                "scaler {} is not implemented yet".format(
                    self.config["data_prep"]["scaler"]
                )
            )
        self.data_scaler.fit(train)
        self.model_scaler.fit(train)

    def create_data_related_config(self):
        data_related_config = {
            "num_ts": self.full_ts_data.shape[1],
            "lookback_period": self.config["data_prep"]["lookback_period"],
            "pred_period": self.config["data_prep"]["pred_period"],
            "in_channels": 1,
            "out_channels": 1,
            "scaler": self.model_scaler,
        }
        return data_related_config

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_ts_dataset,
            batch_size=self.config["data_prep"]["batch_size"],
            shuffle=True,
            num_workers=self.config["data_prep"]["num_workers"],
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_ts_dataset,
            batch_size=self.config["data_prep"]["batch_size"],
            num_workers=self.config["data_prep"]["num_workers"],
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if len(self.test_ts_dataset) != 0:
            return DataLoader(
                self.test_ts_dataset,
                batch_size=self.config["data_prep"]["batch_size"],
                num_workers=self.config["data_prep"]["num_workers"],
                pin_memory=True,
                collate_fn=collate_fn,
            )
        else:
            return DataLoader(
                self.val_ts_dataset,
                batch_size=self.config["data_prep"]["batch_size"],
                num_workers=self.config["data_prep"]["num_workers"],
                pin_memory=True,
                collate_fn=collate_fn,
            )
        
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        if len(self.test_ts_dataset) != 0:
            return DataLoader(
                self.test_ts_dataset,
                batch_size=self.config["data_prep"]["batch_size"],
                num_workers=self.config["data_prep"]["num_workers"],
                pin_memory=True,
                collate_fn=collate_fn,
            )
        else:
            return DataLoader(
                self.val_ts_dataset,
                batch_size=self.config["data_prep"]["batch_size"],
                num_workers=self.config["data_prep"]["num_workers"],
                pin_memory=True,
                collate_fn=collate_fn,
            )


    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx
    ) -> Any:
        out_batch = {
            k: v.to(device) for k, v in batch.items()
        }
        return out_batch
