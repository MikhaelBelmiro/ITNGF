import torch

import numpy as np

from torch import nn
from lightning import LightningModule
from models.icgnn.layers import Model
from utils import create_class_instance
from typing import Any, Union
from lightning.pytorch.utilities.types import STEP_OUTPUT
from loss import rmsse_loss, mse_loss, rmse_loss
from torch.nn import BCELoss
from torchmetrics import AUROC
from copy import deepcopy


class LitModelModule(LightningModule):
    def __init__(self, data_related_config, model_hyperparam_config) -> None:
        super().__init__()
        self.config = data_related_config
        self.config.update(model_hyperparam_config["model"])
        self.other_config = deepcopy(model_hyperparam_config)

        self.model = Model(**self.config)
        self.loss = rmse_loss(self.config["scaler"])
        self.rmsse_loss = rmsse_loss(self.config["scaler"])

    def configure_optimizers(self) -> Any:
        optimizer_config = self.other_config["optimizer"]
        optimizer_config["kwargs"].update({"params": self.model.parameters()})
        optimizer_instance = create_class_instance(optimizer_config["classpath"], optimizer_config["kwargs"])

        lr_scheduler_config = self.other_config["lr_scheduler"]
        lr_scheduler_config["torch_kwargs"].update({"optimizer": optimizer_instance})
        lr_scheduler_instance = create_class_instance(lr_scheduler_config["classpath"], lr_scheduler_config["torch_kwargs"])
        lr_scheduler = {
            "scheduler": lr_scheduler_instance,
            **lr_scheduler_config["lightning_kwargs"]
        }

        out = {
            "optimizer": optimizer_instance,
            "lr_scheduler": lr_scheduler
        }

        return out

    def forward(self, batch):
        del batch["gt_ts"]

        pred = self.model(**batch)
        return pred
    
    def on_fit_start(self) -> None:
        self.config["scaler"].transfer_to_device(self.device)
        return super().on_fit_start()

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        historical_ts = batch["hist_ts"].squeeze(-1)
        gt_ts = batch["fut_ts"].squeeze(-1)
        pred_ts = self.model(batch["hist_ts"], batch["fut_ts"]).squeeze(-1)
        loss = self.loss(pred_ts, gt_ts)

        metrics_dict = {
            "t_loss": loss.item(),
            "t_rmsse": self.rmsse_loss(pred_ts, gt_ts, historical_ts)
        }
        self.log_dict(metrics_dict, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx) -> Union[STEP_OUTPUT, None]:
        historical_ts = batch["hist_ts"].squeeze(-1)
        gt_ts = batch["fut_ts"].squeeze(-1)
        pred_ts = self.model(batch["hist_ts"], batch["fut_ts"]).squeeze(-1)
        loss = self.loss(pred_ts, gt_ts)

        metrics_dict = {
            "v_loss": loss.item(),
            "v_rmsse": self.rmsse_loss(pred_ts, gt_ts, historical_ts)
        }
        self.log_dict(metrics_dict, prog_bar=True)

    def test_step(self, batch, batch_idx) -> Union[STEP_OUTPUT, None]:
        historical_ts = batch["hist_ts"].squeeze(-1)
        gt_ts = batch["fut_ts"].squeeze(-1)
        pred_ts = self.model(batch["hist_ts"], batch["fut_ts"]).squeeze(-1)
        loss = self.loss(pred_ts, gt_ts)

        metrics_dict = {
            "test_loss": loss.item(),
            "test_rmsse": self.rmsse_loss(pred_ts, gt_ts, historical_ts)
        }
        self.log_dict(metrics_dict, prog_bar=True)
