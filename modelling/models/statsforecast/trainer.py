import os

import numpy as np
import pandas as pd

from loss import rmsse_loss, rmse_loss

class Trainer:
    def __init__(self, config) -> None:
        self.config = config

    def log(self):
        print("saving results")
        self.rmse = rmse_loss(None)
        self.rmsse = rmsse_loss(None)

        val_rmse = self.rmse(self.modelmodule.val_preds, self.modelmodule.val_gt).item()
        val_rmsse = self.rmsse(self.modelmodule.val_preds, self.modelmodule.val_gt, self.modelmodule.val_historical).item()
        test_rmse = self.rmse(self.modelmodule.test_preds, self.modelmodule.test_gt).item()
        test_rmsse = self.rmsse(self.modelmodule.test_preds, self.modelmodule.test_gt, self.modelmodule.test_historical).item()


        save_dir = "{}/{}/version_0".format(self.config["logger"]["kwargs"]["save_dir"], self.modelmodule.model_name)
        os.makedirs(save_dir, exist_ok=True)

        val_path = f"{save_dir}/metrics.csv"
        test_path = f"{save_dir}/test_metrics.csv"
        model_name = self.modelmodule.model_name

        val_df = pd.DataFrame({
            "t_rmse": [999],
            "t_rmsse": [999],
            "v_rmse": [val_rmse],
            "v_rmsse": [val_rmsse],
            "epoch": [0],
            "step": [0],
        })
        val_df.to_csv(val_path, index=False)

        test_df = pd.DataFrame({
            "test_rmse": [test_rmse],
            "test_rmsse": [test_rmsse],
            "model_name": [model_name]
        })
        test_df.to_csv(test_path, index=False)

        np.save(f"{save_dir}/test", self.modelmodule.val_preds)


    def fit(self, modelmodule, datamodule):
        self.modelmodule = modelmodule
        self.datamodule = datamodule
        modelmodule.run(datamodule)
        self.log()

def create_trainer(trainer_config):
    trainer = Trainer(trainer_config)
    return trainer
