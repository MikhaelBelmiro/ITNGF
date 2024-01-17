import torch

import numpy as np

from tqdm import tqdm
from utils import create_class_instance

class LitModelModule:
    def __init__(
        self,
        data_related_config,
        model_related_config,
    ) -> None:
        print("creating modelmodule")
        self.data_related_config = data_related_config
        self.model_related_config = model_related_config

    def run(self, data_module):
        self.model_name = self.model_related_config["model"]["kwargs"]["alias"]
        print(f"running trainer for model {self.model_name}")
        
        val_preds = []
        test_preds = []
        with tqdm(total=data_module.train_data.shape[1]) as pbar:
            for i in range(data_module.train_data.shape[1]):
                single_ts = data_module.train_data[:, i]
                model = create_class_instance(
                    self.model_related_config["model"]["classpath"],
                    self.model_related_config["model"]["kwargs"],
                )
                model.fit(single_ts)
                single_pred = model.predict(self.data_related_config["val_size"] + self.data_related_config["test_size"])["mean"].reshape(-1, 1)
                val_preds.append(single_pred[:self.data_related_config["val_size"], :])
                test_preds.append(single_pred[self.data_related_config["val_size"]:, :])
                pbar.update(1)

        self.val_historical = torch.from_numpy(data_module.train_data).float()
        self.val_gt = torch.from_numpy(data_module.val_data).float()
        self.val_preds = torch.from_numpy(np.concatenate(val_preds, axis=1))

        self.test_historical = torch.concat((self.val_historical, self.val_gt), dim=0)[-self.data_related_config["train_size"]:].float()
        self.test_gt = torch.from_numpy(data_module.test_data).float()
        self.test_preds = torch.from_numpy(np.concatenate(test_preds, axis=1)).float()

        self.val_historical = self.val_historical.unsqueeze(0)
        self.val_gt = self.val_gt.unsqueeze(0)
        self.val_preds = self.val_preds.unsqueeze(0)

        self.test_historical = self.test_historical.unsqueeze(0)
        self.test_gt = self.test_gt.unsqueeze(0)
        self.test_preds = self.test_preds.unsqueeze(0)