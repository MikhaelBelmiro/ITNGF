import torch
torch.set_float32_matmul_precision("medium")

import argparse
import importlib

from tqdm import tqdm
from utils import read_yaml
from loss import masked_mae_np, masked_mape_np, masked_rmse_np
from pprint import pprint

CONFIG_DIR = "../configs/modelling"
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", "-mn", required=True)
parser.add_argument("--data_name", "-dn", required=True)
parser.add_argument("--pred_period", "-p", required=True)
parser.add_argument("--model_checkpoint_path", "-ckpt", default="")
parser.add_argument("--device", "-device", default="cuda:0")
args = parser.parse_args()

if __name__ == "__main__":
    import random
    import numpy as np

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    model_name = args.model_name
    data_name = args.data_name + "_" + args.pred_period
    device = torch.device(args.device)

    data_config = read_yaml(f"{CONFIG_DIR}/data/{args.data_name}.yaml")
    data_config["data_prep"]["pred_period"] = int(args.pred_period)
    datamodule_module = importlib.import_module(f"models.{model_name}.dataset")
    datamodule_class = getattr(datamodule_module, "LitDataModule")
    datamodule_instance = datamodule_class(data_config)


    model_config = read_yaml(f"{CONFIG_DIR}/model/{args.data_name}/{model_name}.yaml")
    model_module = importlib.import_module(f"models.{model_name}.model")
    model_class = getattr(model_module, "LitModelModule")
    model_instance = model_class.load_from_checkpoint(args.model_checkpoint_path, data_related_config=datamodule_instance.data_related_config, model_hyperparam_config=model_config, map_location=device)

    pred = []
    gt = []
    test_loader = datamodule_instance.test_dataloader()
    with tqdm(total=len(test_loader)) as pbar:
        pbar.set_description(f"generating predictions on data {args.data_name} from {args.model_checkpoint_path}")
        for batch in test_loader:
            batch = datamodule_instance.transfer_batch_to_device(batch, device, 0)

            gt.append(batch["gt_ts"].detach().cpu())
            pred.append(model_instance(batch).detach().cpu())
            pbar.update(1)

    pred = torch.concat(pred, dim=0).squeeze(-1)
    gt = torch.concat(gt, dim=0).squeeze(-1)

    pred_period = gt.shape[1]
    for i in range(pred_period):
        pred[:, i, :] = datamodule_instance.data_scaler.inverse_transform(pred[:, i, :])
        gt[:, i, :] = datamodule_instance.data_scaler.inverse_transform(gt[:, i, :])

    pred_numpy = pred.numpy()
    gt_numpy = gt.numpy()

    metrics = {
        "mae": masked_mae_np,
        "mape": masked_mape_np,
        "rmse": masked_rmse_np,
    }

    out_metrics = {}
    eval_periods = [3, 6, 12]
    for metric_name, metric_fn in metrics.items():
        for i in eval_periods:
            out_metrics[f"{metric_name}_length_{i}"] = metric_fn(pred_numpy[:, i-1, :], gt_numpy[:, i-1, :], 0)
    print(out_metrics)