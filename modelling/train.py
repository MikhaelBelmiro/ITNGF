import torch
torch.set_float32_matmul_precision("medium")

import os
import shutil
import argparse
import importlib

import pandas as pd

from utils import read_yaml

CONFIG_DIR = "../configs/modelling"
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", "-mn", required=True)
parser.add_argument("--data_name", "-dn", required=True)
parser.add_argument("--pred_period", "-p", required=True)
parser.add_argument("--continue_training", action="store_true")
parser.add_argument("--model_checkpoint_path", "-ckpt", default="")
args = parser.parse_args()

if __name__ == "__main__":
    import random
    import numpy as np

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    model_name = args.model_name
    data_name = args.data_name + "_" + args.pred_period

    if args.continue_training and args.model_checkpoint_path == "":
        raise ValueError(
            "to continue training, please provide model_checkpoint_path argument"
        )
    if args.model_checkpoint_path != "" and not(args.continue_training):
        raise ValueError(
            "to continue training, please use --continue_training flag"
        )

    data_config = read_yaml(f"{CONFIG_DIR}/data/{args.data_name}/{model_name}.yaml")
    data_config["data_prep"]["pred_period"] = int(args.pred_period)
    datamodule_module = importlib.import_module(f"models.{model_name}.dataset")
    datamodule_class = getattr(datamodule_module, "LitDataModule")
    datamodule_instance = datamodule_class(data_config)

    model_config = read_yaml(f"{CONFIG_DIR}/model/{args.data_name}/{model_name}.yaml")
    model_module = importlib.import_module(f"models.{model_name}.model")
    model_class = getattr(model_module, "LitModelModule")
    model_instance = model_class(datamodule_instance.data_related_config, model_config)

    trainer_config = read_yaml(f"{CONFIG_DIR}/trainer/{args.data_name}/{model_name}.yaml")
    trainer_config["callbacks"]["model_checkpoint"]["kwargs"]["dirpath"] = (
        trainer_config["callbacks"]["model_checkpoint"]["kwargs"]["dirpath"]
        + f"/{data_name}"
    )
    if os.path.exists(
        trainer_config["callbacks"]["model_checkpoint"]["kwargs"]["dirpath"]
    ) and not (args.continue_training):
        shutil.rmtree(
            trainer_config["callbacks"]["model_checkpoint"]["kwargs"]["dirpath"]
        )
    trainer_config["logger"]["kwargs"]["save_dir"] = (
        trainer_config["logger"]["kwargs"]["save_dir"] + f"/{data_name}"
    )

    trainer_module = importlib.import_module(f"models.{model_name}.trainer")
    trainer = getattr(trainer_module, "create_trainer")(trainer_config)
    if args.continue_training:
        if os.path.isdir(args.model_checkpoint_path) or ".ckpt" not in args.model_checkpoint_path:
            raise ValueError(f"{args.model_checkpoint_path} is not a .ckpt file")
        if not(os.path.exists(args.model_checkpoint_path)) or not os.path.isfile(args.model_checkpoint_path) :
            raise ValueError(f"{args.model_checkpoint_path} does not exist")
        trainer.fit(model_instance, datamodule=datamodule_instance, ckpt_path=args.model_checkpoint_path)
    else:
        trainer.fit(model_instance, datamodule=datamodule_instance)


    if model_name != "statsforecast":
        test_metrics = []
        test_metrics_path = "{}/{}/version_0/test_metrics.csv".format(trainer_config["logger"]["kwargs"]["save_dir"], trainer_config["logger"]["kwargs"]["name"])
        test_pred_path = trainer_config["logger"]["kwargs"]["save_dir"] + "/" + trainer_config["logger"]["kwargs"]["name"] + "/version_0/test"
        for item in os.listdir(trainer_config["callbacks"]["model_checkpoint"]["kwargs"]["dirpath"]):
            ckpt_path = "{}/{}".format(trainer_config["callbacks"]["model_checkpoint"]["kwargs"]["dirpath"], item)
            test_metric = trainer.test(ckpt_path=ckpt_path, dataloaders=datamodule_instance.test_dataloader())
            test_metric[0]["model_name"] = item
            test_metrics.extend(test_metric)
        test_metrics_df = pd.DataFrame.from_records(test_metrics)
        if os.path.exists(test_metrics_path):
            test_metrics_df_old = pd.read_csv(test_metrics_path) if args.continue_training else pd.DataFrame()
            test_metrics_df = pd.concat((test_metrics_df_old, test_metrics_df), axis=0)
            test_metrics_df = test_metrics_df.drop_duplicates(subset=["model_name"], keep="last")
        test_metrics_df.to_csv(test_metrics_path, index=False)
        
        test_predict = trainer.predict(model=model_instance, datamodule=datamodule_instance, return_predictions=True, ckpt_path="best")[0]
        np.save(test_pred_path, test_predict.detach().numpy())
