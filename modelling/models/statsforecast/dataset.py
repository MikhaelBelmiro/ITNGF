from utils import load_numpy

class LitDataModule:
    def __init__(self, config) -> None:
        print("creating datamodule")
        self.config = config
        self.full_ts_data = load_numpy(
            self.config["data_loading"]["preprocessed_ts_data_path"]
        ).transpose()

        train_split, val_split, test_split = self.config["data_prep"]["split"]
        train_size = round(len(self.full_ts_data) * train_split)
        val_size = round((len(self.full_ts_data) * val_split))
        test_size = len(self.full_ts_data) - train_size - val_size

        train_start_index = 0
        val_start_index = train_size
        test_start_index = train_size + val_size

        self.train_data = self.full_ts_data[train_start_index:val_start_index][-self.config["data_prep"]["lookback_period"]:, :]
        self.val_data = self.full_ts_data[val_start_index:val_start_index + self.config["data_prep"]["pred_period"]]
        self.test_data = self.full_ts_data[test_start_index:test_start_index + self.config["data_prep"]["pred_period"]]

        self.data_related_config = {
            "train_size": self.train_data.shape[0],
            "val_size": self.val_data.shape[0],
            "test_size": self.test_data.shape[0],
        }