from utils import create_class_instance
from lightning import Trainer


def create_logger(logger_config):
    logger_instance = create_class_instance(
        logger_config["classpath"], logger_config["kwargs"]
    )
    return logger_instance


def create_callbacks(callback_config):
    callbacks = []
    for callback_name in callback_config:
        config = callback_config[callback_name]
        callback_instance = create_class_instance(config["classpath"], config["kwargs"])
        callbacks.append(callback_instance)
    return callbacks


def create_trainer(trainer_config):
    trainer = Trainer(
        accelerator=trainer_config["accelerator"],
        devices=trainer_config["devices"],
        max_epochs=trainer_config["max_epochs"],
        callbacks=create_callbacks(trainer_config["callbacks"]),
        logger=create_logger(trainer_config["logger"]),
        log_every_n_steps=2,
    )
    return trainer
