import os
import copy
import pytorch_lightning as pl
import datetime
import wandb


from easydict import EasyDict

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from model.model_interface import ModelInterface
from dataset.data_interface import DataInterface
from pytorch_lightning.strategies import (
    DDPStrategy,
    DeepSpeedStrategy,
    Strategy,
)

from saprot.utils.tasks import (
    TASK2MODEL,
    TASK2DATASET,
    TASK_TYPE,
    ALL_TASKS,
    ALL_TASKS_HINT,
)


class ModelDispatcher:
    def __init__(self, task: ALL_TASKS_HINT, config: EasyDict):
        self.task: ALL_TASKS_HINT = task
        if not task in ALL_TASKS:
            raise ValueError(f"Task {task} not supported")

        self.model_config = copy.deepcopy(config)

        self.model_config.pop("model_py_path")

        if "kwargs" in self.model_config.keys():
            kwargs = self.model_config.pop("kwargs")
        else:
            kwargs = {}

        self.model_config.update(kwargs)

    def dispatch(self):
        if self.task == "classification":
            from model.saprot.saprot_classification_model import (
                SaprotClassificationModel,
            )

            return SaprotClassificationModel(**self.model_config)

        if self.task == "token_classification":
            from model.saprot.saprot_token_classification_model import (
                SaprotTokenClassificationModel,
            )

            return SaprotTokenClassificationModel(**self.model_config)

        if self.task == "regression":
            if "num_labels" in self.model_config:
                del self.model_config["num_labels"]
            from model.saprot.saprot_regression_model import (
                SaprotRegressionModel,
            )

            return SaprotRegressionModel(**self.model_config)

        if self.task == "pair_classification":
            from model.saprot.saprot_pair_classification_model import (
                SaprotPairClassificationModel,
            )

            return SaprotPairClassificationModel(**self.model_config)

        if self.task == "pair_regression":
            from model.saprot.saprot_pair_regression_model import (
                SaprotPairRegressionModel,
            )

            return SaprotPairRegressionModel(**self.model_config)

        def __call__(self):
            return self.dispatch()


################################################################################
################################ load model ####################################
################################################################################
def my_load_model(config):
    model_config = copy.deepcopy(config)
    model_type = model_config.pop("model_py_path")

    if "kwargs" in model_config.keys():
        kwargs = model_config.pop("kwargs")
    else:
        kwargs = {}

    model_config.update(kwargs)

    if model_type == "saprot/saprot_classification_model":
        from model.saprot.saprot_classification_model import (
            SaprotClassificationModel,
        )

        return SaprotClassificationModel(**model_config)

    if model_type == "saprot/saprot_token_classification_model":
        from model.saprot.saprot_token_classification_model import (
            SaprotTokenClassificationModel,
        )

        return SaprotTokenClassificationModel(**model_config)

    if model_type == "saprot/saprot_regression_model":
        if "num_labels" in model_config:
            del model_config["num_labels"]
        from model.saprot.saprot_regression_model import SaprotRegressionModel

        return SaprotRegressionModel(**model_config)

    if model_type == "saprot/saprot_pair_classification_model":
        from model.saprot.saprot_pair_classification_model import (
            SaprotPairClassificationModel,
        )

        return SaprotPairClassificationModel(**model_config)

    if model_type == "saprot/saprot_pair_regression_model":
        from model.saprot.saprot_pair_regression_model import (
            SaprotPairRegressionModel,
        )

        return SaprotPairRegressionModel(**model_config)


class DatasetDispatcher:
    def __init__(self, task: ALL_TASKS_HINT, config: EasyDict):
        self.task: ALL_TASKS_HINT = task
        self.dataset_config = copy.deepcopy(config)

        if self.task not in ALL_TASKS:
            raise ValueError(f"Task {self.task} is not supported.")

        # Handle additional keyword arguments if present
        if "kwargs" in self.dataset_config:
            kwargs = self.dataset_config.pop("kwargs")
            self.dataset_config.update(kwargs)

    def dispatch(self):
        if self.task == "classification":
            from dataset.saprot.saprot_classification_dataset import (
                SaprotClassificationDataset,
            )

            return SaprotClassificationDataset(**self.dataset_config)

        elif self.task == "token_classification":
            if "plddt_threshold" in self.dataset_config:
                del self.dataset_config["plddt_threshold"]
            from dataset.saprot.saprot_token_classification_dataset import (
                SaprotTokenClassificationDataset,
            )

            return SaprotTokenClassificationDataset(**self.dataset_config)

        elif self.task == "regression":
            from dataset.saprot.saprot_regression_dataset import (
                SaprotRegressionDataset,
            )

            return SaprotRegressionDataset(**self.dataset_config)

        elif self.task == "pair_classification":
            from dataset.saprot.saprot_pair_classification_dataset import (
                SaprotPairClassificationDataset,
            )

            return SaprotPairClassificationDataset(**self.dataset_config)

        elif self.task == "pair_regression":
            from dataset.saprot.saprot_pair_regression_dataset import (
                SaprotPairRegressionDataset,
            )

            return SaprotPairRegressionDataset(**self.dataset_config)

    def __call__(self):
        return self.dispatch()


################################################################################
################################ load dataset ##################################
################################################################################
def my_load_dataset(config):
    dataset_config = copy.deepcopy(config)
    dataset_type = dataset_config.pop("dataset_py_path")
    kwargs = dataset_config.pop("kwargs")
    dataset_config.update(kwargs)

    if dataset_type == "saprot/saprot_classification_dataset":
        from dataset.saprot.saprot_classification_dataset import (
            SaprotClassificationDataset,
        )

        return SaprotClassificationDataset(**dataset_config)

    if dataset_type == "saprot/saprot_token_classification_dataset":
        if "plddt_threshold" in dataset_config:
            del dataset_config["plddt_threshold"]
        from dataset.saprot.saprot_token_classification_dataset import (
            SaprotTokenClassificationDataset,
        )

        return SaprotTokenClassificationDataset(**dataset_config)

    if dataset_type == "saprot/saprot_regression_dataset":
        from dataset.saprot.saprot_regression_dataset import (
            SaprotRegressionDataset,
        )

        return SaprotRegressionDataset(**dataset_config)

    if dataset_type == "saprot/saprot_pair_classification_dataset":
        from dataset.saprot.saprot_pair_classification_dataset import (
            SaprotPairClassificationDataset,
        )

        return SaprotPairClassificationDataset(**dataset_config)

    if dataset_type == "saprot/saprot_pair_regression_dataset":
        from dataset.saprot.saprot_pair_regression_dataset import (
            SaprotPairRegressionDataset,
        )

        return SaprotPairRegressionDataset(**dataset_config)


def load_wandb(config):
    # initialize wandb
    wandb_config = config.setting.wandb_config
    wandb_logger = WandbLogger(
        project=wandb_config.project,
        config=config,
        name=wandb_config.name,
        settings=wandb.Settings(),
    )

    return wandb_logger


def load_model(config):
    # initialize model
    model_config = copy.deepcopy(config)

    if "kwargs" in model_config.keys():
        kwargs = model_config.pop("kwargs")
    else:
        kwargs = {}

    model_config.update(kwargs)
    return ModelInterface.init_model(**model_config)


def load_dataset(config):
    # initialize dataset
    dataset_config = copy.deepcopy(config)

    if "kwargs" in dataset_config.keys():
        kwargs = dataset_config.pop("kwargs")
    else:
        kwargs = {}

    dataset_config.update(kwargs)
    return DataInterface.init_dataset(**dataset_config)


# def load_plugins():
#     config = get_config()
#     # initialize plugins
#     plugins = []
#
#     if "Trainer_plugin" not in config.keys():
#         return plugins
#
#     if not config.Trainer.logger:
#         if hasattr(config.Trainer_plugin, "LearningRateMonitor"):
#             config.Trainer_plugin.pop("LearningRateMonitor", None)
#
#     if not config.Trainer.enable_checkpointing:
#         if hasattr(config.Trainer_plugin, "ModelCheckpoint"):
#             config.Trainer_plugin.pop("ModelCheckpoint", None)
#
#     for plugin, kwargs in config.Trainer_plugin.items():
#         plugins.append(eval(plugin)(**kwargs))
#
#     return plugins


# Initialize strategy
def load_strategy(config):
    config = copy.deepcopy(config)
    if "timeout" in config.keys():
        timeout = int(config.pop("timeout"))
        config["timeout"] = datetime.timedelta(seconds=timeout)

    cls = config.pop("class")
    return eval(cls)(**config)


# Initialize a pytorch lightning trainer
def load_trainer(config):
    trainer_config = copy.deepcopy(config.Trainer)

    # Initialize wandb
    if trainer_config.logger:
        trainer_config.logger = load_wandb(config)
    else:
        trainer_config.logger = False

    # Initialize strategy
    # strategy = load_strategy(trainer_config.pop('strategy'))
    # Strategy is not used in Colab
    trainer_config.pop("strategy")

    return pl.Trainer(
        **trainer_config, callbacks=[], use_distributed_sampler=False
    )
