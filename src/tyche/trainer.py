import datetime
import json
import logging
import os
from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple, List

import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import _Loss

from tyche.utils import param_scheduler as p_scheduler
from tyche.utils.helper import create_instance


class BaseTrainingProcedure(metaclass=ABCMeta):

    def __init__(self, model, loss, metrics, optimizer, resume, params, train_logger, data_loader):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.params = params
        self._prepare_dirs()
        self.t_logger = self._setup_logging()
        self.summary = SummaryWriter(self.tensorboard_dir)
        self.device = self.__get_device(params)
        self.model.to(self.device)
        self.start_epoch = 0
        self.n_epochs = self.params["trainer"]["epochs"]
        self.batch_size = self.params["data_loader"]["args"]["batch_size"]
        self.bm_metric = self.params["trainer"]["args"]['bm_metric']

        self.data_loader = data_loader
        self.n_train_batches = len(data_loader.train)
        self.n_val_batches = len(data_loader.validate)

        self.global_step = 0
        self.best_model = {'train_loss': float("inf"),
                           'val_loss': float("inf"),
                           'train_metric': float("inf"),
                           'val_metric': float("inf")}

        self.train_logger = train_logger
        if resume:
            self._resume_check_point(resume)

    @abstractmethod
    def _train_step(self):
        raise NotImplementedError

    def train(self):
        e_bar = tqdm.tqdm(
            desc="Epoch: ",
            total=self.n_epochs,
            unit="epoch",
            initial=self.start_epoch,
            postfix="train loss: nan, validation loss: nan")
        for epoch in range(self.start_epoch, self.n_epochs):
            train_log = self._train_epoch(epoch)
            validate_log = self._validate_epoch(epoch)

            self._check_and_save_best_model(train_log, validate_log)
            self.__update_p_bar(e_bar, train_log, validate_log)
            self._save_check_point(epoch)
        e_bar.close()
        self.best_model['name'] = self.params["name"]
        return self.best_model

    def _train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        p_bar = tqdm.tqdm(
            desc="Training batch: ", total=self.n_train_batches, unit="batch")

        epoch_stat = self._new_stat()
        for batch_idx, input in enumerate(self.data_loader.train):
            batch_stat = self._train_step(input, batch_idx, epoch, p_bar)
            for k, v in batch_stat.items():
                epoch_stat[k] += v
        p_bar.close()

        self._normalize_stats(self.n_train_batches, epoch_stat)
        self._log_epoch("train/epoch/", epoch_stat)

        return epoch_stat

    def _normalize_stats(self, n_batches, statistics):
        for k in statistics.keys():
            statistics[k] /= n_batches
        return statistics

    def _log_epoch(self, log_label, statistics):
        for k, v in statistics.items():
            self.summary.add_scalar(log_label + k, v, self.global_step)

    def __del__(self):
        self.summary.close()

    def __get_device(self, params):
        gpus = params.get("gpus", [])
        if len(gpus) > 0:
            if not torch.cuda.is_available():
                self.logger.warning("No GPU's available. Using CPU.")
                device = torch.device("cpu")
            else:
                device = torch.device("cuda:" + str(gpus[0]))
        else:
            device = torch.device("cpu")
        return device

    def _prepare_dirs(self) -> None:
        trainer_par = self.params["trainer"]
        start_time = datetime.datetime.now().strftime("%d%m_%H%M%S")
        self.checkpoint_dir = os.path.join(trainer_par["save_dir"],
                                           self.params["name"], start_time)
        self.logging_dir = os.path.join(trainer_par["logging"]["logging_dir"],
                                        self.params["name"], start_time)
        self.tensorboard_dir = os.path.join(trainer_par["logging"]["tensorboard_dir"],
                                            self.params["name"], start_time)

        os.makedirs(self.logging_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)

    def __save_model(self, file_name, **kwargs) -> None:
        model_type = type(self.model).__name__
        state = {
            "model_type": model_type,
            "epoch": kwargs.get("epoch"),
            "model_state": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "params": self.params
        }
        torch.save(state, file_name)

    def _save_model_parameters(self, file_name):
        """
        Args:
            model_directory:
        """
        with open(file_name, "w") as f:
            json.dump(self.params, f, indent=4)

    def _save_check_point(self, epoch: int) -> None:
        """

        :param epoch:
        :returns:
        :rtype:^^

        """

        file_name = os.path.join(self.checkpoint_dir,
                                 "checkpoint-epoch{}.pth".format(epoch))
        self.t_logger.info("Saving checkpoint: {} ...".format(file_name))
        self.__save_model(file_name, epoch=epoch)

    def _save_best_model(self) -> None:
        file_name = os.path.join(self.checkpoint_dir,
                                 "best_model.pth")
        self.t_logger.info("Saving best model ...")
        self.__save_model(file_name)

    def _resume_check_point(self, path: str) -> None:
        """

        :param path:
        :returns:
        :rtype:

        """
        self.logger.info("Loading checkpoint: {} ...".format(path))
        state = torch.load(path)
        self.params = state["params"]
        self.start_epoch = state["epoch"] + 1
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.logger.info("Finished loading checkpoint: {} ...".format(path))

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("train_logger")
        logger.propagate = False
        logger.setLevel(logging.INFO)

        file_name = os.path.join(self.logging_dir, "train.log")
        fh = logging.FileHandler(file_name)
        formatter = logging.Formatter(
            self.params["trainer"]["logging"]["formatters"]["simple"])
        fh.setLevel(logging.INFO)

        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

    def _log_train_step(self, epoch: int, batch_idx: int, logs: Dict, batch_size: int) -> None:
        data_len = len(self.data_loader.train.dataset)
        l = self.__build_raw_log_str("Train epoch", batch_idx, epoch, logs, data_len, batch_size)
        self.t_logger.info(l)
        for k, v in logs.items():
            self.summary.add_scalar('train/batch/' + k, v, self.global_step)

    def _log_validation_step(self, epoch: int, batch_idx: int, logs: Dict, batch_size: int) -> None:
        data_len = len(self.data_loader.validate.dataset)
        l = self.__build_raw_log_str("Validation epoch", batch_idx, epoch, logs, data_len, batch_size)
        self.t_logger.info(l)
        # for k, v in logs.items():
        #     self.summary.add_scalar('validate/batch/' + k, v, self.global_step)

    def __build_raw_log_str(self, prefix: str, batch_idx: int, epoch: int, logs: Dict, data_len: int, batch_size: int):
        l = prefix + ": {} [{}/{} ({:.0%})]".format(
            epoch,
            batch_idx * batch_size,
            data_len,
            100.0 * batch_idx / data_len)
        for k, v in logs.items():
            l += " {}: {:.6f}".format(k, v)
        return l

    def _check_and_save_best_model(self, train_log: Dict, validate_log: Dict) -> None:
        if validate_log[self.bm_metric] < self.best_model['val_metric']:
            self._save_best_model()
            self.__update_best_model_flag(train_log, validate_log)

    def __update_p_bar(self, e_bar, train_log: Dict, validate_log: Dict) -> None:
        e_bar.update()
        e_bar.set_postfix_str(
            "train loss: {:6.4f}, validation loss: {:5.4f}".format(
                train_log["loss"], validate_log["loss"]))

    def __update_best_model_flag(self, train_log: Dict, validate_log: Dict) -> None:
        self.best_model['train_loss'] = train_log['loss']
        self.best_model['val_loss'] = validate_log['loss']
        self.best_model['train_metric'] = train_log[self.bm_metric]
        self.best_model['val_metric'] = validate_log[self.bm_metric]
