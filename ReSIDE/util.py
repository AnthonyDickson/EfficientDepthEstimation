# original script: https://github.com/fangchangma/sparse-to-dense/blob/master/utils.lua
import datetime
import warnings
from abc import ABC
from typing import Callable, Union

import torch
import math
import numpy as np


class MetricsTracker:
    """Tracks running averages for several common metrics for depth estimation."""

    def __init__(self):
        self.mae = AverageMeter()
        self.mse = AverageMeter()
        self.rmse = 0
        self.abs_rel = AverageMeter()
        self.log10 = AverageMeter()
        self.delta1 = AverageMeter()
        self.delta2 = AverageMeter()
        self.delta3 = AverageMeter()

    def __getitem__(self, item):
        self.__getattribute__(item.lower())

    def to_dict(self):
        result = dict()

        for key, metric in self.__dict__.items():
            if isinstance(metric, AverageMeter):
                result[key] = metric.value
            else:
                result[key] = metric

        return result

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        Update the running averages.

        :param outputs: The neural network predictions.
        :param labels: The ground truth depth maps
        """
        outputs_, labels_ = outputs.detach(), labels.detach()

        nan_mask = torch.isnan(labels_)
        invalid_mask = ~(labels_ > 0)
        num_valid = (~nan_mask).sum().item()

        batch_size = labels_.shape[0]

        residuals = outputs_ - labels_
        abs_residuals = torch.abs(residuals)

        mae = batch_size * torch.sum(abs_residuals).item() / num_valid
        mse = batch_size * torch.sum(torch.pow(residuals, 2)).item() / num_valid

        abs_rel = abs_residuals / labels_
        abs_rel[nan_mask] = 0
        abs_rel[invalid_mask] = 0
        abs_rel = batch_size * torch.sum(abs_rel).item() / num_valid

        log10 = torch.abs(torch.log10(outputs_) - torch.log10(labels_))
        log10[nan_mask] = 0
        log10[invalid_mask] = 0
        log10 = torch.sum(log10).item() / num_valid

        max_ratio = torch.max(outputs_ / labels_, labels_ / outputs_)
        delta1 = self.threshold_accuracy(max_ratio, math.pow(1.25, 1), num_valid) * batch_size
        delta2 = self.threshold_accuracy(max_ratio, math.pow(1.25, 2), num_valid) * batch_size
        delta3 = self.threshold_accuracy(max_ratio, math.pow(1.25, 3), num_valid) * batch_size

        self.mae.update(mae, batch_size)
        self.mse.update(mse, batch_size)
        self.rmse = math.sqrt(self.mse.value)
        self.abs_rel.update(abs_rel, batch_size)
        self.log10.update(log10, batch_size)

        self.delta1.update(delta1, batch_size)
        self.delta2.update(delta2, batch_size)
        self.delta3.update(delta3, batch_size)

    def __str__(self):
        return f"ABS_REL: {self.abs_rel:.3f} - MAE: {self.mae:.3f} - " \
               f"MSE: {self.mse:.3f} - RMSE: {self.rmse:.3f} - LOG10: {self.log10:.3f} - " \
               f"DELTA1: {self.delta1:.3f} - DELTA2: {self.delta2:.3f} - DELTA3: {self.delta3:.3f}        "

    @staticmethod
    def threshold_accuracy(max_ratio: torch.Tensor, threshold, N):
        return torch.sum((max_ratio <= threshold).float()).item() / N


class MetricsMeter(ABC):
    @property
    def value(self):
        raise NotImplementedError

    def update(self, value):
        raise NotImplementedError

    def __str__(self):
        return str(self.value)

    def __format__(self, format_spec):
        return f"{self.value:{format_spec}}"


class AverageMeter(MetricsMeter):
    """Implements a basic running average."""

    def __init__(self):
        self._sum = 0
        self._count = 0

    @property
    def value(self):
        """The average size."""
        try:
            return self._sum / self._count
        except ZeroDivisionError:
            return float("nan")

    def update(self, value, num_elements=1):
        """
        Add a value to the average.

        :param value: The value to add.
        :param num_elements: The number of elements that `value` was calculated over.
        """
        if not math.isnan(value) and not math.isinf(value):
            self._sum += value
            self._count += num_elements


class LambdaMeter(MetricsMeter):
    """Tracks the value of a given metric by a given function (e.g. min, max)."""

    def __init__(self, lambda_fn: Callable[[Union[float, int], Union[float, int]], Union[float, int]]):
        """
        :param lambda_fn: A function that takes two numbers, the current tracked value and the value to (possibly)
            update it with, and returns the number to set the tracked value to. For example, if you want to track the
            minimum value of a metric, such as a loss function, you could use the builtin `min` function. Similarly you
            could use `max` for tracking the maximum value of a metric.
        """
        self._value = float("nan")
        self.lambda_fn = lambda_fn

    @property
    def value(self):
        """The tracked value (defaults to NaN)."""
        return self._value

    def update(self, value):
        """
        Update the tracked value.

        :param value: The value to (possibly) set the tracked value to. Whether the tracked value is actually set to
        this value depends on the lambda function given to the constructor.
        """
        if not math.isnan(value) and not math.isinf(value):
            self._value = value if math.isnan(self._value) else self.lambda_fn(self._value, value)
        else:
            warnings.warn("Invalid value encountered (NaN or +/- infinity), ignoring value.")


class BestMetricsTracker:
    """Tracks the best values for the benchmark metrics."""

    def __init__(self):
        self.mae = LambdaMeter(min)
        self.mse = LambdaMeter(min)
        self.rmse = LambdaMeter(min)
        self.abs_rel = LambdaMeter(min)
        self.log10 = LambdaMeter(min)
        self.delta1 = LambdaMeter(max)
        self.delta2 = LambdaMeter(max)
        self.delta3 = LambdaMeter(max)

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def update(self, metrics: MetricsTracker):
        self.mae.update(metrics.mae.value)
        self.mse.update(metrics.mse.value)
        self.rmse.update(metrics.rmse)
        self.abs_rel.update(metrics.abs_rel.value)
        self.log10.update(metrics.log10.value)
        self.delta1.update(metrics.delta1.value)
        self.delta2.update(metrics.delta2.value)
        self.delta3.update(metrics.delta3.value)

    def to_dict(self):
        return {key: metric.value for key, metric in self.__dict__.items()}


class Timer:
    """Utility for timing operations. Can be used as a context manager."""

    def __init__(self):
        self._start_time = datetime.datetime.fromtimestamp(0)
        self._stop_time = None
        self._is_running = False

    @property
    def start_time(self):
        """
        :return: The `datetime` object when the timer was started. See `start()`.
        """
        return self._start_time

    @property
    def stop_time(self):
        """
        :return: The `datetime` object when the timer was stopped. See `stop()`.
        """
        return self._stop_time

    @property
    def elapsed(self):
        """
        :return: The `timedelta` object indicating how much time elapsed. If the timer has been stopped, will return
          the elapsed time between `start_time` and `end_time`, otherwise will return the elapsed time between
          `start_time` and now.
        """
        if self._stop_time is not None:
            return self._stop_time - self._start_time
        else:
            return datetime.datetime.now() - self._start_time

    def start(self):
        """Start the timer."""
        self._start_time = datetime.datetime.now()
        self._stop_time = None
        self._is_running = True

    def stop(self):
        """Stop the timer."""
        self._stop_time = datetime.datetime.now()
        self._is_running = False

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
