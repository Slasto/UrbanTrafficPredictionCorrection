import os
import time

import numpy as np

import matplotlib.pyplot as plt
from IPython.display import clear_output

from tensorflow.keras.callbacks import Callback


class TrainingPlot(Callback):
    """
    Callback per tracciare il plot di training/validation dopo ogni epoca
    """

    def __init__(self, Plot_only_on_final_epoch: bool = False, loss_metric: str = "mean_absolute_error", path: str = None):
        super(TrainingPlot, self).__init__()
        self.Plot_only_on_final_epoch = Plot_only_on_final_epoch
        self.loss_metric = loss_metric
        self.path = path

    def on_train_begin(self, logs={}):
        # Init di liste che salveranno i loe e metriche varie
        self.losses = []
        self.val_losses = []
        self.logs = []

    def loss_train_val_compare(self):
        N = np.arange(1, (len(self.losses) + 1))

        print("\n\n\033[1mTraining:\033[0m\n")
        plt.xlabel("Epoch")
        plt.ylabel(self.loss_metric)
        plt.xlim([1, len(self.losses)])
        plt.plot(N, self.losses, label="train")  # my_scaler_tv.apply(lambda x: x*1800)
        plt.plot(N, self.val_losses, label="valid.")

        #plt.grid(color="lightgrey", linestyle=":", linewidth=0.5)
        plt.title("loss func: Train vs Validation")
        plt.legend(loc="upper right")
        if self.path is not None : 
            plt.savefig(self.path + '.svg')
        plt.show()
        return

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get(self.loss_metric)
        val_loss = logs.get("val_" + self.loss_metric)

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(loss)
        self.val_losses.append(val_loss)

        # Plots every n-th epoch
        if epoch > 0 and not self.Plot_only_on_final_epoch:
            clear_output(wait=True)  # Clear the previous plot
            self.loss_train_val_compare()
        return

    def on_train_end(self, logs=None):
        if self.Plot_only_on_final_epoch and len(self.logs) > 0:
            self.loss_train_val_compare()
        return


class GPUThermalMonitorCallback(Callback):
    """
    Callback per limitare la temperatura della GPU nvidia durante il training
    """

    def __init__(self, max_temperature_degree_celsius: int = 80, timeout_s: int = 20):
        super(GPUThermalMonitorCallback, self).__init__()
        self.max_temperature = max_temperature_degree_celsius
        self.timeout = timeout_s

    def get_current_temperature(self) -> int:
        result = os.popen(
            "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader"
        ).read()
        return int(result.strip())

    def on_epoch_end(self, epoch, logs=None):
        while self.get_current_temperature() >= self.max_temperature:
            print(
                f"GPUThermalMonitorCallback I: \033[91mTemperature exceeded {self.max_temperature}°C. Limiting GPU usage for {self.timeout}sec...\033[0m"
            )
            time.sleep(self.timeout)
        return