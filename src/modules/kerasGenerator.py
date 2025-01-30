import os
import copy
import numpy as np
import pandas as pd

from tqdm import tqdm
from IPython.display import display

import tensorflow as tf
from modules.Plotter import GeneralPlotter


class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(
        self,
        df: pd.DataFrame,
        X_col: list[str],
        y_col: list[str],
        batch_size: int,
        input_width: int = 10,
        label_width: int = 1,
        y_offset: int = 0,
        **kwargs,
    ):
        super(CustomDataGen, self).__init__(**kwargs)
        # Copia del dataFrame di input
        self.df = df.copy().reset_index(drop=True)

        # Nomi delle colonne che rappresentano rispettivamente i dati di input (features) e i dati di output (labels)
        self.X_col = X_col
        self.y_col = y_col

        # Numero di campioni che saranno inclusi in ogni batch
        self.batch_size = batch_size

        # Dimensione esatta della window e dei target
        self.input_width = input_width
        self.label_width = label_width

        # offset tra la x windows e la y windows
        self.y_offset = y_offset

        # Numero totale di campioni nel DataFrame
        self.n = len(self.df)

    def on_epoch_end(self):
        pass

    def __getitem__(self, index: int):
        X = []
        y = []

        batch_start = index * self.batch_size
        for i in range(self.batch_size):
            start_X = batch_start + i  # pt di partenza è incluso
            end_X = start_X + self.input_width - 1  # pt di arrivo è incluso

            start_y = end_X + self.y_offset + 1  # pt di partenza è incluso
            end_y = start_y + self.label_width - 1  # pt di arrivo è incluso

            X.append(self.df.loc[start_X:end_X, self.X_col].values)
            y.append(self.df.loc[start_y:end_y, self.y_col].values)

        return np.array(X), np.array(y)

    # Determina il numero di batch per epoca
    def __len__(self) -> int:
        return (self.n // self.batch_size) - 1  # HOTFIX -1

    # EXTRA  - - - - - - - -

    def compute_metrics(self, y_pred: np.array, scale_factor=1800) -> dict:
        """
        Compute performance metrics for predictions.

        Parameters:
        y_pred (np.array): Predicted values array.
        scale_factor (int): Scale factor applied to predictions and true values. Default is 1800.

        Returns:
        dict: A dictionary with mean squared error and mean absolute error, rounded to 4 decimal places.
        """
        y_pred = np.asarray(y_pred).ravel() * scale_factor
        y_true = np.asarray(self.get_y_in_epoch).ravel()[: len(y_pred)] * scale_factor

        y_true = self.df[self.y_col][
            self.input_width + self.y_offset : self.input_width
            + self.y_offset
            + len(y_pred)
        ]
        y_true = np.asarray(y_true).ravel() * scale_factor

        mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
        mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)

        return {
            "mean_squared_error": round(np.float32(mse), 4),
            "mean_absolute_error": round(np.float32(mae), 4),
        }

    def plot_df(self, title: str = "Hourly_Traffic"):
        src = self.df.copy()
        src["date"] = pd.to_datetime(src.date)
        GeneralPlotter.plot_traffic(src, title)

    def free_run(
        self, model: tf.keras.models.Sequential, plot_x_frame: int = -1
    ) -> np.array:
        """
        Perform a free run of predictions replacing outputs in the dataset.

        Parameters:
        model (tf.keras.models.Sequential): The trained model to use for predictions.
        plot_x_frame (int): The number of frames for which predictions are displayed. Default is -1 (no display).

        Returns:
        np.array: Array of model predictions.
        """
        prediction = []
        data = self.src_copy

        for i in tqdm(
            range(
                0, len(data) - self.input_width - self.label_width - self.y_offset + 1
            ),
            desc="Free-Run",
        ):
            # Output da sostituire post predizione
            start_y = i + self.input_width + self.y_offset
            end_y = start_y + self.label_width - 1

            # x_window corrente
            x = data[self.X_col][i : i + self.input_width]

            # predizione
            predict = model(np.array(x).reshape(1, self.input_width, len(self.X_col)))

            # sostituzione sul dataFrame della predizione
            data.loc[start_y:end_y, self.y_col] = predict.numpy()

            prediction.append(predict)

            if i < plot_x_frame:
                display(x)
                display(predict)

            del x, predict

        return np.array(prediction)

    def predict_without_state_reset(
        self, on_model: tf.keras.models.Sequential, plot_x_frame: int = -1
    ) -> np.array:
        """
        Predict outcomes using the provided model without resetting state.

        Parameters:
        on_model (tf.keras.models.Sequential): The model used for making predictions.
        plot_x_frame (int): The frame index for which predictions are plotted. Default is -1 (no plotting).

        Returns:
        np.array: Array of predictions.
        """
        prediction = []
        on_model = copy.copy(on_model)
        data = self.src_copy
        for i in tqdm(
            range(
                0, len(data) - self.input_width - self.label_width - self.y_offset + 1
            ),
            desc="Predicting"
        ):
            # x_window corrente
            x = data[self.X_col][i : i + self.input_width]

            # predizione
            predict = on_model(
                np.array(x).reshape(1, self.input_width, len(self.X_col))
            )

            prediction.append(predict)

            del x, predict
        return np.array(prediction)

    # x property
    @property
    def shape_X(self):
        return (self.input_width, len(self.X_col))

    @property
    def get_x_in_epoch(self) -> pd.DataFrame:
        return self.df[self.X_col][: len(self) * self.batch_size]

    # y property
    @property
    def shape_y(self):
        return self.label_width

    @property
    def get_y_in_epoch(self) -> pd.DataFrame:
        return self.df[self.y_col][
            self.input_width + self.y_offset : len(self) * self.batch_size
            + (self.input_width + self.y_offset)
        ]

    def get_y_until(self, last_index: int):
        return self.df[self.y_col][
            self.input_width + self.y_offset : self.input_width
            + self.y_offset
            + last_index
        ]

    # copy
    @property
    def src_copy(self) -> pd.DataFrame:
        return self.df.copy()

    def change_df(self, value: pd.DataFrame):
        self.df = value.reset_index(drop=True)
        self.n = len(value)

    def copy(self):
        return copy.deepcopy(self)


# --- --- --   Altro   -- --- ---


def plot_model_info(model, folder: str = "temp/") -> None:
    """
    Generate a visual representation of a Keras model and save as an image.

    Parameters:
    model (tf.keras.Model): The Keras model to visualize.
    folder (str): The folder where the model image will be saved. Default is "temp/".
    """
    model.summary()
    copy = tf.keras.models.clone_model(model)
    copy.build()

    if folder != "":
        os.makedirs(folder, exist_ok=True)

    display(
        tf.keras.utils.plot_model(
            copy,
            show_shapes=True,
            show_layer_names=False,
            rankdir="LR",
            dpi=96,
            show_layer_activations=True,
            to_file=folder + "model.png",
        )
    )
    return