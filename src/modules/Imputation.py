from modules.kerasGenerator import CustomDataGen
from modules.DatasetWrapper import Dataset
from modules.Plotter import GeneralPlotter
from modules.TrainOnOneSite import Trainer
from IPython.display import display
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import copy
import os


class Imputation:
    def __init__(
        self,
        Gen: CustomDataGen,
        workspace: str = "temp/imputer_" + time.strftime("%Y%m%d-%H%M"),
    ):
        """
        Initializes the Imputation class with a given data generator and workspace.

        Args:
            Gen (CustomDataGen): An instance of CustomDataGen to be used for data generation.
            workspace (str): Name used for the workspace directory. Default is a timestamp-based string.

        Attributes:
            workspace (str): Directory path for storing images generated during imputation.
            Gen (CustomDataGen): The data generator instance for creating and handling dataset.
        """
        self.workspace = "img/" + workspace + "/"
        self.Gen = Gen

    def select_site_and_full_range(
        self, src: Dataset, site_no: int, export_range: range
    ):
        """
        Selects data for a specified site and range from the source dataset,
        resets the index, and assigns it to the data generator.

        Args:
            src (Dataset): The data source from which to select a site.
            site_no (int): Identifier for the site to select.
            export_range (range): The range of data indices to export.

        Side Effects:
            Updates self.Gen.df with selected site data.
            Updates self.n with the length of the selected data.
            Calls self.Gen.plot_df() to plot the data.

        Notes:
            - Assumes src has a method copy() for duplication.
            - Assumes src has a method select_site() for site selection.
            - Resets the index of the sliced data.
        """
        data = src.copy()
        data.select_site(site_no)

        dataFrame = data.site_data[export_range.start : export_range.stop].reset_index(
            drop=True
        )  # Si suppone che

        self.Gen.df = dataFrame
        self.n = len(self.Gen.df)
        self.Gen.plot_df()

    def __compute_metrics_on_interested_intervals(
        self,
        pred: list[float],
        Gen: CustomDataGen,
        to_compare: range,
        scale: int = 1800,
        folder: str = "img/temp/model_" + time.strftime("%Y%m%d-%H%M"),
    ) -> tuple[dict, dict]:
        """
            Computes the metrics of interest over specified intervals and generates plots.

            Args:
                pred (list[float]): List of predicted values.
                Gen (CustomDataGen): Data generator providing ground truth values.
                to_compare (range): The range of indices over which metrics are computed.
                scale (int, optional): Scale factor for de-normalized metric computation. Defaults to 1800.
                folder (str, optional): Directory path for saving plots. Defaults to a timestamp-based path.

            Returns:
                tuple[dict, dict]: A tuple containing two dictionaries:
                    {str: float}: Normalized mean squared error and mean absolute error metrics.
                    {str: float}: De-normalized mean squared error and mean absolute error metrics.

            Side Effects:
                Generates a violin plot for the predictions versus the actual values over the specified interval.

            Notes:
                - Utilizes TensorFlow's built-in methods for computing mean squared and absolute errors.
                - Utilizes a temporary copy of 'Gen' to avoid modifying the original dataset while plotting.
            """
        start = to_compare.start
        stop = to_compare.stop

        y_true = np.asarray(Gen.get_y_until(len(pred))).ravel()[start:stop]
        y_pred = pred.ravel()[start:stop]

        N_mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
        N_mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
        Norm_metrics = {
            "mean_squared_error": round(np.float32(N_mse), 4),
            "mean_absolute_error": round(np.float32(N_mae), 4),
        }

        D_mse = tf.keras.metrics.mean_squared_error(y_true * scale, y_pred * scale)
        D_mae = tf.keras.metrics.mean_absolute_error(y_true * scale, y_pred * scale)
        DeNorm_metrics = {
            "mean_squared_error": round(np.float32(D_mse), 4),
            "mean_absolute_error": round(np.float32(D_mae), 4),
        }

        temp_gen = Gen.copy()
        temp_gen.df = Gen.df[
            to_compare.start - Gen.input_width - Gen.y_offset : to_compare.stop
        ]
        temp_gen.n = len(temp_gen.df)
        GeneralPlotter.violin_plot_after_fit(
            y_pred, temp_gen, path=folder + "violin_plot"
        )

        return Norm_metrics, DeNorm_metrics

    def predict_and_free_run(
        self,
        on_model: tf.keras.models.Sequential,
        free_run_range: range,
        save_name: str = "",
    ) -> tuple[np.array, dict, dict]:
        """
        Executes predictions on a given model, performs a free run over a specified range,
        and generates plots and metrics for the predictions.

        Args:
            on_model (tf.keras.models.Sequential): The model on which to perform predictions.
            free_run_range (range): The range over which the model will perform the free run.
            save_name (str): An optional name for saving the results and plots. Default is an empty string.

        Returns:
            tuple[np.array, dict, dict]: A tuple containing:
                - np.array: The prediction results after the free run.
                - dict: Normalized metrics computed based on the prediction results.
                - dict: De-normalized metrics computed based on the prediction results.

        Side Effects:
            - Saves a visual plot of the model architecture to a directory.
            - Prints normalized and de-normalized metrics to output.
            - Generates and saves a violin plot based on the prediction results.

        Note:
            - Utilizes the `Generator` class' methods to prepare datasets and compute metrics.
            - Assumes existence of a directory structure to save model plots and results.
            - Assumes the `Trainer.free_run_with_plots` function is responsible for handling the free run
              and generation of initial plots.
        """
        folder = self.workspace + on_model.name + "/"
        os.makedirs(folder, exist_ok=True)
        tf.keras.utils.plot_model(
            on_model,
            show_shapes=True,
            show_layer_names=False,
            rankdir="LR",
            dpi=96,
            show_layer_activations=True,
            to_file=folder + "model.png",
        )

        predict_df = self.Gen.df[: free_run_range.start].reset_index(drop=True)
        free_run_df = self.Gen.df[
            free_run_range.start : free_run_range.stop
        ].reset_index(drop=True)

        Generator: CustomDataGen = self.Gen.copy()
        Generator.df = predict_df
        Generator.n = len(predict_df)
        _ = Generator.predict_without_state_reset(on_model=on_model)

        Generator.df = free_run_df
        Generator.n = len(free_run_df)

        pred, _, _ = Trainer.free_run_with_plots(
            on_model=on_model,
            Generator=Generator,
            plot_range=range(0, len(free_run_range) - Generator.input_width),
            compute_metrics=False,
            folder=folder + save_name,
        )
        Norm_metrics, DeNorm_metrics = (
            Generator.compute_metrics(pred, 1),
            Generator.compute_metrics(pred, 1800),
        )
        print(Norm_metrics)
        print(DeNorm_metrics)

        GeneralPlotter.violin_plot_after_fit(
            pred,
            Generator,
            path=folder + save_name + "_predict_and_free_run_violin_plot",
        )

        return pred, Norm_metrics, DeNorm_metrics

    def run_free_and_compute_metrics_within_range(
        self,
        on_model: tf.keras.models.Sequential,
        to_compare: range = range(0, 24 * 14),
        save_name: str = "",
    ) -> None:
        """
        Executes a free run on the given model over the specified range and computes
        the associated metrics. Additionally, it saves visualizations of the model
        and prints computed metrics.

        Args:
            on_model (tf.keras.models.Sequential): The model to execute the free run on.
            to_compare (range): The range over which to compare predictions and truth values.
                                Default is a two-week range.
            save_name (str): Name for saving outputs like plots. Default is an empty string.

        Side Effects:
            - Saves a plot of the model's architecture in the specified folder.
            - Prints normalized and de-normalized metrics to standard output.
            - Files are created in the workspace including model visualizations.

        Notes:
            - Utilizes TensorFlow's plot_model function for visualizing model structure.
            - Assumes `Trainer.free_run_with_plots` to perform the free run and generate plots.
            - Metrics are computed by `Imputation.__compute_metrics_on_interested_intervals`.
        """
        folder = self.workspace + on_model.name + "/"
        os.makedirs(folder, exist_ok=True)
        tf.keras.utils.plot_model(
            on_model,
            show_shapes=True,
            show_layer_names=False,
            rankdir="LR",
            dpi=96,
            show_layer_activations=True,
            to_file=folder + "model.png",
        )
        free_run_df: pd.DataFrame = self.Gen.df[
            : self.Gen.input_width + to_compare.stop
        ].copy()

        Generator: CustomDataGen = self.Gen.copy()
        Generator.df = free_run_df
        Generator.n = len(free_run_df)

        pred, _, _ = Trainer.free_run_with_plots(
            on_model=on_model,
            Generator=Generator,
            plot_range=to_compare,
            compute_metrics=False,
        )

        Norm_metrics, DeNorm_metrics = (
            Imputation.__compute_metrics_on_interested_intervals(
                pred, Generator, to_compare, folder=folder + save_name
            )
        )
        print(Norm_metrics)
        print(DeNorm_metrics)

    def copy(self):
        return copy.deepcopy(self)


class ModelFactory:
    def __init__(
        self,
        path: str = "imputation/",
        extension: str = "weights.h5",
        shape_X=(13, 8),
        shape_y=(1),
    ):
        """
        Initializes the ModelFactory class with default or provided parameters that define the model's saving path,
        file extension, and input/output dimensions.

        Args:
            path (str): Directory path where model weights are stored or will be stored.
            extension (str): File extension for the saved model weights.
            shape_X (tuple): Dimension of the input features for the models.
            shape_y (tuple): Dimension of the output target for the models.

        Attributes:
            path (str): Maintains the directory path for model weight storage.
            extension (str): Represents the file extension for model weight files.
            shape_X (tuple): Stores the dimension of model inputs.
            shape_y (tuple): Stores the dimension of model outputs.
        """
        self.path = path
        self.extension = extension
        self.shape_X = shape_X
        self.shape_y = shape_y

    def get_new_linear_model(self) -> tf.keras.models.Sequential:
        """
        Creates and returns a new linear regression model configured with TensorFlow Keras.

        This function initializes a sequential model with the following layers:
        - An Input layer, which expects an input shape defined by the class attribute `shape_X`.
        - A Flatten layer, which serves to flatten the multi-dimensional input.
        - A Dense layer, which serves as the output layer and contains a single unit whose shape is 
          defined by the class attribute `shape_y`.

        Returns:
            tf.keras.models.Sequential: A compiled Keras Sequential model configured with 
                                         an input layer, a flatten process, and a dense output layer.
        """
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(self.shape_X)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self.shape_y),
            ]
        )
        model.name = "Linear"
        return model

    def get_new_simple_lstm_model(self) -> tf.keras.models.Sequential:
        """
        Creates and returns a new simple LSTM model with a single LSTM layer followed by a dense output layer.

        This function constructs a sequential model configured with:
        - An Input layer that matches the input shape `shape_X`.
        - A single LSTM layer with 256 units.
        - A Dense layer with linear activation that outputs to the shape defined by `shape_y`.

        Returns:
            tf.keras.models.Sequential: A Keras Sequential model instance representing 
                                        a simple LSTM network with one LSTM layer.

        Side Effects:
            - Assigns a name 'LSTM_1_Layer' to the model for easy identification.

        """
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(self.shape_X)),
                tf.keras.layers.LSTM(units=256),
                tf.keras.layers.Dense(units=self.shape_y, activation="linear"),
            ]
        )

        model.name = "LSTM_1_Layer"

        return model

    def get_new_lstm_return_sequenze_model(self) -> tf.keras.models.Sequential:
        """
        Constructs and returns a new LSTM model configured to return sequences
        with two stacked LSTM layers and additional dense output layers.

        The model architecture comprises:
        - An Input layer that expects inputs shaped according to `shape_X`.
        - Two LSTM layers, where the first LSTM layer is configured to return sequences,
          facilitating stacking of a second LSTM layer.
        - A Dense layer with 64 units using a linear activation function.
        - A final Dense layer matching the output dimension specified by `shape_y`.

        Returns:
            tf.keras.models.Sequential: A Keras Sequential model with stacked
            LSTM layers and dense output layers.

        Note:
            - The model is named "Return_Sequenze" for identification.
        """
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(self.shape_X)),
                tf.keras.layers.LSTM(units=256, return_sequences=True),
                tf.keras.layers.LSTM(units=256, return_sequences=False),
                tf.keras.layers.Dense(units=64, activation="linear"),
                tf.keras.layers.Dense(units=self.shape_y, activation="linear"),
            ]
        )
        model.name = "Return_Sequenze"
        return model

    def get_new_BiLSTM_model(self) -> tf.keras.models.Sequential:
        """
        Creates and returns a new Bidirectional LSTM model designed to process sequential data twofold.
        The model structure is composed of:
        - An Input layer that handles input shaped according to `shape_X`.
        - Two Bidirectional LSTM layers; the first layer returns sequences to facilitate stacking.
        - A Dense layer with 64 units to refine the LSTM outputs.
        - A final Dense layer to adjust the outputs to the dimensions defined by `shape_y`.

        Returns:
            tf.keras.models.Sequential: A Keras Sequential model incorporating Bidirectional
                                         LSTM layers suited for sequence prediction tasks.

        Notes:
            - The model is assigned the name "Bidirectional_LSTM" for ease of identification.
        """
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(self.shape_X)),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(units=256, return_sequences=True)
                ),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(units=256, return_sequences=False)
                ),
                tf.keras.layers.Dense(units=64, activation="linear"),
                tf.keras.layers.Dense(units=self.shape_y, activation="linear"),
            ]
        )
        model.name = "Bidirectional_LSTM"
        return model

    def get_new_simple_gru_model(self) -> tf.keras.models.Sequential:
        """
        Creates and returns a new Gated Recurrent Unit (GRU) model with a single GRU layer.

        This method initializes a sequential Keras model consisting of the following layers:
        - An Input layer that accepts inputs with the shape specified by the `shape_X` attribute.
        - A GRU layer with 256 units to process the sequence data.
        - A Dense layer configured with linear activation corresponding to the output shape `shape_y`.

        Returns:
            tf.keras.models.Sequential: A Keras Sequential model configured as a simple GRU-based 
                                         architecture suitable for sequence modeling tasks.

        Side Effects:
            - Sets the name of the model to 'GRU_1_Layer' for easy identification and reference.
        """
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(self.shape_X)),
                tf.keras.layers.GRU(units=256),
                tf.keras.layers.Dense(units=self.shape_y, activation="linear"),
            ]
        )
        model.name = "GRU_1_Layer"
        return model

    def get_new_cnn_model(self) -> tf.keras.models.Sequential:
        """
        Creates and returns a new 1D Convolutional Neural Network (CNN) model.

        This function constructs a sequential model with the following layers:
        - An Input layer that matches the input shape specified by `shape_X`.
        - A 1D Convolutional layer with 256 filters and a ReLU activation function. The kernel size is set
          to cover the entire temporal dimension indicated by `shape_X[0]`.
        - A Flatten layer to transform the 2D convolutional output into a 1D vector.
        - A Dense layer aimed at producing an output of dimension `shape_y`.

        Returns:
            tf.keras.models.Sequential: A Keras Sequential model configured as a simple 1D CNN.

        Notes:
            - The model is named 'cnn_1d' for identification purposes.
        """
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.shape_X)),
                tf.keras.layers.Conv1D(256, activation="relu", kernel_size=(self.shape_X[0])),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self.shape_y)
            ]
        )
        model.name = "cnn_1d"
        return model
    
    def restore_all_models(self):
        """
        Restores all predefined machine learning models, loads their weights, and returns them.

        This method initializes a series of models defined by the ModelFactory class,
        attempts to load their corresponding pretrained weights from a specified directory,
        and then stores them in a dictionary with the model names as keys and the model 
        instances as values.

        Returns:
            dict: A dictionary where each key is a model's name (str) and the value is
                  the corresponding Keras Sequential model instance with its weights loaded. 
                  For example, {'Linear': Sequential model instance, ..., 'BiLSTM': Sequential model instance}
        """
        temp = []
        temp.append(self.get_new_linear_model())
        temp.append(self.get_new_cnn_model())
        temp.append(self.get_new_simple_lstm_model())
        temp.append(self.get_new_lstm_return_sequenze_model())
        temp.append(self.get_new_simple_gru_model())
        temp.append(self.get_new_BiLSTM_model())

        log = {}
        for model in temp:
            model.load_weights(f"./models/{self.path}/{model.name}.{self.extension}")
            log[model.name] = model
        return log


def freeze_not_dense_layer(model: tf.keras.models.Sequential):
    """
    Freezes all non-dense layers within a given Keras Sequential model and visually displays the model architecture.

    This function iterates over each layer in the input model and sets the 'trainable' attribute to False
    for any layer that is not a Dense layer. It subsequently displays the model's architecture graphically,
    highlighting which layers are trainable.

    Args:
        model (tf.keras.models.Sequential): The Keras Sequential model whose layers are to be modified.

    Side Effects:
        - Modifies the 'trainable' property of non-dense layers in the input model.
        - Utilizes the display function to present a plotted graphical representation of the model's structure,
          showing shapes, layers, trainability status, direction of data flow, and layer activations.
    """
    for layer in model.layers:  # freeze the last 3 layers
        if "Dense" not in layer.__class__.__name__:
            layer.trainable = False
    display(
        tf.keras.utils.plot_model(
            model,
            show_shapes=True,
            show_layer_names=False,
            show_trainable=True,
            rankdir="LR",
            dpi=96,
            show_layer_activations=True,
        )
    )


def fit_model(model, Gen: CustomDataGen, epochs: int):
    """
    Compiles and fits a neural network model using TensorFlow Keras with a specified data generator.

    This function takes a Keras model and a custom data generator, compiles the model with Adam 
    optimizer and mean squared error as the loss function, and fits it over a specified number of 
    epochs.

    Args:
        model: The Keras model to be compiled and trained.
        Gen (CustomDataGen): The data generator providing input data and labels.
        epochs (int): The number of training epochs to execute.

    Returns:
        A History object. Its `History.history` attribute is a record of training loss values and 
        metrics values at successive epochs, as well as validation loss values and validation 
        metrics values (if applicable).

    Side Effects:
        - Compiles the model with the specified optimizer and loss function.
        - Fits the model on the data provided by the `Gen` generator without outputting any verbosity
          (i.e., silent mode).
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
    return model.fit(Gen, epochs=epochs, verbose=0)
