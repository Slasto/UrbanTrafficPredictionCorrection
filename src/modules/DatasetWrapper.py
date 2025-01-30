import os
import time
import copy
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from IPython.display import display

from modules.kerasGenerator import CustomDataGen
from modules.Plotter import GeneralPlotter

class Dataset :
# --- --- --- Variabili e Costruttori --- --- --- 
    def __init__(self, path : str = '../data/3_DataSet_Normalized.gz') :
        """
        Initialize the Dataset instance.

        Parameters:
        path (str): The file path to the gzipped dataset CSV. Defaults to '../data/3_DataSet_Normalized.gz'.

        Initializes:
        df (pd.DataFrame): The dataframe containing the dataset read from the specified path, sorted by 'date' and 'site_no'.
        """
        self.df = pd.read_csv(filepath_or_buffer = path, compression='gzip', date_format=["date"])
        self.df.sort_values(by=['date', 'site_no'], inplace=True)
            
    def __call__(self, src: int) -> pd.DataFrame : 
        """
        Callable method, allowing the object to be called as a function.

        Parameters:
        src (int): Determines which dataset to return. 
                   - If 1, returns the full dataset as a DataFrame.
                   - If 2, returns the site-specific data (if previously set via `select_site`).

        Returns:
        pd.DataFrame or None: The requested dataset, or None if `src` is not 1 or 2.
        """
        if   src == 1:
            return self.df
        elif src == 2:
            return self.site_data
        return None
    
    def copy(self) :
        """
        Create a deep copy of the Dataset instance.

        Returns:
        Dataset: A new instance of Dataset, which is a deep copy of the current instance.
        """ 
        return copy.deepcopy(self)

# --- --- --- Metodi --- --- ---

    def select_site(self, site_no : int, hide_in_summary : tuple[str, ...] = ["hour_of_day(sin)", "hour_of_day(cos)", "day_of_week(sin)", "day_of_week(cos)", "site_no", "holiday"]) -> pd.DataFrame:
        """
            Select data for a specific site and perform analysis.

            Parameters:
            site_no (int): The site number to select from the dataset.
            hide_in_summary (tuple[str, ...]): Columns to hide in the summary statistics. Defaults to ["hour_of_day(sin)", "hour_of_day(cos)", "day_of_week(sin)", "day_of_week(cos)", "site_no", "holiday"].

            Actions:
            - Sets `self.site_no` to the target site number.
            - Filters `self.df` to `self.site_data` for the selected site.
            - If the site data is empty, prints an error message and resets site attributes to None.
            - If data exists, generates traffic plots and displays descriptive statistics excluding hidden columns.
            - Prints the site type based on available signalization data.

            Returns:
            pd.DataFrame: The dataframe containing the filtered data for the specified site.
            """
        self.site_no   = site_no
        self.site_data = self.df[self.df.site_no == site_no].reset_index(drop=True)

        if self.site_data.empty :
            print("Invalid site_no!")
            self.site_no = None
            self.site_data = None
        else :
            GeneralPlotter.plot_traffic(src=self.site_data, title=f"Sito number id :{site_no}")
            display(self.site_data.describe().drop(columns=hide_in_summary).T.drop(columns=["count"]))
            site_desc = {
                (True, False, False):"SIGNALISED_4_WAY_INTERSECTION", 
                (False, True, False):"SD_SIGNALISED_PEDESTRIAN_CROSSING", 
                (False, False, True):"SD_SIGNALISED_T_JUNCTION"
            }
            result   = self.site_data["SD_SIGNALISED_4_WAY_INTERSECTION"][0], self.site_data["SD_SIGNALISED_PEDESTRIAN_CROSSING"][0],self.site_data["SD_SIGNALISED_T_JUNCTION"][0]

            print(f"Site type: {site_desc[result]}")
        
        return self.site_data
    
    def select_multiple_site(self,sites_no: list[int], targets: list[int])-> pd.DataFrame:
        """
        Select multiple site data and prepare the dataset with selected features.

        Parameters:
        sites_no (list[int]): List of site numbers to use as predictors for the dataset.
        targets (list[int]): List of site numbers to be used as target variables.

        Returns:
        Tuple[pd.DataFrame, list[str], list[str]]: The transformed dataframe and lists of predictor and target column names.

        Actions:
        1. Initialize empty dataframe and column lists for predictors (X_col) and targets (Y_col).
        2. For the first target in the preferences, filter the dataframe and rename the traffic column to represent the target.
        3. If there are more targets, extend the dataframe with additional target columns.
        4. For each site in the site_no list, add its hourly traffic data to the dataframe as predictors.
        5. Extend the predictor columns list with common time and weather features.
        """
        self.site_no = sites_no
        self.targets = targets

        self.multiple_site_data = pd.DataFrame()
        self.X_col = []
        self.Y_col = []

        self.multiple_site_data = self.df[self.df.site_no == targets[0]].drop(columns=['site_no','SD_SIGNALISED_4_WAY_INTERSECTION', 'SD_SIGNALISED_PEDESTRIAN_CROSSING', 'SD_SIGNALISED_T_JUNCTION']).reset_index(drop=True).rename(columns={"hourly_traffic": f"target_{targets[0]}"})
        self.Y_col.append(f'target_{targets[0]}')

        if len(targets) > 1:
            for i in range(1, len(targets)):
                target = targets[i]
                self.multiple_site_data[f'target_{target}'] = self.df[self.df.site_no == target]['hourly_traffic'].reset_index(drop=True)
                self.Y_col.append(f'site_{target}')


        for i in range(len(sites_no)):
            site_n = sites_no[i]
            self.multiple_site_data[f'site_{site_n}'] = self.df[self.df.site_no == site_n]['hourly_traffic'].reset_index(drop=True)
            self.X_col.append(f'site_{site_n}')
        self.X_col.extend(["hour_of_day(sin)", "hour_of_day(cos)", "day_of_week(sin)", "day_of_week(cos)", "holiday", "temperature_2m", "apparent_temperature", "relative_humidity_2m", "precipitation", "wind_speed_10m", "cloud_cover"])


        display(self.multiple_site_data)
        return self.multiple_site_data, self.X_col, self.Y_col

    def split_and_get_generators_multi(self,data : pd.DataFrame , training : int = 3, validation : int = 1, test : int = 1,
                     batch_size : int  = 32, input_size : int = None, output_size = 1,
                     X_col : list[str] = None,
                     y_col : list[str] = ["hourly_traffic"],
                     y_offset : int = 0,
                     stop : bool = False) -> tuple[CustomDataGen, CustomDataGen, CustomDataGen] :
        """
        Split the dataset into training, validation, and test sets and return corresponding data generators.

        Parameters:
        data (pd.DataFrame): The dataset to split.
        training (int): The number of years to use for training. Defaults to 3.
        validation (int): The number of years to use for validation. Defaults to 1.
        test (int): The number of years to use for testing. Defaults to 1.
        batch_size (int): The size of the batches to generate. Defaults to 32.
        input_size (int, optional): The size of the input sequence for the generators. Defaults to None.
        output_size (int): The size of the output sequence for the generators. Defaults to 1.
        X_col (list[str], optional): List of column names to use as input features. Defaults to None.
        y_col (list[str]): List of column names to use as output features. Defaults to ["hourly_traffic"].
        y_offset (int): Offset for the output sequence. Defaults to 0.
        stop (bool): If True, limit the test set within a certain range. Defaults to False.

        Returns:
        tuple[CustomDataGen, CustomDataGen, CustomDataGen]: A tuple containing the training, validation, and test data generators.

        Notes:
        - Checks if the sum of training, validation, and test years exceeds 5 and returns if true.
        - Checks if both X_col and input_size are not provided and returns if true.
        """
        if (training + validation + test) > 5 :
            print("Dataset E: Invalid split size train_size + validation_size + test_size > 5 year")
            return
        if  X_col is None and input_size is None:
            print("Dataset E: I need input_size and X_col")
            return
        
        BASE = 2010
        if stop is False: 
            self.training_set   = data[(data['date'] >=  "2010-01-01 00:00:00") & (data['date'] < f"{BASE+training}-01-01 00:00:00")].copy()
            self.validation_set = data[(data['date'] >= f"{BASE+training}-01-01 00:00:00") & (data['date'] < f"{BASE+training+validation}-01-01 00:00:00")].reset_index(drop=True)
            self.test_set       = data[(data['date'] >= f"{BASE+training+validation}-01-01 00:00:00")].reset_index(drop=True)
        else:
            self.training_set   = data[(data['date'] >=  "2010-01-01 00:00:00") & (data['date'] < f"{BASE+training}-01-01 00:00:00")].copy()
            self.validation_set = data[(data['date'] >= f"{BASE+training}-01-01 00:00:00") & (data['date'] < f"{BASE+training+validation}-01-01 00:00:00")].reset_index(drop=True)
            self.test_set       = data[(data['date'] >= f"{BASE+training+validation}-01-01 00:00:00")& (data['date'] < f"{BASE+training+validation+test}-01-01 00:00:00")].reset_index(drop=True)

        self.GenTraining   = CustomDataGen(self.training_set  , X_col, y_col, batch_size, input_size, output_size, y_offset)
        self.GenValidation = CustomDataGen(self.validation_set, X_col, y_col, batch_size, input_size, output_size, y_offset)
        self.GenTest       = CustomDataGen(self.test_set      , X_col, y_col, batch_size, input_size, output_size, y_offset)

        return (self.GenTraining, self.GenValidation, self.GenTest)
    
    #TODO: Nuova funzione che permette di avere a partire di una data n giorni consecutivi divisi in len({}) dataFrame 

    def split_and_get_generators(self, training : int = 3, validation : int = 1, test : int = 1,
                     batch_size : int  = 32, input_size : int = None, output_size = 1,
                     X_col : list[str] = None,
                     y_col : list[str] = ["hourly_traffic"],
                     y_offset : int = 0
                     ) -> tuple[CustomDataGen, CustomDataGen, CustomDataGen] :
        """
        Split the site's dataset into training, validation, and test sets, and return data generators.

        Parameters:
        training (int): Number of years to use for the training set. Defaults to 3.
        validation (int): Number of years to use for the validation set. Defaults to 1.
        test (int): Number of years to use for the test set. Defaults to 1.
        batch_size (int): The batch size for the data generators. Defaults to 32.
        input_size (int, optional): Input sequence length for the generators. Defaults to None.
        output_size (int): Output sequence length for the generators. Defaults to 1.
        X_col (list[str], optional): List of column names to use as the input features. Defaults to None.
        y_col (list[str]): List of column names to use as the output features. Defaults to ["hourly_traffic"].
        y_offset (int): Offset for the output sequence, measured in number of steps. Defaults to 0.

        Returns:
        tuple[CustomDataGen, CustomDataGen, CustomDataGen]: A tuple containing generators for the training, validation, and test sets.

        Notes:
        - Ensures that the sum of training, validation, and test durations does not exceed 5 years.
        - Requires either `X_col` or `input_size` to be specified to proceed.
        """
        if (training + validation + test) > 5 :
            print("Dataset E: Invalid split size train_size + validation_size + test_size > 5 year")
            return
        if  X_col is None and input_size is None:
            print("Dataset E: I need input_size and X_col")
            return
        
        BASE = 2010
        training_set   = self.site_data[(self.site_data['date'] >=  "2010-01-01 00:00:00") & (self.site_data['date'] < f"{BASE+training}-01-01 00:00:00")].copy()
        validation_set = self.site_data[(self.site_data['date'] >= f"{BASE+training}-01-01 00:00:00") & (self.site_data['date'] < f"{BASE+training+validation}-01-01 00:00:00")].reset_index(drop=True)
        test_set       = self.site_data[(self.site_data['date'] >= f"{BASE+training+validation}-01-01 00:00:00")].reset_index(drop=True)

        GenTraining   = CustomDataGen(training_set  , X_col, y_col, batch_size, input_size, output_size, y_offset)
        GenValidation = CustomDataGen(validation_set, X_col, y_col, batch_size, input_size, output_size, y_offset)
        GenTest       = CustomDataGen(test_set      , X_col, y_col, batch_size, input_size, output_size, y_offset)

        return (GenTraining, GenValidation, GenTest)

# --- --- --- Proprietà --- --- --
    @property
    def max_site_no(self) -> int :
        return max(self.df["site_no"])
        
class Log :
    def __init__(self, Workspace:str = 'temp/'+time.strftime("%Y%m%d-%H%M")+'/') :
        """
        Initialize the Log instance.

        Parameters:
        Workspace (str): Path for the workspace directory. Defaults to 'temp/' with a timestamp.

        Initializes:
        log_test_performance (dict): Dictionary to log test performance metrics.
        log_val_performance (dict): Dictionary to log validation performance metrics.
        log_performance (dict): Dictionary to log overall performance metrics.
        my_models (dict): Dictionary to log models.

        Attributes:
        log (dict): Aggregated log containing all performance metrics and model logs.
        Workspace (str, optional): Directory for logging data. Created if not None.
        """
        self.log_test_performance = {}
        self.log_val_performance = {}
        self.log_performance = {}
        self.my_models = {}

        self.log = {
            'my_models': self.my_models,
            'performance': self.log_performance,
            'val_performance': self.log_val_performance,
            'test_performance' : self.log_test_performance
        }
        self.Workspace = None
        if Workspace is not None:
            self.Workspace     = 'img/' + Workspace +'/log/'
            os.makedirs(self.Workspace, exist_ok=True)
    
    def __setitem__(self, key, value):
        self.log[key] = value

    def __getitem__(self, key):
        return self.log[key]

    def __repr__(self) :
        return repr(self.log)
        
    def plot_bar_metric(self, metric_name : str = 'mean_absolute_error') :
        """
            Plot a bar chart displaying validation and test performance metrics.

            Parameters:
            metric_name (str): The name of the metric to plot. Defaults to 'mean_absolute_error'.

            Actions:
            - Generates a bar chart comparing validation and test metrics over multiple logs.
            - Saves the plot as a SVG file in the specified workspace directory.

            Visual:
            - Bars for validation and test metrics are plotted side-by-side for each log entry.
            - X-axis represents different models/entries in the log.
            - Y-axis represents the value of the specified metric.
            - Includes grid lines, labels, and legend for clarity.
            """
        width = 0.3
        mp_range = np.arange(len(self.log_performance))
        
        val_mae  = [v[metric_name] for v in self.log_val_performance.values()]
        test_mae = [v[metric_name] for v in self.log_test_performance.values()]

        plt.figure(figsize=(15,5))
        plt.title(metric_name)

        plt.bar(mp_range - 0.17, val_mae, width, label='Validation')
        plt.bar(mp_range + 0.17, test_mae, width, label='test')

        plt.xticks(ticks=mp_range, labels=self.log_performance.keys(), rotation=90)

        plt.yticks(np.arange(0, max(max(val_mae), max(test_mae)), step=0.01))
        plt.ylabel(f'{metric_name} (average over all times and outputs)')
        
        plt.legend()
        plt.grid(axis='y')
        plt.savefig(self.Workspace+'bar_metric.svg')
        plt.show()
        return

    def plot_metric(self, metric_name : str = 'mean_absolute_error') :
        """
        Plot performance metrics over models.

        Parameters:
        metric_name (str): Name of the metric to plot. Defaults to 'mean_absolute_error'.

        Actions:
        - Plots the specified metric for both validation and test performance across different models.
        - Saves the plot as 'plot_metric.svg' in the workspace directory.

        Visual:
        - X-axis represents different models.
        - Y-axis represents the value of the specified metric.
        - Includes a legend, grid, and appropriate labels for clarity.
        """
        plotrange = np.arange(len(self.log_performance))
        
        val_mae  = [v[metric_name] for v in self.log_val_performance.values()]
        test_mae = [v[metric_name] for v in self.log_test_performance.values()]

        plt.figure(figsize=(15,5))
        plt.title(metric_name)
        plt.plot(plotrange, val_mae, marker="." ,label="Validation")
        plt.plot(plotrange, test_mae,marker=".", label="Test")

        plt.xticks(ticks=plotrange, labels=self.log_performance.keys(), rotation=90)
        plt.xlabel("model")

        plt.ylabel("hourly_traffic")
        
        plt.grid(visible=True)
        plt.legend()
        plt.savefig(self.Workspace+'plot_metric.svg')
        plt.show()
        return
    
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert logged performance metrics to a pandas DataFrame.

        Returns:
        pd.DataFrame: A DataFrame containing training, validation, and test performance metrics.

        Actions:
        - Converts each performance log (training, validation, test) to a DataFrame.
        - Renames validation and test DataFrame columns to include 'val_' and 'test_' prefixes respectively.
        - Merges the DataFrames column-wise on their indices to create a single comprehensive DataFrame.
        """
        train = pd.DataFrame.from_dict(self.log_performance, orient='index')

        val = pd.DataFrame.from_dict(self.log_val_performance, orient='index')
        val = val.rename(columns=lambda x: f"val_{x}")
    
        test = pd.DataFrame.from_dict(self.log_test_performance, orient='index')
        test = test.rename(columns=lambda x: f"test_{x}")

        return pd.merge(pd.merge(train, val, left_index=True, right_index=True), test, left_index=True, right_index=True)
    
    def save_into_workspace(self) :
        self.to_dataframe().to_csv(self.Workspace+'metrics.csv')