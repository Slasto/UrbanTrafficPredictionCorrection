import os
import time
import shutil

import numpy as np
import pandas as pd

from modules.Plotter import GeneralPlotter, OnePlotter

import tensorflow as tf
from modules.kerasGenerator import CustomDataGen
from modules.Callbacks import GPUThermalMonitorCallback, TrainingPlot

class Trainer:    
# --- --- --- Variabili e Costruttori --- --- ---  
    OLD_X_FEATURES = ["hourly_traffic", "hour_of_day(sin)", "hour_of_day(cos)", "day_of_week(sin)", "day_of_week(cos)", "holiday", "temperature_2m","apparent_temperature","relative_humidity_2m", "precipitation", "wind_speed_10m", "cloud_cover"]
    OLD_X_WINDOWS  = 10
    
    NEW_X_FEATURES = ["hourly_traffic", "hour_of_day(sin)", "hour_of_day(cos)", "day_of_week(sin)", "day_of_week(cos)", "holiday", "temperature_2m","apparent_temperature"]
    NEW_X_WINDOWS  = 13
    
    def __init__(self, TrainingGen : CustomDataGen = None, ValidationGen : CustomDataGen= None, TestGen : CustomDataGen = None, Workspace : str = 'temp/trainer_'+time.strftime("%Y%m%d-%H%M")+'/') :
        self.GenTraining   = TrainingGen
        self.GenValidation = ValidationGen
        self.GenTest       = TestGen

        self.Workspace     = 'img/' + Workspace
        os.makedirs(self.Workspace, exist_ok=True)

        with open(f'{self.Workspace}/Trainer_info.txt', 'w') as file:
            file.write(f"Features: {self.GenTest.X_col}\n")
            file.write(f"Window length: {self.GenTest.input_width}\n")

    @property
    def shape_X(self) :
        return self.GenTraining.shape_X

    @property
    def shape_y(self) :
        return self.GenTraining.shape_y


# --- --- --- Metodi --- --- --- 

    def fit_on(self, on_model: tf.keras.models.Sequential, N_epochs : int = 15,
                                patience_RLR : int = -1, patience_ES : int = -1, learning_rate : float = 1e-3,
                                save_model : bool = False, save_step : bool = False,
                                log : dict = None,
                                Plot_only_on_final_epoch : bool = False, plot_range : range = range(0,480)
                                ) ->tuple[any,np.array] :
        
        '''
            Questo metodo comprende oltre al fit altri step utili per la valutazione del modello, come il calcolo dello Score su Train, Validation e Test con annesso violin plot
            plot di confronto pred e test, salvataggio del modello, ed aggiunta metriche al log   
        '''
        if not hasattr(self, 'GenTraining') and not hasattr(self, 'GenValidation') and not hasattr(self, 'GenTest') :
            print("Error: You need to use 'split_dataset(...)' first!")
            return
        

        folder = self.Workspace+"/"+on_model.name+"/"
        os.makedirs(folder, exist_ok=True)

        # --- ---Allenamento del modello e Score sul test --- ---
        on_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )

        tf.keras.utils.plot_model(
            on_model,
            show_shapes=True,
            show_layer_names=False,
            rankdir='LR',
            dpi=96,
            show_layer_activations=True,
            to_file=folder+"model.png"
        )

        file_path = f'./models/{on_model.name}/'
        if save_step :
            os.makedirs(file_path, exist_ok=True)
            shutil.rmtree(file_path)
        
        history = on_model.fit(
            self.GenTraining, 
            validation_data=self.GenValidation, 
            epochs=N_epochs, 
            callbacks=[
                TrainingPlot(Plot_only_on_final_epoch=Plot_only_on_final_epoch, path=folder+'training_plot'),
                GPUThermalMonitorCallback(),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_ES, restore_best_weights=True, verbose=1) if patience_ES > 0 else tf.keras.callbacks.Callback(),
                tf.keras.callbacks.ReduceLROnPlateau(patience=patience_RLR, verbose=1) if patience_RLR > 0  else tf.keras.callbacks.Callback(),
                tf.keras.callbacks.ModelCheckpoint(filepath=file_path+'{epoch:02d}-{val_mean_absolute_error:.4f}.weights.h5') if save_step else tf.keras.callbacks.Callback()
            ], 
            verbose = 2 if not Plot_only_on_final_epoch else 0
        )

        if save_model :
            file_path = f"./models/{on_model.name}.keras"
            print(f"Saving model in {file_path}")
            on_model.save(file_path)
        
        # --- --- Calcolo del mae e mse --- ---
        train, validation, test = self.evaluate_metrics_on_dataset(on_model, plot_range)

        # --- ---Aggiunta al log e salvataggio del modello --- --- 
        if log:
            print("adding result to the log...")
            log['my_models'][on_model.name]        = on_model
            log['performance'][on_model.name]      = train
            log['val_performance'][on_model.name]  = validation
            log['test_performance'][on_model.name] = test

        return history

    def evaluate_metrics_on_dataset(self, on_model : tf.keras.models.Sequential = None, plot_range : range = range(0,500), pre_name : str = "") -> tuple[dict, dict, dict]:
        '''
            Questo metodo calcola sul modello gli score rispetto al Train, Validation e Test con il plot di confronto Test vs predizione
        '''
        folder = self.Workspace+"/"+on_model.name+"/"+pre_name
        os.makedirs(folder, exist_ok=True)

        print("\n\033[1mTraining Score\033[0m:")
        predictions = on_model.predict(self.GenTraining, verbose=1)
        train = self.GenTraining.compute_metrics(predictions, 1)
        DeNorm_train = self.GenTraining.compute_metrics(predictions, 1800)

        print(f'mean_squared_error: {train["mean_squared_error"]:.4f}({np.sqrt(train["mean_squared_error"]):.4f}) - mean_absolute_error: {train["mean_absolute_error"]:.4f}')
        print(f'DeNorm_mean_squared_error: {DeNorm_train["mean_squared_error"]:.4f}({np.sqrt(DeNorm_train["mean_squared_error"]):.4f}) - DeNorm_mean_absolute_error: {DeNorm_train["mean_absolute_error"]:.4f}')
        GeneralPlotter.violin_plot_after_fit(pred = predictions, Gen = self.GenTraining, path=folder+'train_violin_plot')
        # --- --- --- --- --- ---
        print("\n\033[1mValidation Score\033[0m:")
        predictions = on_model.predict(self.GenValidation, verbose=1)
        validation = self.GenValidation.compute_metrics(predictions, 1)
        DeNorm_validation = self.GenValidation.compute_metrics(predictions, 1800)

        print(f'mean_squared_error: {validation["mean_squared_error"]:.4f}({np.sqrt(validation["mean_squared_error"]):.4f}) - mean_absolute_error: {validation["mean_absolute_error"]:.4f}')
        print(f'DeNorm_mean_squared_error: {DeNorm_validation["mean_squared_error"]:.4f}({np.sqrt(DeNorm_validation["mean_squared_error"]):.4f}) - DeNorm_mean_absolute_error: {DeNorm_validation["mean_absolute_error"]:.4f}')
        GeneralPlotter.violin_plot_after_fit(pred = predictions, Gen = self.GenValidation, path=folder+'validation_violin_plot')
        # --- --- --- --- --- ---
        _, test = self.predict_and_compare(on_model, self.GenTest, plot_range, 'Test Score', path=folder+'test_')
        # --- --- --- --- --- ---
        return train,validation,test

    def violin_plot_compare_model(self, models : dict[str : tf.keras.models.Sequential], cut  : int = 0, filename: str = ""):
        '''
            Dati n modelli organizzati in un dict vengono confrontati i relativi absolute error su Train, Validation, Test
        '''
        sets = {'Training' : self.GenTraining, 'Validation' : self.GenValidation, 'Test' : self.GenTest}
        plot_data = pd.DataFrame().assign(error=[],set=[],classe=[])
        for name, model in models.items(): 
            print(f"Computing on {name}...")
            for key, set in sets.items():
                print(f"-> {key}...")
                pred = model.predict(set, verbose=0)
                true = set.df['hourly_traffic'][set.input_width + set.y_offset : set.input_width + set.y_offset + len(pred)]

                data_abs = pd.DataFrame(abs(true-pred.flatten())*1800).rename(columns={"hourly_traffic": "error"})
                data_abs['set'] = key
                data_abs['classe'] = name
                plot_data = pd.concat([data_abs, plot_data], axis=0).reset_index(drop=True)
        
        GeneralPlotter.violin_plot_compare(plot_data, cut, path = self.Workspace + filename)

# - - - - - - Statici - - - - - -
    @staticmethod    
    def predict_and_compare(on_model : tf.keras.models.Sequential, Generator : CustomDataGen, 
                            plot_range : range = range(0,500), title : str ='Prediction Scores', path: str = None, do_plot: bool=True) -> tuple[np.array,float,float] :
        '''
            Predizione sul generator, calcolo dello score e plot di confronto dati reali vs predetti
        '''
        path_violin        = None
        path_cmp_real_pred = None

        if path is not None:
            path_violin = path+'violi_plot'
            path_cmp_real_pred = path+'cmp_real'


        print(f'\n\033[1m{title}:\033[0m')

        predictions = on_model.predict(Generator)

        metrics = Generator.compute_metrics(predictions, 1)
        DeNorm_metrics = Generator.compute_metrics(predictions)

        print(f'mean_squared_error: {metrics["mean_squared_error"]:.4f}({np.sqrt(metrics["mean_squared_error"]):.4f}) - mean_absolute_error: {metrics["mean_absolute_error"]:.4f}')
        print(f'DeNorm_mean_squared_error: {DeNorm_metrics["mean_squared_error"]:.4f}({np.sqrt(DeNorm_metrics["mean_squared_error"]):.4f}) - DeNorm_mean_absolute_error: {DeNorm_metrics["mean_absolute_error"]:.4f}')
        if do_plot is not False:
            GeneralPlotter.violin_plot_after_fit(pred = predictions, Gen = Generator, path=path_violin)
            OnePlotter.compare_real_pred(Generator, predictions, plot_range, path=path_cmp_real_pred)
        return predictions, metrics

    @staticmethod
    def free_run_with_plots(on_model : tf.keras.models.Sequential, Generator : CustomDataGen, plot_x_frame : int = -1, plot_range : range = range(0,500), compute_metrics : bool = True, folder : str = None) -> tuple[np.array, dict, dict]:
        '''
            Free run del modello sul CustomDataGen con plot di confronto dati reali vs free_run con calcolo dello score e violin plot 
        '''
        path_violin        = None
        path_cmp_real_pred = None

        if folder is not None:
            path_violin = folder + 'free_run_violi_plot'
            path_cmp_real_pred = folder + 'free_run_cmp_real'

        prediction = Generator.free_run(on_model, plot_x_frame)
        OnePlotter.compare_real_pred(Generator, prediction, plot_range, path = path_cmp_real_pred)
        
        if compute_metrics :
            norm   = Generator.compute_metrics(prediction,1)
            deNorm = Generator.compute_metrics(prediction)
            print(f'mean_squared_error: {norm["mean_squared_error"]:.4f}({np.sqrt(norm["mean_squared_error"]):.4f}) - mean_absolute_error: {norm["mean_absolute_error"]:.4f}')
            print(f'DeNorm_mean_squared_error: {deNorm["mean_squared_error"]:.4f}({np.sqrt(deNorm["mean_squared_error"]):.4f}) - DeNorm_mean_absolute_error: {deNorm["mean_absolute_error"]:.4f}')
            GeneralPlotter.violin_plot_after_fit(prediction, Generator, path_violin)
        else : 
            norm = None
            deNorm = None
        return prediction, norm, deNorm