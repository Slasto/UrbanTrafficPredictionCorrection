import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display 

class GeneralPlotter :
    '''
        Classe contenente ogni tipo di plot che va bene sia per predizione su singolo che  multiplo sito
    '''
    def __init__(self) :
        pass
    
    @staticmethod
    def plot_traffic( src : pd.DataFrame, title : str = "Hourly_Traffic", path : str = None):
        '''
        Generate a plot of hourly traffic over time.

        Parameters:
        src (pd.DataFrame): A DataFrame containing data for plotting, with the following columns:
            - 'hourly_traffic': The x-axis values representing hourly traffic data.
            - 'date': The y-axis values representing the date of the traffic data.

        title (str): Title of the plot. Default is "Hourly_Traffic".
        path (str): Path to save the plot. If None, the plot isn't saved. Default is None.
        '''
        src = src.copy()
        src["date"] = pd.to_datetime(src.date)
    
        plt.figure(figsize=(25,3))
        plt.title(title)

        plt.plot(src["date"],src["hourly_traffic"]*1800)
        
        plt.xlabel("time")
        plt.ylabel("hourly_traffic[entity/hour]")    
        plt.ylim(bottom = 0)

        plt.grid(visible=True)
        plt.margins(0)

        if path is not None: 
            plt.savefig(path+'.svg')
        plt.show()

    @staticmethod
    def violin_plot_after_fit(pred : np.array , Gen, path : str = None) -> None:
        '''
            Violin plot verticale del mea, con plot dei percentili, min, max, mean.
            è usabile sia con .predict sia in free run a patto che questo sia effettuato partendo dal inizio del Gen
        '''
        plot_data = pd.DataFrame().assign(error=[],classe=[], metric=[])

        base = Gen.input_width + Gen.y_offset
        true_traffic = Gen.df[Gen.y_col][base : base + len(pred)]
        
        true_traffic = np.asarray(true_traffic).flatten()
        pred         = pred.flatten()


        plot_data["error"] = abs(true_traffic-pred) * 1800

        plt.figure(figsize=(15, 2))
        plt.grid(True)
        plt.margins(0.02)
        plot_data["metric"] = "abs_err"

        sns.violinplot(
            data=plot_data,
            y="metric",
            x='error',
            hue="metric",
            inner='quarter',
            native_scale=True,
            linewidth=1,
            gridsize=1000,
            cut=0,
            bw_method='silverman',
            common_norm=True,
            fill=True,
            legend=False,
            palette='Set3'
        )
        plt.xticks(ticks = range( int(plot_data["error"].min()), int(plot_data["error"].max()), int(plot_data["error"].max()/25)))
        if path is not None: 
            plt.savefig(path + '.svg')
        plt.show()
        display(plot_data.describe().drop(columns="classe").T.drop(columns="count"))
        return

    @staticmethod
    def violin_plot_mea_mse(Generator, predictions : np.array, mae: bool = True, mse: bool = False, path : str = None ) -> None:
        '''
            Violin plot di absolute_error|squared_error
        '''
        if mae is False and mse is False:
            return print('wait, thats illegal')

        true_traffic = Generator.df[Generator.y_col][Generator.input_width + Generator.y_offset : Generator.input_width + Generator.y_offset + len(predictions)]
        true_traffic = np.asarray(true_traffic).flatten()
        
        predictions  = predictions.flatten()
        
        plot_data = {}

        if mae :
            temp = pd.DataFrame( abs(true_traffic-predictions.flatten()) * 1800)
            temp.rename(columns={"hourly_traffic": "error"}, inplace=True)
            temp['classe'] = 'absolute_error'
            plot_data['absolute_error'] =  temp


        if mse:
            temp = pd.DataFrame(((true_traffic - predictions.flatten()) * 1800) ** 2)
            temp.rename(columns={"hourly_traffic": "error"}, inplace=True)
            temp['classe'] = 'square_error'
            plot_data['square_error'] =  temp


        for key, value in plot_data.items():
            plt.figure(figsize=(3, 8))
            plt.title(key)

            sns.violinplot(data=value, y="error", hue="classe", legend=False, fill=True, inner='quart', palette="Set3")

            plt.yticks(ticks = range( int(value["error"].min()), int(value["error"].max()), int(value["error"].max()/25)), rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)

            plt.margins(0.02)
            if path is not None: 
                plt.savefig(path +"_"+ key +'.svg')
            plt.show()
            display(value.describe().T.drop(columns=["count"]))
        return
    
    @staticmethod
    def violin_plot_compare(src : pd.DataFrame, cut  : int = 0, path : str = None):
        '''
            src = dataFrame con 3 colonne: ['error', 'set', 'classe'] 
                    - error : Absolute error
                    - set   : String id del generator
                    - classe: Nome modello
            cut = quanto tagliare dal max del errore generale
        '''
        src = src[src["error"] < src['error'].max() - cut]
        
        plt.figure(figsize=(10,7))
        plt.title("Absolute error")

        sns.violinplot(
            data=src,
            y="error",
            x='set',
            hue="classe",
            split=True,
            fill=True,
            inner='quarter',
            native_scale=True,
            linewidth=1,
            gridsize=1000,
            cut=0,
            bw_method='silverman',
            #common_norm=True,
            palette='Set3')
        
        plt.grid()
        plt.margins(0.02)
        if(path is not None): 
            plt.savefig(path + '.svg')

        plt.show()

    @staticmethod
    def compute_site_correlation(src: pd.DataFrame, site_no_list: list, save_fig_on_folder : str = None):
        correlation = pd.DataFrame()
        for site_no in site_no_list:
            correlation[f"Site {site_no}"] = src[src.site_no == site_no][
                "hourly_traffic"
            ].reset_index(drop=True)

        # --- --- --- --- pearson
        plt.figure(figsize=(15, 4))

        plt.subplot(1, 2, 1)
        rounded_corr_matrix = correlation.corr(method="pearson").round(2)
        heatmap = sns.heatmap(rounded_corr_matrix, annot=True)
        heatmap.set_title(
            "Heatmap w\\Pearson’s correlation coefficient",
            fontdict={"fontsize": 12},
            pad=12,
        )
        # --- --- --- --- spearman
        plt.subplot(1, 2, 2)
        rounded_corr_matrix = correlation.corr(method="spearman").round(2)
        heatmap = sns.heatmap(rounded_corr_matrix, annot=True)
        heatmap.set_title(
            "Heatmap w\\Spearman rank correlation", fontdict={"fontsize": 12}, pad=12
        )
        if save_fig_on_folder is not None :
            stringa = '_'.join(map(str, site_no_list))
            plt.savefig(f"{save_fig_on_folder}correlation_between_{stringa}.svg")
        plt.show()
        del rounded_corr_matrix, heatmap, correlation

class OnePlotter :
    '''
        Classe contenente ogni tipo di plot che va bene SOLO per predizione su singolo sito
    '''
    def __init__(self) :
        pass

    @staticmethod
    def compare_real_pred(y_gen, predictions : np.array, plot_range : range = range(0, 100), title : str = "None", path : str = None) :
        '''
            Plot di comparazione del hourly traffic vero e quello predetto di un solo sito
            plot_range deve essere tale che:
                - start: inizia dal primo punto preso da y (es se input_width = 13 allora start = 0 -> df[13])
                - stop : 
        '''
        y_col_w_date = y_gen.y_col.copy()
        y_col_w_date.append("date")

        plot_data = y_gen.df[plot_range.start : plot_range.stop + y_gen.input_width + y_gen.y_offset][y_col_w_date].copy()

        y_p = predictions.ravel()[plot_range]
        y_p = np.insert(y_p, 0, np.array(plot_data[y_gen.y_col][:y_gen.input_width + y_gen.y_offset]).ravel())

        plot_data["prediction"] = y_p
        plot_data.date = pd.to_datetime(plot_data.date).dt.strftime('%Y-%m-%d %H')+"h"
        plot_data.set_index("date",inplace=True)


        plt.figure(figsize=(25,6))
        if title != "None" :
            plt.title(title)

        plt.plot(plot_data[y_gen.y_col]* 1800, marker='.', label='True')
        plt.plot(plot_data["prediction"]    * 1800, marker='*', label='Predicted')


        plt.margins(0.02)
        plt.ylim(bottom=0)
        plt.plot([y_gen.input_width + y_gen.y_offset -1, y_gen.input_width + y_gen.y_offset-1], plt.ylim(), '--', color='red', label="Start prediction")

        plt.xticks(ticks=range(0, len(plot_data), len(plot_data)//25), rotation=20) # 25 tick

        plt.ylabel("hourly_traffic[entity/hour]")
        plt.xlabel("time")

        plt.grid()
        plt.legend(frameon=True, ncol=6, loc='upper left')
        if path is not None : 
            plt.savefig(path+'.svg')
        plt.show()
    
    @staticmethod
    def multiple_model_compare_on_real(y_gen, predictions : dict, plot_range : range = range(0, 24*7), title : str = "None", path : str = None) :
        '''
            Plot di comparazione del hourly traffic vero con quello predetto da più modelli
        '''
        y_col_w_date = y_gen.y_col.copy()
        y_col_w_date.append("date")

        plot_data = y_gen.df[plot_range.start : plot_range.stop + y_gen.input_width + y_gen.y_offset][y_col_w_date].copy()

        for key, value in predictions.items():
            y_p = value.ravel()[plot_range]
            y_p = np.insert(y_p, 0, np.array(plot_data[y_gen.y_col][:y_gen.input_width + y_gen.y_offset]).ravel())

            plot_data[key] = y_p
            
        plot_data.date = pd.to_datetime(plot_data.date).dt.strftime('%Y-%m-%d %H')+"h"
        plot_data.set_index("date",inplace=True)


        plt.figure(figsize=(25,6))
        if title != "None" :
            plt.title(title)

        plt.plot(plot_data[y_gen.y_col]* 1800, marker='.', label='True')
        
        for key, value in predictions.items():
            plt.plot(plot_data[key]    * 1800, marker='.', label=key)


        plt.margins(0.02)
        plt.ylim(bottom=0)
        plt.plot([y_gen.input_width + y_gen.y_offset -1, y_gen.input_width + y_gen.y_offset-1], plt.ylim(), '--', color='red', label="Start prediction")

        plt.xticks(ticks=range(0, len(plot_data), len(plot_data)//25), rotation=20) # 25 tick

        plt.xlabel('time')
        plt.ylabel('hourly_traffic[entity/hour]')


        plt.grid()
        plt.legend(frameon=True, ncol=6)
        if path is not None : 
            plt.savefig(path+'.svg')
        plt.show()