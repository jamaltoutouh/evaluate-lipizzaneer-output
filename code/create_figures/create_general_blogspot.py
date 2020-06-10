from pathlib import Path
from matplotlib import pyplot
from scipy.stats import shapiro
import random
import imageio
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.figure import figaspect
import seaborn as sns
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
import seaborn as sns

data_folder = '../../data/'
images_folder = '../../images/'

whole_metrics_list = ['fid','tvd','execution_time','best FID client','n_iterations','grid_size','label_rate','batch_size','acc stats client id','acc stats client folder','most voted acc','max acc','mean acc','std acc','improvement over max acc','improvement over mean acc']

list_grid_size = [1, 4, 9, 16]

# We use that for Semi-supervised learning with 100 labels
# Batch size=600 and label_rate=0.00167
def get_elements_by_grid_size(data_df, grid_size, metric, label_rate=0.00167, batch_size=600):
    data = data_df[(data_df['label_rate'] <= label_rate) & (data_df['batch_size'] == batch_size) & (data_df['grid_size'] == grid_size)][['label_rate', 'grid_size', metric]]
    return data

def create_df_boxplot(data_df, metric, list_metric_group=[]):
    list_of_df = list()
    for grid_size in list_grid_size:
        data = get_elements_by_grid_size(data_df, grid_size, metric)
        list_of_df.append(data)
        print('Processed grid_size={}. Got df with shape {}.'.format(grid_size, data.shape))
    return pd.concat(list_of_df)

def remove_no_finished(data_df, iterations):
    return data_df[(data_df['n_iterations'] == iterations) & (data_df['mean acc'] > 0.5)]

# def remove_no_semisupervised(data_df):
#     return data_df[(data_df['mean acc'] > 0.5)]

def create_boxplot(data_df, metric, metric_name):
    w, h = figaspect(3 / 3)
    f, ax = plt.subplots(figsize=(w, h))
    #my_pal = {'2014': (0.67, 0.74, 0.63, 1), '2015': (0.67, 0.74, 0.63, 1), '2016': (0.67, 0.74, 0.63, 1),
    #          '2017': (0.67, 0.74, 0.63, 1), '2018': (0.67, 0.74, 0.63, 1), '2019': (0.92, 0.4, 0.2, 0.9)}
    grid_sizes = ['1x1', '2x2', '3x3', '4x4']
    sns.set(style="ticks")
    sns.set(font_scale=2)
    #data_df[metric] = data_df[metric] * 100
    ax = sns.boxplot(x='grid_size', y=metric, data=data_df) #, palette=my_pal)
    ax.tick_params(labelsize=20)
    ax.set_xlabel('Grid size', fontsize=20)
    ax.set_ylabel(metric_name, fontsize=20)
    ax.set_xticklabels(grid_sizes)
    # ax.set_ylim(0, 1.25)
    #ax.set_ylabel(pollutant_graph_label, fontsize=20)
    plt.tight_layout()
    plt.savefig('../../images/boxplot-{}.png'.format(metric))
    plt.show()




data_file1 = 'mnist-summary_results-8.csv'
data_file2 = 'mnist-summary_results-sub.csv'
data_file3 = 'mnist-summary_results-9.csv'

data_df1 = pd.read_csv(data_folder + data_file1, index_col=False)
data_df2 = pd.read_csv(data_folder + data_file2, index_col=False)
data_df3 = pd.read_csv(data_folder + data_file3, index_col=False)


iterations = 100
data_df1 = remove_no_finished(data_df1, iterations)
data_df2 = remove_no_finished(data_df2, iterations)
data_df3 = remove_no_finished(data_df3, iterations)




used_metrics_list = ['fid','tvd','execution_time','n_iterations','grid_size','label_rate','batch_size', 'most voted acc','max acc','mean acc','improvement over max acc','improvement over mean acc']
used_metric_names_list = ['FID','TVD','Execution time','n_iterations','Grid size','label_rate','batch_size', 'Accuracy (%)', 'Accuracy (%)', 'Accuracy (%)','improvement over max acc','improvement over mean acc']
# i = 7 # Most voted acc
i = 8 # Max acc
i = 0 # FID
#i = 1 # TVD
#i = 2 # Execution time

data_df = pd.concat([data_df1, data_df2, data_df3])
data = create_df_boxplot(data_df, used_metrics_list[i])
create_boxplot(data, used_metrics_list[i], used_metric_names_list[i])


