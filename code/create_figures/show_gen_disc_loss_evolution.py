from pathlib import Path
from matplotlib import pyplot
from scipy.stats import shapiro
import random
import imageio
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re
from matplotlib.figure import figaspect
import seaborn as sns
sns.set(style="whitegrid")
sns.set(font_scale=2)
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
import seaborn as sns

data_folder = '../../data/distributed_log_files/'
data_evolution_folder = '../../data/evolution/'
images_folder = '../../images/'


def get_stats_to_paint(df):
    final_result = df.iloc[19].values.tolist()
    df['max'] = df[list(df)].max(axis=1)
    df['min'] = df[list(df)].min(axis=1)
    df['mean'] = df[list(df)].mean(axis=1)
    df['median'] = df[list(df)].median(axis=1)

    return df

def show_evolution_of_df(data_df):
    data_df = get_stats_to_paint(data_df)
    sns.set(style="whitegrid")
    sns.set_style("ticks")
    x = np.arange(data_df.shape[0])
    fig = plt.figure()
    ax = plt.axes()
    ax.tick_params(direction='out')
    ax.plot(x, data_df['median'].values, 'b-')
    ax.fill_between(x, data_df['min'].values, data_df['max'].values, color='b', alpha=0.3)

    # no plot without labels on the axis
    ax.set_xlabel(r"Training epoch", fontweight='bold')
    ax.set_ylabel(r"FID score", fontweight='bold')

    # always call tight_layout before saving ;)
    fig.tight_layout()
    fig.show()

def show_accuracy_label_evolution(data_df, image_path, epoch):
    labels = list(range(10))
    sns.set(style="whitegrid")
    sns.set_style("ticks")
    fig, ax = plt.subplots()
    plt.ylim(0, 100)
    plt.bar(labels, data_df.iloc[epoch])
    plt.title('MNIST - Training epoch: {}'.format(epoch))
    plt.ylabel('Classification accuracy (%)')
    plt.xticks(labels)
    plt.savefig(image_path)
    #plt.show()



def createa_video(data_df, step=1):
    tmp = '/tmp/'
    images = []
    Path(images_folder + tmp).mkdir(parents=True, exist_ok=True)

    max_generations = data_df.shape[0]
    for epoch in range(max_generations):
        if epoch % step == 0:
            image_path = images_folder+ tmp + 'for-video-{:04d}'.format(epoch) + '.png'
            show_losses_evolution(data_df, epoch, step)
            print('Created frame {}'.format(epoch))
            images.append(imageio.imread(image_path))

    # Create some frames to stop at the ende
    for epoch in range(30):
        if epoch % step == 0:
            image_path = images_folder + tmp + 'for-video-{:04d}'.format(epoch) + '.png'
            show_losses_evolution(data_df, max_generations + epoch, step)
            print('Created frame {}'.format(epoch))
            images.append(imageio.imread(image_path))
    imageio.mimsave(images_folder + '/video.gif', images)
    print('Finished: Created animation in file {}'.format('video.gif'))


def show_evolution(log_file):
    data_df = pd.read_csv(log_file, index_col=False)
    show_evolution_of_df(data_df)

def show_losses_evolution(data_df, epoch=1000, step_size=10, show_full_line = True): # data_acc, data_fid,  image_path, epoch):
    data_gen_loss = data_df['generator_loss'].tolist()
    data_disc_loss = data_df['discriminator_loss'].tolist()
    gen_loss = data_gen_loss[::step_size]
    disc_loss = data_disc_loss[::step_size]

    sns.set(style="whitegrid")
    sns.set_style("ticks")
    epoch = int(epoch / step_size)
    print(epoch)
    x = np.arange(0, epoch)
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    x_label = 'Training epoch' if step_size > 1 else 'Training epoch (x {})'.format(step_size)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Generator loss', color=color)
    #ax1.set_ylim(0,200)
    ax1.plot(x, gen_loss[:epoch], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Discriminator loss', color=color)  # we already handled the x-label with ax1
    #ax2.set_ylim(0, 100)
    ax2.plot(x, disc_loss[:epoch], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if show_full_line: plt.xlim(0, int(len(data_gen_loss)/ step_size))
    plt.show()
    #plt.savefig(image_path)

def split_equal(data):
    container = data.split("=")
    return container[0], container[1]

def get_losses(analized_data):
    return float(split_equal(analized_data[1])[1]), float(split_equal(analized_data[2])[1])

def create_df_from_log_file(filename):
    data = []
    f = open(data_folder + filename, 'r')
    line = f.readline()
    while line:
        if 'Iteration=' in line:
            data_dict = dict()
            splitted_data = re.split("- |,|%", line)
            analized_data = splitted_data[3:9]
            data_dict['generator_loss'], data_dict['discriminator_loss'] = get_losses(analized_data)
            data.append(data_dict)
        line = f.readline()
    if len(data)>0:
        return pd.DataFrame(data)

def get_list_of_df_from_csv(csv_file):
    df_list = []
    df = pd.read_csv(data_evolution_folder + csv_file)
    clients = int(df.shape[1] / 2)
    for client in range(clients):
        gen_loss_label = 'gen_loss-{}'.format(client)
        disc_loss_label = 'disc_loss-{}'.format(client)
        aux_df = df[[gen_loss_label, disc_loss_label]]
        aux_df = aux_df.rename(columns={gen_loss_label: 'generator_loss', disc_loss_label: 'discriminator_loss'})
        df_list.append(aux_df)
    return df_list

use_log_file = False
use_csv_file = not use_log_file
if use_log_file:
    log_file = 'lipizzaner_2020-07-10_11-09.log'
    losses_df = create_df_from_log_file(log_file)
    show_losses_evolution(losses_df, 1000, 10) #, data_acc, data_fid,  image_path, epoch):
if use_csv_file:
    csv_file = 'mnist-gen_vs_disc_loss-evolution-lipizzaner_2020-05-17_21-30.csv'
    lossesdf_list = get_list_of_df_from_csv(csv_file)
    for losses_df in lossesdf_list:
        show_losses_evolution(losses_df, 100, 1)
# createa_video(acc_label_log_file, acc_log_file, fid_log_file, client_id, step=1)
#show_evolution_fid_vs_acc(acc_log_file, fid_log_file)
#show_evolution(acc_log_file)

#createa_video(acc_label_log_file, client_id)

#show_evolution('/home/jamal/Documents/Research/sourcecode/evaluate-lipizzaneer-output/data/output/evolution/mnist-training_accuracy-evolution-lipizzaner_2020-05-16_14-46.csv')