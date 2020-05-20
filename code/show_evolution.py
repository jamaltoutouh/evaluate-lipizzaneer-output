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
sns.set(style="whitegrid")
sns.set(font_scale=2)
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
import seaborn as sns

data_folder = '../data/'
images_folder = '../images/'


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


def show_all_evolution(data_label_acc, data_acc, data_fid, image_path, epoch, max_generations):

    epoch = (max_generations-1) if epoch >= max_generations else epoch

    labels = list(range(10))
    sns.set(style="whitegrid")
    sns.set_style("ticks")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    #fig = plt.subplot(1, 2, 1)
    axes[0].set_ylim(0, 100)
    axes[0].bar(labels, data_label_acc.iloc[epoch])
    #axes[0].title('MNIST - Training epoch: {}'.format(epoch))
    axes[0].set_ylabel('Labels classification accuracy (%)')
    axes[0].set_xlabel('MNIST labels')
    axes[0].set_xticks(labels)

    data_acc = get_stats_to_paint(data_acc)
    data_fid = get_stats_to_paint(data_fid)
    sns.set(style="whitegrid")
    sns.set_style("ticks")
    x = np.arange(0, epoch + 1)
    #fig, ax1 = plt.subplot(1, 2, 2)
    color = 'tab:red'
    axes[1].set_xlabel('Training epoch')
    axes[1].set_ylabel('FID score', color=color)
    axes[1].set_ylim(0, 200)
    axes[1].plot(x, data_fid['min'][:epoch + 1], color=color)
    axes[1].tick_params(axis='y', labelcolor=color)
    ax2 = axes[1].twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Classification accuracy (%)', color=color)  # we already handled the x-label with ax1
    ax2.set_ylim(0, 100)
    ax2.plot(x, data_acc['max'][:epoch + 1], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.xlim(0, max_generations)
    plt.savefig(image_path)


def createa_video(acc_label_log_file, acc_log_file, fid_log_file, client_id, step=1):
    data_acc = pd.read_csv(acc_log_file, index_col=False)
    data_fid = pd.read_csv(fid_log_file, index_col=False)
    data_label_acc = pd.read_csv(acc_label_log_file, index_col=False)
    experiment = acc_label_log_file.split('/')[-1][:-4]
    data = dict()
    for label in range(10):
        col_name = '{} - {}'.format(client_id, label)
        data['{}'.format(label)] = data_label_acc[col_name].tolist()
    data_label_acc = pd.DataFrame(data)

    tmp = '/tmp/'
    images = []
    Path(images_folder + tmp).mkdir(parents=True, exist_ok=True)

    max_generations = data_label_acc.shape[0]
    for epoch in range(max_generations):
        if epoch % step == 0:
            image_path = images_folder+ tmp + experiment + '-{:04d}'.format(epoch) + '.png'
            #show_accuracy_label_evolution(data_df, image_path, i)
            #show_evolution_of_2df(data_acc, data_fid, image_path, epoch)
            show_all_evolution(data_label_acc, data_acc, data_fid, image_path, epoch, max_generations)
            print('Created frame {}'.format(epoch))
            images.append(imageio.imread(image_path))

    # Create some frames to stop at the ende
    for epoch in range(30):
        if epoch % step == 0:
            image_path = images_folder + tmp + experiment + '-{:04d}'.format(epoch) + '.png'
            # show_accuracy_label_evolution(data_df, image_path, i)
            # show_evolution_of_2df(data_acc, data_fid, image_path, epoch)
            show_all_evolution(data_label_acc, data_acc, data_fid, image_path, max_generations + epoch, max_generations)
            print('Created frame {}'.format(epoch))
            images.append(imageio.imread(image_path))
    imageio.mimsave(images_folder + '/' + experiment + '.gif', images)
    print('Finished: Created animation in file {}'.format(experiment + '.gif'))


def show_evolution(log_file):
    data_df = pd.read_csv(log_file, index_col=False)
    show_evolution_of_df(data_df)

def show_evolution_of_2df(data_acc, data_fid,  image_path, epoch):
    data_acc = get_stats_to_paint(data_acc)
    data_fid = get_stats_to_paint(data_fid)
    sns.set(style="whitegrid")
    sns.set_style("ticks")
    x = np.arange(0, epoch+1)
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Training epoch')
    ax1.set_ylabel('FID score', color=color)
    ax1.set_ylim(0,200)
    ax1.plot(x, data_fid['min'][:epoch+1], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Classification accuracy (%)', color=color)  # we already handled the x-label with ax1
    ax2.set_ylim(0, 100)
    ax2.plot(x, data_acc['max'][:epoch+1], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.xlim(0,100)
    plt.savefig(image_path)


def show_evolution_fid_vs_acc(acc_log_file, fid_log_file):
    data_acc = pd.read_csv(acc_log_file, index_col=False)
    data_fid = pd.read_csv(fid_log_file, index_col=False)
    show_evolution_of_2df(data_acc, data_fid)


metrics = ['fid', 'gen_loss', 'disc_loss', 'gen_lr', 'disc_lr', 'per label accuracy', 'training_accuracy']

client_id = 1
acc_label_log_file = '/home/jamal/Documents/Research/sourcecode/evaluate-lipizzaneer-output/data/evolution/mnist-per_label_accuracy-evolution-lipizzaner_2020-05-17_08-21.csv'
acc_log_file = '/home/jamal/Documents/Research/sourcecode/evaluate-lipizzaneer-output/data/evolution/mnist-training_accuracy-evolution-lipizzaner_2020-05-17_08-21.csv'
fid_log_file = '/home/jamal/Documents/Research/sourcecode/evaluate-lipizzaneer-output/data/evolution/mnist-fid-evolution-lipizzaner_2020-05-17_08-21.csv'

createa_video(acc_label_log_file, acc_log_file, fid_log_file, client_id, step=1)
#show_evolution_fid_vs_acc(acc_log_file, fid_log_file)
#show_evolution(acc_log_file)

#createa_video(acc_label_log_file, client_id)

#show_evolution('/home/jamal/Documents/Research/sourcecode/evaluate-lipizzaneer-output/data/output/evolution/mnist-training_accuracy-evolution-lipizzaner_2020-05-16_14-46.csv')