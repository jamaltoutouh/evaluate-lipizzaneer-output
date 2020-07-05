from pathlib import Path
import string
import re
import json
import pandas as pd
from datetime import datetime
import numpy as np
import math


from collections import OrderedDict
from datetime import date

import os
import glob
import sys
from scipy.stats import shapiro

output_folder = '../../data/output/'
# output_folder = '/home/jamaltoutouh/semi-supervised/lipizzaner-gan/src/output/'
# output_folder = '../data/output-medium/'
data_folder = '../../data/'
dataset = 'covid'

def get_all_master_log_files():
    return [filepath for filepath in glob.iglob(output_folder + 'log/*.log')]

def get_distributed_log_files_given_master_log(master_log_filename):
    return [filepath for filepath in glob.iglob(output_folder +'lipizzaner_gan/distributed/' + dataset + '/*/*/' + master_log_filename)]

def get_independent_run_params(file_name):
    parameters = None
    for line in open(file_name, 'r'):
        if 'Parameters: ' in line:
            splitted_data = re.split("Parameters: ", line)
            parameters = json.loads(str(splitted_data[1]).replace("\'", "\"").replace("True", "true").replace("False", "false").replace("None", "null"))
    return parameters


def get_loss_type(parameters):
    return parameters['network']['loss'] if not (parameters is None) else parameters

def get_client_id(parameters):
    return parameters['general']['distribution']['client_id'] if not (parameters is None) else parameters

def get_iterations(parameters):
    return parameters['trainer']['n_iterations'] if not (parameters is None) else parameters

def get_batch_size(parameters):
    return parameters['dataloader']['batch_size'] if not (parameters is None) else parameters

def get_label_rate(parameters):
    return parameters['dataloader']['label_rate'] if not (parameters is None) else parameters

def split_equal(data):
    container = data.split("=")
    return container[0], container[1]

def get_metric_value(analized_data, metric='fid'):
    if metric == 'fid':
        return float(split_equal(analized_data[5])[1]) #score
    elif metric == 'gen_loss':
        return float(split_equal(analized_data[1])[1])
    elif metric == 'disc_loss':
        return float(split_equal(analized_data[2])[1])
    elif metric == 'gen_lr':
        return float(split_equal(analized_data[3])[1])
    elif metric == 'disc_lr':
        return float(split_equal(analized_data[4])[1])
    elif metric == 'training_accuracy':
        print(analized_data)

def get_evolution_one_client(client_log, metric='fid'):
    data = []
    f = open(client_log, 'r')
    line = f.readline()
    while line:
        if metric =='training_accuracy' and 'Label Prediction Accuracy' in line:
            splitted_data = re.split(" |,|%", line)
            data.append(float(splitted_data[-3]))
        elif not metric in ['per_label_accuracy', 'training_accuracy'] and 'Iteration=' in line:
            splitted_data = re.split("- |,|%", line)
            analized_data = splitted_data[3:9]
            data.append(get_metric_value(analized_data, metric))
        elif metric == 'per_label_accuracy' and \
                'Label, Number of Labeled Data points, Classification Rate for this label' in line:
            data_row = []
            label = 0
            line = f.readline()
            while label <= 9:
                data_row.append(float(line.split(',')[-1]))
                label += 1
                line = f.readline()
            data.append(data_row)
        line = f.readline()

    if len(data)>0:
        return data
    else:
        None


def get_evolution_distributed(master_log_filename, metric='fid'):
    distributed_log_files = get_distributed_log_files_given_master_log(master_log_filename)
    data_set = dict()
    n_iterations = get_iterations(get_independent_run_params(distributed_log_files[0]))

    for distributed_log_file in distributed_log_files:
        client_id = get_client_id(get_independent_run_params(distributed_log_file))
        if metric != 'per_label_accuracy':
            data = get_evolution_one_client(distributed_log_file, metric)
            if not data is None and len(data) == n_iterations:
                data_set['{}'.format(client_id)] = data
        else:
            data = get_evolution_one_client(distributed_log_file, metric)
            if not data is None and len(data) == n_iterations:
                data = np.array(data).T
                for i in range(len(data)):
                    dict_label = '{} - {}'.format(client_id, i)
                    data_set[dict_label] = data[i]

    pd.DataFrame(data_set).to_csv(data_folder + '/evolution/' + dataset + '-' + metric + '-evolution-' +
                                  master_log_filename[:-4] + '.csv', index=False)


def get_evolution(metric='fid'):
    Path(data_folder + '/evolution/').mkdir(parents=True, exist_ok=True)
    data_set = []
    processed_independent_runs = 0
    for master_log in get_all_master_log_files():
        master_log_filename = master_log.split('/')[-1]
        distributed_log_files = get_distributed_log_files_given_master_log(master_log_filename)
        if len(distributed_log_files) != 0:
            get_evolution_distributed(master_log_filename, metric)
            processed_independent_runs += 1

    print('Processed {} independent runs. '.format(processed_independent_runs))




import sys

metrics = ['fid', 'gen_loss', 'disc_loss', 'gen_lr', 'disc_lr', 'per_label_accuracy', 'training_accuracy']


if len(sys.argv)<2:
    print('We need an argument for the metric to get')
    print('Metrics: {}'.format(metrics))
else:
    print('Creating evolution files of: {}'.format(metrics[int(sys.argv[1])]))
    get_evolution(metrics[int(sys.argv[1])])
