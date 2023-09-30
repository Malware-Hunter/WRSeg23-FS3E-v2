import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from termcolor import colored, cprint
import argparse
import sys
import os
import logging

def add_arguments(parser):
    parser.add_argument(
        '--ignore-mi', help = 'Ignore Mutual Information before calculating weights',
        action = 'store_true')
    parser.add_argument( '-t', '--weights-threshold', type = float, default = 0.2,
        help = 'Only features with weight greater than or equal to this value will be selected. Default: 0.2')

def select_features_with_mi(X, y):
    mi_model = mutual_info_regression(X, y, random_state = 1)
    mi_scores = pd.DataFrame({'score': mi_model}, index = list(X.columns))
    selected_features = list(mi_scores[mi_scores['score'] == 0.0].index)
    return X[selected_features]

def get_weights_from_classifiers(X, y, classifiers):
    weights_list = list()
    for classifier in classifiers.values():
        constructor = classifier['constructor']
        constructor.fit(X, y)
        weights_attribute = classifier['importance_metric']
        weights = getattr(constructor, weights_attribute)
        # ensuring weights is not a list of list
        weights = weights if isinstance(weights[0], np.float64) else weights[0]
        weights_list.append(list(weights))
    return weights_list

def get_normalized_weights_average(weights_list):
    normalized_weights_list = list()
    for weights in weights_list:
        max_value = max(weights)
        min_value = min(weights)
        normalized_weights = list()
        denominator = max_value - min_value
        denominator_is_zero = False if max_value != min_value else True
        for weight in weights:
            value = 0.5 if denominator_is_zero else (weight - min_value) / denominator
            normalized_weights.append(value)
        normalized_weights_list.append(normalized_weights)
    return np.average(normalized_weights_list, axis = 0)

def run_jowmdroid(X, y, weight_classifiers):
    global logger_jowmdroid
    logger_jowmdroid.info('Calculating Weights ...')
    weights_list = get_weights_from_classifiers(X, y, weight_classifiers)
    initial_weights = get_normalized_weights_average(weights_list)
    return initial_weights

def run(args, ds):
#if __name__=="__main__":
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_jowmdroid
    logger_jowmdroid = logging.getLogger('JOWMDroid')
    logger_jowmdroid.setLevel(logging.INFO)

    parsed_args = args
    try:
        logger_jowmdroid.info(f'Loading Dataset {ds}')
        dataset = pd.read_csv(ds)
    except BaseException as e:
        msg = colored(f'Exception: {e}', 'red')
        logger_jowmdroid.error(msg)
        exit(1)

    X = dataset.drop(parsed_args.class_column, axis = 1)
    y = dataset[parsed_args.class_column]
    logger_jowmdroid.info(f'Number of Initial Features: {X.shape[1]}')

    if not parsed_args.ignore_mi:
        logger_jowmdroid.info('Calculating Mutual Information ...')
        X = select_features_with_mi(X, y)
        logger_jowmdroid.info(f'Number of Features After MI: {X.shape[1]}')

    weight_classifiers = {
            'SVM': {'constructor': SVC(kernel = 'linear', random_state = 1), 'importance_metric': 'coef_'},
            'RF': {'constructor': RandomForestClassifier(random_state = 1), 'importance_metric': 'feature_importances_'},
            'LR': {'constructor': LogisticRegression(max_iter = 500, random_state = 1), 'importance_metric': 'coef_'}}
    weights = run_jowmdroid(X, y, weight_classifiers)
    features = list(X.columns)
    feature_weights = pd.DataFrame({'weight': weights}, index = features)
    selected_features = list(feature_weights[feature_weights['weight'] >= parsed_args.weights_threshold].index)
    X = X[selected_features]
    logger_jowmdroid.info(f'Number of Selected Features: {X.shape[1]}')
    selected_features.append(parsed_args.class_column)
    dataset = dataset[selected_features]
    output_dir = parsed_args.output
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f'jomwdroid_{os.path.basename(ds)}')
    logger_jowmdroid.info(f'Saving the Reduced Dataset in {output_file}')
    dataset.to_csv(output_file, index = False)
