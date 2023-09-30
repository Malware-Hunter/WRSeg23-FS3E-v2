import math
import sys
import pandas as pd
from argparse import ArgumentParser
import os
import logging

def add_arguments(parser):
    parser = parser.add_argument_group("Arguments for SemiDroid")

def calculate_entropy(data, args):
    # Calcula a entropia de um conjunto de dados
    class_counts = data[args.class_column].value_counts()
    total_samples = len(data)
    entropy = 0.0
    for count in class_counts:
        probability = count / total_samples
        entropy -= probability * math.log2(probability)
    return entropy

def calculate_conditional_entropy(data, feature, args):
    # Calcula a entropia condicional de uma característica em relação à classe
    unique_values = data[feature].unique()
    total_samples = len(data)
    conditional_entropy = 0.0
    for value in unique_values:
        subset = data[data[feature] == value]
        subset_entropy = calculate_entropy(subset, args)
        probability = len(subset) / total_samples
        conditional_entropy += probability * subset_entropy
    return conditional_entropy

def calculate_information_gain(data, feature, args):
    # Calcula o Information Gain de uma característica em relação à classe
    entropy = calculate_entropy(data, args)
    conditional_entropy = calculate_conditional_entropy(data, feature, args)
    information_gain = entropy - conditional_entropy
    return information_gain

'''
    Segunda Etapa - Feature Discrimination
    fib representa a frequência do recurso fi em arquivos benignos
    fim representa a frequência do recurso fi em arquivos maliciosos
    Score(fi) = 0 {frequência igual de ocorrência em ambas as classes; sem discriminação}
    Score(fi) ~ 0 {baixa frequência de ocorrência em qualquer uma das classes; pior característica discriminante}
    Score(fi) ~ 1 {alta frequência de ocorrência em qualquer uma das classes; melhor característica discriminativa}
'''

def fib(feature):
    global B
    return len(B[B[feature] != 0])/len(B)

def fim(feature):
    global M
    return len(M[M[feature] != 0])/len(M)

def score(feature):
    fb = fib(feature)
    fm = fim(feature)
    value = 1.0 - (min(fb, fm) / max(fb, fm))
    return value

def get_unique_values(df):
    for column_name in df.columns:
        yield (column_name, df[column_name].unique())

def drop_irrelevant_columns(df):
    # retorna o df sem colunas irrelevantes (colunas com menos de 2 valores possíveis)
    irrelevant_columns = list()
    for column_name, unique_values in get_unique_values(df):
        if len(unique_values) < 2:
            irrelevant_columns.append(column_name)
    return df.drop(columns = irrelevant_columns)

def non_frequent_features(df, args):
    df_len = len(df)
    non_frequent = list()
    X = df.drop(args.class_column, axis = 1)
    for ft in list(X.columns):
        count_nonzero = (X[ft] != 0).sum()
        frequency = count_nonzero/df_len
        if frequency < 0.2:
            non_frequent.append(ft)
    return non_frequent

def features_to_drop(data, th = 0.2):
    max_value = max(data.values())
    min_value = max_value * th
    ft_to_drop = list()
    for ft, value in data.items():
        if value < min_value:
            ft_to_drop.append(ft)
    return ft_to_drop

def run(args, ds):
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_mt
    global B
    global M

    logger_mt = logging.getLogger('MT')
    logger_mt.setLevel(logging.INFO)
    args = args
    try:
        dataset = pd.read_csv(ds)
    except BaseException as e:
        logger_mt.exception(e)
        exit(1)

    # exclusão de permissões comuns
    '''
    if 'INTERNET' in list(dataset.columns):
        dataset = dataset.drop(columns = ['INTERNET'])
    if 'ACCESS_NETWORK_STATE' in list(dataset.columns):
        dataset = dataset.drop(columns = ['ACCESS_NETWORK_STATE'])
    '''
    #dataset = drop_irrelevant_columns(dataset)

    logger_mt.info('Non-Frequent Reduction')
    B = dataset[(dataset[args.class_column] == 0)]
    M = dataset[(dataset[args.class_column] == 1)]
    non_frequent_b_ft = non_frequent_features(B, args)
    non_frequent_m_ft = non_frequent_features(M, args)
    non_frequent_ft = list(set(non_frequent_b_ft).union(set(non_frequent_m_ft)))
    logger_mt.info(f'Non-Frequent Features: {len(non_frequent_ft)}')
    dataset = dataset.drop(columns = non_frequent_ft)

    logger_mt.info('Feature Discrimination')
    B = dataset[(dataset[args.class_column] == 0)]
    M = dataset[(dataset[args.class_column] == 1)]
    ft_score = dict()
    for ft in list(dataset.columns):
        if ft != args.class_column:
            ft_score[ft] = score(ft)
    ft_to_drop = features_to_drop(ft_score)
    logger_mt.info(f'Features to Drop: {len(ft_to_drop)}')
    dataset = dataset.drop(columns = ft_to_drop)

    logger_mt.info('Information Gain')
    ft_ig = dict()
    for ft in list(dataset.columns):
        if ft != args.class_column:
            ft_ig[ft] = calculate_information_gain(dataset, ft, args)
    ft_to_drop = features_to_drop(ft_ig)
    logger_mt.info(f'Features to Drop: {len(ft_to_drop)}')
    dataset = dataset.drop(columns = ft_to_drop)

    output_dir = args.output
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f'mt_{os.path.basename(ds)}')
    print(f'Selected Features: {dataset.shape[1] - 1}')
    dataset.to_csv(output_file, index = False)
