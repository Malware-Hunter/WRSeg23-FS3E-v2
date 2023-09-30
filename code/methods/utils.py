import argparse
import pandas as pd
import os

def get_base_parser():
    """
    Retorna um ArgumentParser com os parâmetros comuns entre os métodos implementados. 
    Para incluir os parâmetros deste parser em outro, passe-o para o outro parser da seguinte forma:
    ```
    from argparse import ArgumentParser
    from utils import get_base_parser

    base_parser = get_base_parser()
    other_parser = ArgumentParser(parents=[base_parser])

    # Adicione os parâmetros específicos do outro parser normalmente:
    other_parser.add_argument("-f", help="Lista de features", ...)
    other_parser.add_argument("-k", help="Qtd de folds na validação cruzada", ...)
    other_parser.add_argument(...)
    ```
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument( '-d', '--dataset', type = str, required = True,
        help = 'Dataset (csv file). It should be already preprocessed.')
    parser.add_argument( '--sep', metavar = 'SEPARATOR', type = str, default = ',',
        help = 'Dataset feature separator. Default: ","')
    parser.add_argument('-c', '--class-column', type = str, default="class", metavar = 'CLASS_COLUMN', 
        help = 'Name of the class column. Default: "class"')
    parser.add_argument('-n', '--n-samples', type=int,
        help = 'Use a subset of n samples from the dataset. By default, all samples are used.')
    parser.add_argument('-o', '--output-file', metavar = 'OUTPUT_FILE', type = str, default = 'results.csv', 
        help = 'Output file name. Default: results.csv')
    parser.add_argument('--output-prefix', metavar = 'OUTPUT_PREFIX', default = '', 
        help = 'Prefix of output file name. Defaults to an empty string')
    return parser

def get_dataset(parsed_args):
    dataset = pd.read_csv(parsed_args.dataset, sep=parsed_args.sep)
    n_samples = parsed_args.n_samples
    if(n_samples):
        if(n_samples <= 0 or n_samples > dataset.shape[0]):
            raise Exception(f"Expected n_samples to be in range (0, {dataset.shape[0]}], but got {n_samples}")
        dataset = dataset.sample(n=n_samples, random_state=1, ignore_index=True)
    return dataset

def get_X_y(parsed_args, dataset):
    if(parsed_args.class_column not in dataset.columns):
        raise Exception(f'Expected dataset {parsed_args.dataset} to have a class column named "{parsed_args.class_column}"')
    X = dataset.drop(columns = parsed_args.class_column)
    y = dataset[parsed_args.class_column]
    return X, y

def get_filename(output_file, prefix='', suffix='', extension='.csv'):
    names = [prefix, output_file.replace(extension, ''), suffix]
    return '_'.join(filter(None, names)) + extension

def balanced_dataset(dataset):
    B = dataset[(dataset['class'] == 0)]
    M = dataset[(dataset['class'] == 1)]

    lenB = len(B)
    lenM = len(M)
    b_dataset = None
    if lenB > lenM:
        random_select = B.sample(n = lenM, random_state = 0)
        b_dataset = pd.concat([random_select, M], ignore_index = True)
    else:
        random_select = M.sample(n = lenB, random_state = 0)
        b_dataset = pd.concat([B, random_select], ignore_index = True)
    #b_dataset.to_csv("balanced_" + args.dataset, index = False)
    return b_dataset

def check_directory(root_path, dir_name):
    dir_path = os.path.join(root_path, dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_name)