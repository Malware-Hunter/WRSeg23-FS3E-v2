from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import sys
import os

import pandas as pd
import numpy as np
from scipy.stats import entropy

def add_arguments(parser):
    parser = parser.add_argument_group("Arguments for FSDroid")
    parser.add_argument("--k", type=int, help="Number of best caracteristics", required = False, default = 10)

def information_gain(feature, target):
    """
    Calcula o ganho de informação de uma característica.
    """
    # Calcula a entropia do conjunto de dados
    total_entropy = entropy(target.value_counts(normalize=True), base=2)

    # Calcula a entropia para cada valor da característica
    value_entropies = []
    for value in feature.unique():
        value_target = target[feature == value]
        value_entropy = entropy(value_target.value_counts(normalize=True), base=2)
        value_weight = len(value_target) / len(target)
        value_entropies.append(value_weight * value_entropy)

    # Calcula o ganho de informação
    gain = total_entropy - sum(value_entropies)

    return gain

def gain_ratio(feature, target):
    """
    Calcula o ganho de informação normalizado pelo split information de uma característica.
    """
    # Calcula o ganho de informação
    gain = information_gain(feature, target)

    # Calcula o split information
    value_ratios = []
    for value in feature.unique():
        value_ratio = len(target[feature == value]) / len(target)
        value_ratios.append(value_ratio)

    split_info = -sum(value_ratios * np.log2(value_ratios))

    # Calcula o ganho de informação normalizado pelo split information
    if split_info == 0:
        ratio = 0
    else:
        ratio = gain / split_info

    return ratio

def select_features_gr(data, target, k):
    """
    Seleciona as k melhores características de um conjunto de dados usando Information Gain e Gain Ratio.
    """
    # Calcula o ganho de informação e o gain ratio para cada característica
    ig = {}
    gr = {}
    for col in data.columns:
        ig[col] = information_gain(data[col], target)
        gr[col] = gain_ratio(data[col], target)

    # Seleciona as k melhores características com base no gain ratio
    features = sorted(gr, key=gr.get, reverse=True)[:k]

    return features

def calculate_error(data, feature, target, value):
    """
    Calcula o número de erros de classificação de uma característica para um valor específico.
    """
    # Cria uma tabela de contingência
    contingency_table = pd.crosstab(data[feature] == value, target)

    # Calcula o número de exemplos com o valor específico da característica
    total = contingency_table.sum().sum()

    # Calcula o número de erros de classificação
    correct = max(contingency_table.sum(axis=0))
    errors = total - correct

    return errors

def select_features_or(data, target, k=None):
    """
    Seleciona as melhores características de um conjunto de dados usando o algoritmo OneR.
    """
    # Inicializa as melhores características e o menor número de erros
    best_features = None
    min_errors = float('inf')

    # Percorre todas as características do conjunto de dados
    for feature in data.columns:
        # Cria uma tabela de contingência para a característica
        contingency_table = pd.crosstab(data[feature], target)

        # Calcula o número de erros de classificação para cada valor da característica
        errors = 0
        for value in contingency_table.index:
            errors += calculate_error(data, feature, target, value)

        # Atualiza as melhores características se o número de erros for menor
        if errors < min_errors:
            best_features = [feature]
            min_errors = errors
        elif errors == min_errors:
            best_features.append(feature)

    # Ordena as características pelo número de erros e retorna as k características com o menor número de erros
    if k is None:
        return best_features
    else:
        return sorted(best_features, key=lambda feature: calculate_error(data, feature, target, True))[:k]


def pca_features(data, n_components, feature_names=None):
    """
    Reduz a dimensionalidade de um conjunto de dados usando o algoritmo Principal Component Analysis (PCA).
    """
    # Cria um objeto PCA com o número de componentes especificado
    pca = PCA(n_components=n_components)

    # Ajusta o PCA ao conjunto de dados e transforma os dados
    data_transformed = pca.fit_transform(data)

    # Obtém os nomes correspondentes às novas features
    if feature_names is None:
        columns = ['PC' + str(i+1) for i in range(n_components)]
    else:
        columns = [feature_names[i] for i in range(n_components)]

    # Cria um DataFrame com as novas features
    principal_components = pd.DataFrame(data_transformed, columns=columns)

    # Retorna os nomes das novas features
    return principal_components.columns.tolist()


def logistic_regression_features(data, target, k=None):
    """
    Classifica os dados usando a análise de regressão logística e retorna as características mais importantes.

    Parâmetros:
    data: DataFrame contendo as características do conjunto de dados
    target: Series contendo o target do conjunto de dados
    k: número inteiro de características mais importantes a serem retornadas. Se None, retorna todas as características.

    Retorno:
    Lista contendo os nomes das características mais importantes, ordenadas pela importância.
    """
    # Cria um objeto de regressão logística
    clf = LogisticRegression(random_state=0, max_iter=100000)

    # Ajusta a regressão logística aos dados e ao target
    clf.fit(data, target)

    # Obtém as características mais importantes
    features_importance = {}
    for i, feature in enumerate(data.columns):
        features_importance[feature] = clf.coef_[0][i]

    # Ordena as características pela importância
    features_sorted = sorted(features_importance.items(), key=lambda x: x[1], reverse=True)

    # Retorna apenas os nomes das características ordenadas pela importância, até o número k se fornecido
    if k is not None:
        return [feature[0] for feature in features_sorted[:k]]
    else:
        return [feature[0] for feature in features_sorted]

def information_gain_features(data, target, n_features):
    """
    Seleciona as melhores características com base no critério de ganho de informação.
    """
    # Calcula a pontuação de ganho de informação para cada característica
    info_gain = mutual_info_classif(data, target, discrete_features='auto', n_neighbors=3, copy=True, random_state=0)

    # Obtém os índices das características com maior pontuação de ganho de informação
    top_feature_indices = np.argsort(info_gain)[::-1][:n_features]

    # Retorna as características selecionadas
    return list(data.columns[top_feature_indices])

def chi2_features(data, target, n_features):
    """
    Seleciona as melhores características com base no teste do qui-quadrado (chi²).
    """
    # Aplica o teste do qui-quadrado às características e ao target
    chi2_scores, _ = chi2(data, target)

    # Obtém os índices das características com maior pontuação do qui-quadrado
    top_feature_indices = np.argsort(chi2_scores)[::-1][:n_features]

    # Retorna as características selecionadas
    return list(data.columns[top_feature_indices])

def check_in_list(list, features):
    for feature in features:
        if feature not in list:
            list.append(feature)

    return list


def run(args,ds):
    best = []

    parsed_args = args

    dataset = pd.read_csv(ds)
    X = dataset.drop(parsed_args.class_column, axis=1)
    y = dataset[parsed_args.class_column]

    #test Information-gain
    selected_features = information_gain_features(X, y, parsed_args.k)
    best = check_in_list(best, selected_features)

    #test Chi-Squared test
    selected_features = chi2_features(X, y, parsed_args.k)
    best = check_in_list(best, selected_features)

    # #test Gain-ratio
    selected_features = select_features_gr(X, y, parsed_args.k)
    best = check_in_list(best, selected_features)

    # test OneR feature selection
    selected_features = select_features_or(X, y, parsed_args.k)
    best = check_in_list(best, selected_features)

    #test PCA
    selected_features = pca_features(X, parsed_args.k, X.columns)
    best = check_in_list(best, selected_features)

    #test Logistic Regression
    selected_features = logistic_regression_features(X, y, parsed_args.k)
    best = check_in_list(best, selected_features)

    new_ds = dataset.copy()
    for col in new_ds.columns:
        if col not in best:
            new_ds = new_ds.drop(col, axis=1)
    new_ds[parsed_args.class_column] = dataset[parsed_args.class_column]
    output_dir = parsed_args.output
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f'fsdroid_{os.path.basename(ds)}')
    new_ds.to_csv(output_file, index = False)
