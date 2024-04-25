import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import sys
from random import choice
from argparse import ArgumentParser
from methods.utils import get_base_parser, get_dataset, get_X_y, get_filename
import logging

def correlation_phase(X, y, k, method, methods):
    global logger_sigapi
    feature_scores = methods[method]['function'](X, y, k)
    new_X = X[list(feature_scores['features'])]

    correlation = new_X.corr()

    model_RF=RandomForestClassifier()
    model_RF.fit(new_X,y)

    feats = {}
    for feature, importance in zip(new_X.columns, model_RF.feature_importances_):
        feats[feature] = importance

    to_drop = set()

    for index in correlation.index:
        for column in correlation.columns:
            if index != column and correlation.loc[index, column] > 0.85:
               ft = column if feats[column] <= feats[index] else index
               to_drop.add(ft)

    logger_sigapi.info("qtd de features removidas: %s" % len(to_drop))

    new_X = new_X.drop(columns = to_drop)
    new_X['class'] = y
    return new_X

def add_arguments(parser):
    parser.add_argument('--threshold-sigapi', type = float, default = 0.03,
        help = 'Threshold for the difference between metrics at each increment on the number of features. When all metrics are less than it, the selection phase finishes. Default: 0.03')
    parser.add_argument( '-f', '--initial-n-features', type = int, default = 1,
        help = 'Initial number of features. Default: 1')
    parser.add_argument( '--increment-sigapi', type = int, default = 1,
        help = 'Value to increment the initial number of features. Default: 1')

def get_moving_average(data, window_size=5):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    return (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

def get_minimal_range_suggestion(df, t=0.001, window_size=5):
    moving_averages = np.array([get_moving_average(np.array(df)[:, i], window_size) for i in range(df.shape[1])]).T
    gradients = np.gradient(moving_averages, axis=0)
    diffs = gradients[1:] - gradients[:-1]

    for i in range(len(diffs) - 1, 1, -1):
        if(any([diff > t for diff in diffs[i]])):
            return int(df.index[i])
    return -1

"""# **Função Incremento** """

def calculateMutualInformationGain(features, target, k):
    feature_names = features.columns
    mutualInformationGain = mutual_info_classif(features, target, random_state = 0)
    data = {"features": feature_names, "score": mutualInformationGain}
    df = pd.DataFrame(data)
    df = df.sort_values(by=['score'], ascending=False)
    return df[:k]

def calculateRandomForestClassifier(features, target,k):
    feature_names= np.array(X.columns.values.tolist())
    test = RandomForestClassifier(random_state = 0)
    test = test.fit(X,y)
    model = SelectFromModel(test,max_features = k, prefit = True)
    model.get_support()
    best_features = feature_names[model.get_support()]
    best_score = test.feature_importances_[model.get_support()]
    df = pd.DataFrame(list(zip(best_features,best_score)),columns=['features','score']).sort_values(by= ['score'],ascending= False)
    return df

def calculateExtraTreesClassifier(features, target, k):
    feature_names= np.array(X.columns.values.tolist())
    test = ExtraTreesClassifier(random_state = 0)
    test = test.fit(X,y)
    model = SelectFromModel(test,max_features = k, prefit = True)
    model.get_support()
    best_features = feature_names[model.get_support()]
    best_score = test.feature_importances_[model.get_support()]
    df = pd.DataFrame(list(zip(best_features,best_score)),columns=['features','score']).sort_values(by= ['score'],ascending= False)
    return df

def calculateRFERandomForestClassifier(features, target, k):
    feature_names= np.array(X.columns.values.tolist())
    rfe = RFE(estimator = RandomForestClassifier(), n_features_to_select = k)
    model = rfe.fit(X,y)
    best_features = feature_names[model.support_]
    best_scores = rfe.estimator_.feature_importances_
    df = pd.DataFrame(list(zip(best_features, best_scores)), columns = ["features", "score"]).sort_values(by = ['score'], ascending=False)
    return df

def calculateRFEGradientBoostingClassifier(features, target,k):
    feature_names= np.array(X.columns.values.tolist())
    rfe = RFE(estimator = GradientBoostingClassifier(), n_features_to_select = k)
    model = rfe.fit(X,y)
    best_features = feature_names[model.support_]
    best_scores = rfe.estimator_.feature_importances_
    df = pd.DataFrame(list(zip(best_features, best_scores)), columns = ["features", "score"]).sort_values(by = ['score'], ascending=False)
    return df


def calculateSelectKBest(features, target,k):
    feature_names= np.array(features.columns.values.tolist())
    chi2_selector= SelectKBest(score_func = chi2, k= k)
    chi2_selector.fit(features,target)
    chi2_scores = pd.DataFrame(list(zip(feature_names,chi2_selector.scores_)),columns= ['features','score'])
    df = pd.DataFrame(list(zip(feature_names,chi2_selector.scores_)),columns= ['features','score']).sort_values(by = ['score'], ascending=False)
    return df[:k]


def calculateMetricas(new_X,y):
    new_X_train,new_X_test, y_train, y_test = train_test_split(new_X, y, test_size = 0.3,random_state = 0)

    teste = RandomForestClassifier()
    teste.fit(new_X_train, y_train)
    resultado_teste = teste.predict(new_X_test)

    acuracia = accuracy_score(y_test, resultado_teste)
    precision = precision_score(y_test, resultado_teste, zero_division = 0)
    recall = recall_score(y_test, resultado_teste, zero_division = 0)
    f1 = f1_score(y_test, resultado_teste, zero_division = 0)

    metricas = [acuracia,precision,recall,f1]
    return metricas

methods = { 'mutualInformation': { 'function': calculateMutualInformationGain, 'results': [[0,0,0,0,0]], 'is_stable': False },
    'selectRandom': { 'function': calculateRandomForestClassifier, 'results': [[0,0,0,0,0]], 'is_stable': False },
    'selectExtra': { 'function': calculateExtraTreesClassifier, 'results': [[0,0,0,0,0]], 'is_stable': False },
    'RFERandom': { 'function': calculateRFERandomForestClassifier, 'results': [[0,0,0,0,0]], 'is_stable': False },
    'RFEGradient': { 'function': calculateRFEGradientBoostingClassifier, 'results': [[0,0,0,0,0]], 'is_stable': False },
    'selectKBest': { 'function': calculateSelectKBest, 'results': [[0,0,0,0,0]], 'is_stable': False }
}

def is_method_stable(previous_metrics, current_metrics, t=0.03):
    differences = abs(current_metrics - previous_metrics)
    if(all(differences < t)):
        return True
    return False

def selection_phase(X, y, methods, total_features,num_features=1, increment=1):
    global logger_sigapi
    has_found_stable_method = False
    best_stable_method = None
    best_metric_value = 0
    while num_features < (total_features + increment) and not has_found_stable_method:
        k = total_features if num_features > total_features else num_features
        logger_sigapi.info("qtd de features: %s" % k)

        for method_name in methods.keys():
            feature_scores = methods[method_name]['function'](X, y, k)
            new_X = X[list(feature_scores['features'])]
            metrics =  calculateMetricas(new_X,y)
            methods[method_name]['results'] = np.append(methods[method_name]['results'],[[k,metrics[0],metrics[1],metrics[2],metrics[3]]],axis=0)
            previous_metrics = methods[method_name]['results'][-2][1:]
            current_metrics = methods[method_name]['results'][-1][1:]

            # A primeira expressão booleana (len(...) > 2) é para evitar comparar as métricas calculadas contra o vetor [0,0,0,0],
            # que é definido inicialmente no dicionário "methods"
            if(len(methods[method_name]['results']) > 2 and is_method_stable(previous_metrics, current_metrics, parsed_args.threshold)):
                has_found_stable_method = True
                accuracy = current_metrics[0]
                if(accuracy > best_metric_value):
                    best_metric_value = accuracy
                    best_stable_method = method_name
        num_features += increment

    if(not has_found_stable_method):
        best_stable_method = choice(list(methods.keys()))

    k = int(methods[best_stable_method]["results"][-1][0])
    return best_stable_method, k

def run(args,ds):
    import os
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_sigapi
    logger_sigapi = logging.getLogger('SigAPI')
    logger_sigapi.setLevel(logging.INFO)

    global parsed_args
    parsed_args = args
    dataset = pd.read_csv(ds)
    global X, y
    X = dataset.drop(parsed_args.class_column, axis=1)
    y = dataset[parsed_args.class_column]
    total_features = len(X.columns)
    if(parsed_args.initial_n_features > total_features):
        logger_sigapi.error(f"--initial-n-features ({parsed_args.initial_n_features}) maior que a qtd de features do dataset ({total_features})")
        exit(1)
    logger_sigapi.info("INÍCIO DA SELEÇÃO DE FEATURES")
    best_stable_method, lower_bound = selection_phase(X, y ,methods, total_features,num_features=parsed_args.initial_n_features, increment=parsed_args.increment)
    logger_sigapi.info("SUGESTÃO DE LIMITE PARA A FASE DE CORRELAÇÃO")
    logger_sigapi.info(f'Menor limite inferior encontrado: {best_stable_method}, {lower_bound}')
    logger_sigapi.info("INICIO DA CORRELAÇÃO")
    new_X = correlation_phase(X, y, lower_bound, best_stable_method, methods)
    output_dir = args.output
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f'sigapi_{os.path.basename(ds)}')

    new_X.to_csv(output_file, index=False)
    logger_sigapi.info("Dataset final criado")
