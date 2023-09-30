import pandas as pd
import numpy as np
import sys
import argparse
from sklearn.feature_selection import chi2
from scipy.stats import entropy
from collections import Counter
from math import log
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

"""# Dataset"""
def add_arguments(parser):
    parser = parser.add_argument_group("Arguments for SemiDroid")
    parser.add_argument("-p", "--percent", type=float,
                        help="Percentage of Selected Features. Default: 0.1", default=0.1)


"""# **Funções Auxiliares**"""
def _Ex_a_v_(Ex, a, v, nan = True):
    if nan:
        return [x for x, t in zip(Ex, a) if (isinstance(t, float)
                                        and isinstance(v, float)
                                        and math.isnan(t)
                                        and math.isnan(v))
                                        or t == v]
    else:
        return [x for x, t in zip(Ex, a) if t == v]

def intrinsic_value(Ex, a, nan = True):
    sum_v = 0
    for v in set(a):
        Ex_a_v = _Ex_a_v_(Ex, a, v, nan)
        if len(Ex_a_v) != 0:
            return (len(Ex_a_v) / len(Ex)) * (log(len(Ex_a_v) / len(Ex), 2))

def get_subset(X, y, data, args):
    data.sort_values(by = ['score'], ascending = False, inplace = True)
    #n = int(log(len(feature_names), 2) + 1)
    n = int(X.shape[1] * args.percent + 1)
    selected_features = data.feature[:n]
    new_df = X.loc[:,selected_features]
    new_df[args.class_column] = y
    return new_df

"""# **Funções de Ranking**"""
#Função ChiSquared
def chi_squared(X, y, args):
    feature_names = X.columns
    chi2score, pval = chi2(X, y)
    data = pd.DataFrame({'feature':feature_names, 'score':chi2score})
    return get_subset(X, y, data, args)

#Função Information Gain
def i_gain(Ex, a, nan = True):
    H_Ex = entropy(list(Counter(Ex).values()))
    sum_v = 0
    for v in set(a):
        Ex_a_v = _Ex_a_v_(Ex, a, v, nan)
        sum_v += (len(Ex_a_v) / len(Ex)) *\
            (entropy(list(Counter(Ex_a_v).values())))
    result = H_Ex - sum_v
    return result

def info_gain(X, y, args):
    feature_names = X.columns
    data = pd.DataFrame(columns = ['feature', 'score'])
    for i in X.columns:
        data = pd.concat([data, pd.DataFrame([[i, i_gain(i, X)]], columns = ['feature', 'score'])])
    return get_subset(X, y, data, args)

#Função Gain Ratio
def g_ratio(Ex, a, nan = True):
    result = i_gain(Ex, a) /(-intrinsic_value(Ex, a))
    return result

def gain_ratio(X, y, args):
    feature_names = X.columns
    data = pd.DataFrame(columns = ['feature', 'score'])
    for i in X.columns:
        data = pd.concat([data, pd.DataFrame([[i, g_ratio(i, X)]], columns = ['feature', 'score'])])
    return get_subset(X, y, data, args)

#Função OneR
def one_r(X, y, args):
    feature_names = X.columns
    result_list = list()
    summary_dict = {}
    for feature_name in feature_names:
        summary_dict[feature_name] = {}
        join_data = pd.concat([X[feature_name], y], axis=1)
        freq_table = pd.crosstab(join_data[feature_name], join_data[y.name])
        summary = freq_table.idxmax(axis = 1)
        summary_dict[feature_name] = dict(summary)
        counts = 0
        for idx, row in join_data.iterrows():
            if row[y.name] == summary[row[feature_name]]:
                counts += 1
        accuracy = counts/len(y)
        result_feature = {'feature': feature_name, 'score':accuracy}
        result_list.append(result_feature)
    data = pd.DataFrame(result_list)
    return get_subset(X, y, data, args)

#Função Principal Component Analysis
def pca_analysis(X, y, args):
    values = X.values
    x = StandardScaler().fit_transform(X)
    principal = PCA()
    principal.fit_transform(x)
    eigenvalues = principal.explained_variance_
    feature_names = X.columns
    data = pd.DataFrame({'feature':feature_names, 'score':eigenvalues})
    data = data[data['score'] > 1.0]
    return get_subset(X, y, data, args)

#Função Logistic Regression
def logistic_regression(X, y, args):
    feature_names = X.columns
    model = LogisticRegression().fit(X, y)
    score = model.coef_[0]
    data = pd.DataFrame({'feature':feature_names, 'score':score})
    data = data[data['score'] > 0.05]
    return get_subset(X, y, data, args)

def random_forest(dataset, args):
    X = dataset.drop(args.class_column, axis = 1)
    y = dataset[args.class_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    rf = RandomForestClassifier(n_estimators = 100, max_depth = 10)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

"""# **MAIN**"""
def run(args,ds):
    args = args
    try:
        dataset = pd.read_csv(ds)
    except BaseException as e:
        print(e)
        exit(1)
    X = dataset.drop(columns = args.class_column)
    y = dataset[args.class_column]

    best_accuracy = 0.0
    best_subset = pd.DataFrame()
    function_list = ['chi_squared', 'info_gain', 'gain_ratio', 'one_r', 'logistic_regression', 'pca_analysis']
    for function in function_list:
        function_real = globals()[function]
        print(f'>>> Testing Dataset With {function} <<<')
        subset = function_real(X, y, args)
        acc = random_forest(subset, args)
        print(f'Subset Accuracy: {acc}')
        if acc > best_accuracy:
            best_accuracy = acc
            best_subset = subset
    output_dir = args.output
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f'semidroid_{os.path.basename(ds)}')
    best_subset.to_csv(output_file, index = False)
