from sklearn.feature_selection import chi2, f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from argparse import ArgumentParser
import sys
from methods.utils import get_base_parser, get_dataset, get_X_y, get_filename
import logging

heuristic_metrics = ['precision', 'accuracy', 'recall', 'f-measure']

def add_arguments(parser):
    parser = parser.add_argument_group("Arguments for RFG")
    parser.add_argument("--folds", type=int, help="Number of folds to use in k-fold cross validation", required = False, default = 10)
    parser.add_argument("--list", type=int, help="List of number of features to select", required = False, nargs='*')
    parser.add_argument("--samples", type=int, help="Use a subset of n samples from the dataset. RFG uses the whole dataset by default.", required = False)
    parser.add_argument("--increment", type=int, help="Increment. Default: 20", required = False, default = 20)
    parser.add_argument("--feature_only", type=bool, help="Feature Selection Only", required = False, default = False)
    parser.add_argument(
        '--rfg-threshold',
        help = 'Threshold to choose the best dataset of selected features based on --heuristic-metric. Default: 0.95.',
        type = float,
        default = 0.95)
    parser.add_argument(
        '-s', '--decrement-step',
        help = "If the heuristic couldn't find't the dataset of features selected, try again decreasing the threshold by this amount. Default: 0.05.",
        type = float,
        default = 0.05)
    parser.add_argument(
        '-m', '--heuristic-metric',
        help = f"Metric to base the choice of the best dataset of selected features. Options: {','.join(heuristic_metrics)}. Default: 'recall'.",
        choices=heuristic_metrics,
        default = 'recall')

def run_experiment(X, y, classifiers, is_feature_only = False,
                   score_functions=[chi2, f_classif],
                   folds=10,
                   k_increment=20,
                   k_list=[]):
    """
    Esta função implementa um experimento de classificação binária usando validação cruzada e seleção de características.
    Os "classifiers" devem implementar as funções "fit" e "predict", como as funções do Scikit-learn.
    Se o parâmetro "k_list" for uma lista não vazia, então ele será usado como a lista das quantidades de características a serem selecionadas.
    """
    global logger_rfg
    results = []
    feature_rankings = {}
    if(len(k_list) > 0):
        k_values = k_list
    else:
        k_values = range(1, X.shape[1], k_increment)
    for k in k_values:
        if(k > X.shape[1]):
            logger_rfg.warning(f"Skipping K = {k}, since it's greater than the number of features available ({X.shape[1]})")
            continue

        logger_rfg.info("K = %s" % k)
        for score_function in score_functions:
            if(k == max(x for x in k_values if x <= X.shape[1])):
                selector = SelectKBest(score_func=score_function, k=k).fit(X, y)
                X_selected = X.iloc[:, selector.get_support(indices=True)].copy()
                feature_scores_sorted = pd.DataFrame(list(zip(X_selected.columns.values.tolist(), selector.scores_)), columns= ['features','score']).sort_values(by = ['score'], ascending=False)
                X_selected_sorted = X_selected.loc[:, list(feature_scores_sorted['features'])]
                X_selected_sorted['class'] = y
                feature_rankings[score_function.__name__] = X_selected_sorted
                if(X_selected.shape[1] == 1):
                    logger_rfg.warning("Nenhuma caracteristica selecionada")
            if(is_feature_only):
                continue
            X_selected = SelectKBest(score_func=score_function, k=k).fit_transform(X, y)
            kf = KFold(n_splits=folds, random_state=256, shuffle=True)
            fold = 0
            for train_index, test_index in kf.split(X_selected):
                X_train, X_test = X_selected[train_index], X_selected[test_index]
                y_train, y_test = y[train_index], y[test_index]

                for classifier_name, classifier in classifiers.items():
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    report = classification_report(y_test, y_pred, output_dict=True, zero_division = 0)
                    results.append({'n_fold': fold,
                                    'k': k,
                                    'score_function':score_function.__name__,
                                    'algorithm': classifier_name,
                                    'accuracy': report['accuracy'],
                                    'precision': report['macro avg']['precision'],
                                    'recall': report['macro avg']['recall'],
                                    'f-measure': report['macro avg']['f1-score']
                                })
                fold += 1

    return pd.DataFrame(results), feature_rankings

def get_best_result(results, threshold=0.95, heuristic_metric='recall', decrement_step=0.05):

    averages = results.groupby(['k','score_function']).mean(numeric_only=True).drop(columns=['n_fold'])
    maximun_score = max(averages.max())
    th = threshold
    while th > 0:
        for k, score_function in averages.index:
            if(averages.loc[(k, score_function)][heuristic_metric] > th * maximun_score):
                return (k, score_function)
        th -= decrement_step

    logger_rfg.error("Não foi possível encontrar o dataset de características selecionadas, tente novamente variando o --heuristic_metric e/ou --threshold")

def get_best_features_dataset(best_result, feature_rankings, class_column):
    k, score_function = best_result
    X = feature_rankings[score_function].drop(columns=[class_column])
    y = feature_rankings[score_function][class_column]
    X_selected = X.iloc[:, :k]
    X_selected = X_selected.join(y)

    return X_selected


def run(args, ds):
    import os
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_rfg
    logger_rfg = logging.getLogger('RFG')
    logger_rfg.setLevel(logging.INFO)

    parsed_args = args
    dataset = pd.read_csv(ds)
    X = dataset.drop(parsed_args.class_column, axis=1)
    y = dataset[parsed_args.class_column]
    k_list = parsed_args.list if parsed_args.list else list()

    classifiers = {
        'RandomForest': RandomForestClassifier(),
    }

    logger_rfg.info("Executando experimento")
    results, feature_rankings = run_experiment(
        X, y,
        classifiers,
        folds = parsed_args.folds,
        k_increment = parsed_args.increment,
        k_list=k_list,
        is_feature_only=parsed_args.feature_only
    )

    logger_rfg.info("Selecionando as melhores caracteristicas")
    output_dir = args.output
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f'rfg_{os.path.basename(ds)}')

    get_best_features_dataset(get_best_result(results, parsed_args.threshold, parsed_args.heuristic_metric, parsed_args.decrement_step), feature_rankings, parsed_args.class_column).to_csv(output_file, index=False)
