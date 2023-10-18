from sklearn import svm
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, cross_val_predict
import logging
from sklearn import metrics as skmetrics
from sklearn.metrics import confusion_matrix
import os

def get_X_y(args, dataset):
    X = dataset.drop(columns = args.class_column)
    y = dataset[args.class_column]
    return X, y

def get_classifier(ml_model):
    if ml_model == 'svm':
        clf = svm.SVC()
    elif ml_model == 'rf':
        clf = RandomForestClassifier(random_state = 0)
    return clf

def cross_validation(classifier, X, y, n_folds = 10, metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']):
    y_pred = cross_val_predict(estimator = classifier, X = X, y = y, cv = n_folds)

    metrics_results = dict()
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    metrics_results['tn'] = tn
    metrics_results['fp'] = fp
    metrics_results['fn'] = fn
    metrics_results['tp'] = fp
    metrics_results['accuracy'] = skmetrics.accuracy_score(y, y_pred)
    metrics_results['precision'] = skmetrics.precision_score(y, y_pred, zero_division = 0)
    metrics_results['recall'] = skmetrics.recall_score(y, y_pred, zero_division = 0)
    metrics_results['f1'] = skmetrics.f1_score(y, y_pred, zero_division = 0)
    metrics_results['roc_auc'] = skmetrics.roc_auc_score(y, y_pred)
    return metrics_results

def graph_metrics(data_df, args, method, dataset):
    models_index = list(data_df['model'].str.upper())
    metrics_dict = dict()
    metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    for metric in metrics_list:
        metrics_dict[metric] = list(data_df[metric] * 100.0)

    df = pd.DataFrame(metrics_dict, index = models_index)
    df.columns = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AuC']
    ax = df.plot.bar(rot = 0, edgecolor='white', linewidth = 1)
    ax.set_xlabel('Model')
    ax.set_ylabel('Values (%)')
    ax.legend(ncol = 1, loc = 'lower left')
    ax.set_ylim(0, 100)
    ax.set_xlim(-1, len(models_index))
    ax.set_title(f'Results for {method} With Dataset in {dataset}')
    path_graph_file = os.path.join(args.output, f'metrics_{method}_{os.path.basename(dataset).replace(".csv", ".png")}')
    ax.figure.savefig(path_graph_file, dpi = 300)


def graph_class(data_df, args, method, dataset):
    models_index = list(data_df['model'].str.upper())
    classification_dict = dict()
    classification_list = ['TP', 'FP', 'TN', 'FN']
    for classification in classification_list:
        classification_dict[classification] = list(data_df[classification.lower()])

    df = pd.DataFrame(classification_dict, index = models_index)
    stacked_data = df.apply(lambda x: x*100/sum(x), axis = 1)
    ax = stacked_data.plot.barh(rot = 0, stacked = True)
    ax.set_xlabel('Values (%)')
    ax.set_ylabel('Model')
    ax.set_ylim(-1, len(models_index))
    ax.legend(ncol = len(classification_list), loc = 'upper center')
    for container in ax.containers:
        if container.datavalues[0] > 2.5:
            ax.bar_label(container, label_type = 'center', color = 'black', weight='bold', fmt = '%.2f')
    ax.set_title(f'Classification to {method} With Dataset in {dataset}')
    path_graph_file = os.path.join(args.output, f'class_{method}_{os.path.basename(dataset).replace(".csv", ".png")}')
    ax.figure.savefig(path_graph_file, dpi = 300)

def plot_results(df, args, method, dataset):
        graph_metrics(df, args, method, dataset)
        graph_class(df, args, method, dataset)

def run_ml_models(args, models, method, dataset):
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger
    logger = logging.getLogger('Evaluation')
    logger.setLevel(logging.INFO)
    reduced_ds = os.path.join(args.output, f'{method}_{os.path.basename(dataset)}')
    try:
        print(f'Loading Dataset {reduced_ds}')
        rds = pd.read_csv(reduced_ds)
    except BaseException as e:
        msg = colored(e, 'red')
        logger.exception(msg)
        exit(1)

    if args.class_column not in rds.columns:
        msg = colored(f'Expected Dataset {reduced_ds} to Have a Class Column Named "{args.class_column}"', 'red')
        logger.exception(msg)
        exit(1)

    X, y = get_X_y(args, rds)

    results = list()
    for model in models:
        clf = get_classifier(model)
        logger.info(f'Running {model.upper()} to Dataset {reduced_ds}')
        results.append({**cross_validation(clf, X, y), 'model': model})
    results_file = os.path.join(args.output, f'evaluation_{os.path.basename(reduced_ds)}')
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index = False)
    plot_results(results_df, args, method, dataset)
