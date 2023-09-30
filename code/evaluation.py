from sklearn import svm
import sys
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, cross_val_predict
from methods.utils import get_dataset, get_base_parser, get_X_y, get_filename
import logging
from sklearn import metrics as skmetrics
from sklearn.metrics import confusion_matrix

def parse_args(argv):
    parser = argparse.ArgumentParser(parents=[get_base_parser()])
    parser.add_argument(
        '-m', '--model', metavar='str',
        help="Models to run evaluation: 'svm', 'rf' (for Random Forest) or 'all'. Default: 'all'",
        choices=['svm', 'rf','all'],
        type=str, default='all')
    parser.add_argument('--n_folds', metavar = 'INT', type = int, default = 10,
        help="Number of folds to use in cross validation. Default: 10")
    args = parser.parse_args(argv)
    return args

def get_classifiers(classifier_name):
    if(classifier_name == 'svm'):
        return {'svm': svm.SVC()}
    elif(classifier_name == 'rf'):
        return {'rf': RandomForestClassifier(random_state = 0)}
    return {'svm': svm.SVC(), 'rf': RandomForestClassifier(random_state = 0)}

def cross_validation(model, X, y, n_folds = 10, metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']):
    y_pred = cross_val_predict(estimator=model, X=X, y=y, cv=n_folds)

    metrics_results = {}
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

if __name__=="__main__":
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_eval
    logger_eval = logging.getLogger('EVALUATION')
    logger_eval.setLevel(logging.INFO)

    args = parse_args(sys.argv[1:])
    X, y = get_X_y(args, get_dataset(args))

    classifiers = get_classifiers(args.model)
    results = []

    for classifier_name, clf in classifiers.items():
        logger_eval.info('Running %s to dataset %s' % (classifier_name.upper(), args.dataset))
        results.append({**cross_validation(clf, X, y, args.n_folds), "model": classifier_name})
    pd.DataFrame(results).to_csv(get_filename(args.output_file, prefix=args.output_prefix), index = False)
