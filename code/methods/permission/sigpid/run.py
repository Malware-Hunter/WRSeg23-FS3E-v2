import pandas as pd
import numpy  as np
import timeit
import argparse
import csv
import os, sys
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from .spinner import Spinner
from methods.utils import get_base_parser, get_dataset, get_filename
import logging
from tqdm import tqdm

def add_arguments(parser):
    parser = parser.add_argument_group("Arguments for SigPID")

def S_B(j):
    global B
    global M
    sigmaBij = B.sum(axis = 0, skipna = True)[j]
    sizeBj = B.shape[0]
    sizeMj = M.shape[0]
    return (sigmaBij/sizeBj)*sizeMj

def PRNR(j):
    sigmaMij = (M.sum(axis = 0, skipna = True)[j]) * 1.0
    S_Bj = S_B(j)
    r = (sigmaMij - S_Bj)/(sigmaMij + S_Bj) if sigmaMij > 0.0 and S_Bj > 0.0 else 0.0
    return r

def check_dirs():
    import shutil
    root_path = os.getcwd()
    root_path = os.path.join(root_path, 'MLDP')
    #print(root_path)
    if os.path.exists(root_path):
        shutil.rmtree('MLDP')
    dirs = ['PRNR', 'SPR', 'PMAR']
    for dir in dirs:
        path = os.path.join(root_path, dir)
        os.makedirs(path)
        #print('Directory', dir, 'Created.')

def calculate_PRNR(dataset, filename, class_column):
    permissions = dataset.drop(columns = [class_column])
    with open(filename,'w', newline='') as f:
        f_writer = csv.writer(f)
        for p in permissions:
            permission_PRNR_ranking = PRNR(p)
            if permission_PRNR_ranking != 0:
                f_writer.writerow([p, permission_PRNR_ranking])

def permission_list(filename, asc):
    colnames = ['permission', 'rank']
    list = pd.read_csv(filename, names = colnames)
    list = list.sort_values(by = ['rank'], ascending = asc)
    #print(list)
    return list

def SVM(dataset_df, class_column):
    from sklearn import metrics
    state = np.random.randint(100)
    Y = dataset_df[class_column]
    X = dataset_df.drop([class_column], axis = 1)
    #split between train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify = Y, test_size = 0.3,random_state = 1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.3, random_state = 1)

    svm = SVC(kernel = 'linear', C = 1.0, random_state = 1)
    svm.fit(X_train,y_train)

    y_pred=svm.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy

def run_PMAR(dataset, prnr_malware, class_column):
    global logger_sigpid
    features_name = dataset.columns.values.tolist()
    class_apk = dataset[class_column]
    features_dataset = dataset.drop([class_column], axis=1)
    num_apk = features_dataset.shape[0] - 1
    num_features = features_dataset.shape[1]

    logger_sigpid.info("Mining Association Rules")
    records = list()
    for i in range(0,num_apk):
        if class_apk[i] in [0, 1]:
            i_list = []
            for j in range(0,num_features):
                if features_dataset.values[i][j] == 1:
                    i_list.append(features_name[j])
            records.append(i_list)

    te = TransactionEncoder()
    te_ary = te.fit(records).transform(records)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    freq_items = apriori(df,
                        min_support = 0.1,
                        use_colnames = True,
                        max_len = 2,
                        verbose = 0)
    if not freq_items.empty:
        rules = association_rules(freq_items, metric="confidence", min_threshold=0.965)
    else:
        rules = list()

    PMAR_df = dataset
    deleted_ft = []
    for i in range(0, len(rules)):
        ant = list(rules.loc[i,'antecedents'])[0]
        con = list(rules.loc[i,'consequents'])[0]
        rank_ant = prnr_malware.loc[(prnr_malware['permission'] == ant)].values[0,1]
        rank_con = prnr_malware.loc[(prnr_malware['permission'] == con)].values[0,1]
        to_delete = ant if rank_ant < rank_con else con
        if to_delete not in deleted_ft:
            PMAR_df = PMAR_df.drop([to_delete], axis=1)
            deleted_ft.append(to_delete)
    return PMAR_df

def drop_internet(dataset):
    cols = list()
    to_drop = ['android.permission.INTERNET', 'INTERNET']
    features = dataset.columns.values.tolist()
    cols = list(set(features).intersection(to_drop))
    ds = dataset.drop(columns = cols)
    return ds

def run(args, ds):
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger_sigpid
    logger_sigpid = logging.getLogger('SigPID')
    logger_sigpid.setLevel(logging.INFO)

    check_dirs()
    args = args
    try:
        initial_dataset = pd.read_csv(ds)
    except BaseException as e:
        logger_sigpid.exception(e)
        exit(1)

    dataset = drop_internet(initial_dataset)
    global B
    global M
    B = dataset[dataset[args.class_column] == 0]
    M = dataset[dataset[args.class_column] == 1]

    calculate_PRNR(B, 'MLDP/PRNR/PRNR_B_List.csv', args.class_column)
    calculate_PRNR(M, 'MLDP/PRNR/PRNR_M_List.csv', args.class_column)

    benigns_permissions = permission_list('MLDP/PRNR/PRNR_B_List.csv', True)
    malwares_permissions = permission_list('MLDP/PRNR/PRNR_M_List.csv' , False)

    num_permissions = dataset.shape[1] - 1 #CLASS

    logger_sigpid.info('PRNR Generating Subset of Permissions')
    counter = increment = 3
    while counter < num_permissions/2 + increment:
        malwares_head_perms = malwares_permissions['permission'].head(counter).values
        benigns_head_perms = benigns_permissions['permission'].head(counter).values
        subset_permissions = list(set(malwares_head_perms) | set(benigns_head_perms))
        subset_permissions.append(args.class_column)
        subset = dataset[subset_permissions]
        evaluated_ft = counter * 2
        evaluated_ft = num_permissions if evaluated_ft > num_permissions else evaluated_ft
        subset.to_csv(f'MLDP/PRNR/subset_{evaluated_ft}.csv', index = False)
        counter += increment

    counter = increment = 6
    best_PRNR_accuracy = 0.0
    best_PRNR_counter = 0

    with open('MLDP/PRNR/svm_results.csv', 'w', newline='') as f:
        f_writer = csv.writer(f)
        logger_sigpid.info('Running PIS + PRNR')
        pbar = tqdm(range(num_permissions), disable = (logger_sigpid.getEffectiveLevel() > logging.INFO))
        while counter < num_permissions + increment:
            evaluated_ft = num_permissions if counter > num_permissions else counter
            pbar.set_description(f'With {evaluated_ft} Features')
            pbar.n = evaluated_ft
            dataset_df = pd.read_csv(f'MLDP/PRNR/subset_{evaluated_ft}.csv', encoding = 'utf8')
            accuracy = SVM(dataset_df, args.class_column)
            if accuracy > best_PRNR_accuracy:
                best_PRNR_accuracy = accuracy
                best_PRNR_counter = evaluated_ft
            f_writer.writerow([evaluated_ft] + accuracy)
            counter += increment
        pbar.close()

    logger_sigpid.info(f'Best Accuracy: {best_PRNR_accuracy:.3f}. Number of Features: {best_PRNR_counter}')
    #SPR
    PRNR_df = pd.read_csv(f'MLDP/PRNR/subset_{best_PRNR_counter}.csv', encoding = 'utf8')
    PRNR_df = PRNR_df.drop(columns=[args.class_column])

    #calculates the support of each permission
    supp = PRNR_df.sum(axis = 0)
    supp = supp.sort_values(ascending = False)

    logger_sigpid.info('SPR Generating Subset of Permissions')
    counter = increment = 5
    while counter < best_PRNR_counter + increment:
        subset_permissions = list(supp.head(counter).index)
        subset_permissions.append(args.class_column)
        subset = dataset[subset_permissions]
        evaluated_ft = best_PRNR_counter if counter > best_PRNR_counter else counter
        subset.to_csv(f'MLDP/SPR/subset_{evaluated_ft}.csv', index = False)
        counter += increment

    counter = increment = 5
    best_SPR_accuracy = best_PRNR_accuracy
    best_SPR_counter = best_PRNR_counter
    with open('MLDP/SPR/svm_results.csv','w', newline='') as f:
        f_writer = csv.writer(f)
        logger_sigpid.info('Running PIS + SPR')
        pbar_spr = tqdm(range(best_PRNR_counter), disable = (logger_sigpid.getEffectiveLevel() > logging.INFO))
        while counter < best_PRNR_counter + increment:
            evaluated_ft = best_PRNR_counter if counter > best_PRNR_counter else counter
            pbar_spr.set_description(f'With {evaluated_ft} Features')
            pbar_spr.n = evaluated_ft
            dataset_df = pd.read_csv(f'MLDP/SPR/subset_{evaluated_ft}.csv', encoding = 'utf8')
            accuracy = SVM(dataset_df, args.class_column)
            if accuracy >= 0.9 and evaluated_ft < best_SPR_counter:
                best_SPR_accuracy = accuracy
                best_SPR_counter = evaluated_ft
            f_writer.writerow([evaluated_ft] + accuracy)
            counter += increment
        pbar_spr.close()

    logger_sigpid.info(f'SPR Pruning Point >> Best Accuracy: {best_SPR_accuracy:.3f}. Number of Features: {best_SPR_counter}')

    #PMAR
    SPR_df = pd.read_csv(f'MLDP/SPR/subset_{best_SPR_counter}.csv', encoding = 'utf8')
    final_dataset = run_PMAR(SPR_df, malwares_permissions, args.class_column)

    # final_dataset.to_csv("sigpid_"+ds[0]+".csv", index = False)
    output_dir = args.output
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f'sigpid_{os.path.basename(ds)}')
    final_dataset.to_csv(output_file, index = False)
    final_perms = final_dataset.shape[1] - 1
    num_permissions = initial_dataset.shape[1] - 1
    pct = (1.0 - (final_perms/num_permissions)) * 100.0
    logger_sigpid.info(f'{num_permissions} to {final_perms} Features. Reduction of {pct:.2f}')
