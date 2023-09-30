from sklearn.model_selection import cross_val_score
import sys
import argparse
import numpy as np
from abc import ABCMeta
from six import add_metaclass
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


@add_metaclass(ABCMeta)
class ArtificialBee(object): # Abelhas   
    TRIAL_INITIAL_DEFAULT_VALUE = 0
    INITIAL_DEFAULT_PROBABILITY = 0.0

    def __init__(self, source_food,fitness):
        self.source_food = source_food
        self.fitness = fitness
        self.trial = ArtificialBee.TRIAL_INITIAL_DEFAULT_VALUE
        self.prob = ArtificialBee.INITIAL_DEFAULT_PROBABILITY

    #def compute_prob(self, max_fitness):
    #    self.prob = self.fitness / max_fitness
    
    def get_fitness(self):
        return 1 / (1 + self.fitness) if self.fitness >= 0 else 1 + np.abs(self.fitness)


    def explore(self, food, MR):
        R = np.random.rand(food.shape[0])
        mask = R < MR
        food[mask] = 1
        return food

    def random_solutions(self,source_food, n_features):
        i = np.random.randint(0, n_features)
        k = np.random.randint(0, n_features)
        source_food[i] = np.int64(bool(source_food[i]) ^ bool(source_food[i]) ^ bool(source_food[k]))
        return source_food

class ABC:
    def __init__(self, X, y, estimator, scorer, n_iter, max_trials, MR, k_folds):
        self.X = X
        self.y = y
        self.estimator = estimator
        self.scorer = scorer
        self.n_features = X.shape[1]

        self.n_iter = n_iter
        self.max_trials = max_trials
        self.MR = MR
        self.k_folds = k_folds

        self.optimal_solution = None

        self.bees = []

    def initialize(self):   
        initial_population = np.eye(self.n_features, dtype=np.int64)
        
        # Calcula score populacao inicial
        for itr in range(self.n_features):
            self.X_selected = self.X[:, initial_population[itr, :] == 1]
            scores = cross_val_score(estimator=self.estimator, X= self.X_selected, y= self.y, scoring= self.scorer, cv= self.k_folds)
            scores_mean = np.mean(scores)
            self.bees.append(ArtificialBee(initial_population[itr], scores_mean))

     

    '''def employee_bees_phase(self):
        for itr in self.bees:
            
            search_neighbor = itr.explore(itr.source_food, self.MR)
            
            # calcula score do vizinho
            scores_neighbor = cross_val_score(estimator=self.estimator, X= self.X[:, np.array(search_neighbor, dtype=bool)], y= self.y, 
                                        scoring= self.scorer,cv= self.k_folds)
            scores_mean_neighbor = np.mean(scores_neighbor)
            # verifica se o vizinho tem uma pontuacao melhor que a solucao atual
            
            if (itr.fitness < scores_mean_neighbor):
                self.bees.append(ArtificialBee(search_neighbor, scores_mean_neighbor))
            else:
                itr.trial =+ 1
    ''' 
    ##############################################################################
    def employee_bees_phase(self):
        pool = mp.Pool()
        results = [pool.apply_async(self._calculate_score_neighbor, args=(itr.source_food, itr.fitness)) for itr in self.bees]
        pool.close()
        pool.join()
        for bee, result in zip(self.bees, results):
            search_neighbor, scores_mean_neighbor = result.get()
            if (bee.fitness < scores_mean_neighbor):
                self.bees.append(ArtificialBee(search_neighbor, scores_mean_neighbor))
            else:
                bee.trial += 1

    
    def _calculate_score_neighbor(self, source_food, fitness):
        search_neighbor = ArtificialBee.explore(self, source_food, self.MR)
        X_selected = self.X[:, np.array(search_neighbor, dtype=bool)]
        scores_neighbor = cross_val_score(estimator=self.estimator, X=X_selected, y=self.y, scoring=self.scorer, cv=self.k_folds)
        scores_mean_neighbor = np.mean(scores_neighbor)
        return search_neighbor, scores_mean_neighbor

     ##################################################################################

    '''def __onlooker_bees_phase(self):
        for itr in self.best_food_sources:

            search_neighbor = itr.explore(itr.source_food, self.MR)
            
            # calcula score do vizinho
            scores_neighbor = cross_val_score(estimator=self.estimator, X= self.X[:, np.array(search_neighbor, dtype=bool)], y= self.y, 
                                        scoring= self.scorer,cv= self.k_folds)
            scores_mean_neighbor = np.mean(scores_neighbor)
            # verifica se o vizinho tem uma pontuacao melhor que a solucao atual
            
            if (itr.fitness < scores_mean_neighbor):
                self.bees.append(ArtificialBee(search_neighbor, scores_mean_neighbor))
            else:
                itr.trial =+ 1

        self.best_food_sources = []
'''
    def __onlooker_bees_phase(self):
        pool = mp.Pool()
        results = [pool.apply_async(self._calculate_score_neighbor, args=(itr.source_food, itr.fitness)) for itr in self.best_food_sources]
        pool.close()
        pool.join()
        self.bees = []
        for result in results:
            search_neighbor, scores_mean_neighbor = result.get()
            self.bees.append(ArtificialBee(search_neighbor, scores_mean_neighbor))
        self.best_food_sources = []


    def __scout_bee_phase(self):
        max_trial_bees = [itr for itr in self.bees if itr.trial > self.max_trials]
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for itr in max_trial_bees:
                itr.source_food = np.random.randint(2, size=self.n_features)
                itr.fitness = pool.apply_async(self.__compute_fitness, args=(itr.source_food,))
            pool.close()
            pool.join()
        self.bees = [itr for itr in self.bees if itr.trial <= self.max_trials] + [itr for itr in max_trial_bees if itr.fitness > self.optimal_solution.fitness]

    def __compute_fitness(self, source_food):
        return np.mean(cross_val_score(estimator=self.estimator, X= self.X[:, np.array(source_food, dtype=bool)], y= self.y, 
                                        scoring= self.scorer,cv= self.k_folds))

                
    def __update_optimal_solution(self):
        self.optimal_solution = max(self.bees, key=lambda bee: bee.fitness, default=self.optimal_solution)

    
    def __calculate_probabilities(self):
        sum_fitness = sum(bee.get_fitness() for bee in self.bees)
        for bee in self.bees:
            bee.prob = bee.fitness / sum_fitness


    def __select_best_food_sources(self):
        self.best_food_sources = [bee for bee in self.bees if bee.prob > np.random.uniform(low=0, high=1)]
        if not self.best_food_sources:
            self.best_food_sources = [bee for bee in self.bees if bee.prob > np.random.uniform(low=0, high=1)]

    
    def feature_selection(self):
        self.initialize()
        for itr in range(self.n_iter):
            self.employee_bees_phase()
            self.__calculate_probabilities()
            self.__select_best_food_sources()
            self.__onlooker_bees_phase()
            self.__scout_bee_phase()    
            self.__update_optimal_solution()
        return self.optimal_solution.source_food

def float_range(mini,maxi):
    # Define the function with default arguments
    def float_range_checker(arg):
        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("Must be a Floating Point Number")
        if f <= mini or f >= maxi:
            raise argparse.ArgumentTypeError("Must be > " + str(mini) + " and < " + str(maxi))
        return f
    return float_range_checker

def add_arguments(parser):
    parser = parser.add_argument_group("Arguments for FSDroid")
    parser.add_argument(
        '-e', '--estimator', metavar='ESTIMATOR',
        help="Estimator.",
        choices=['svm', 'rf', 'knn', 'dt'],
        type=str, default='svm')
    parser.add_argument(
        '-es', '--escorer', metavar='ESCORER',
        help="Escorer.",
        choices=['accuracy', 'f1-score', 'recall', 'precision'],
        type=str, default='accuracy')
    parser.add_argument(
        '-it', '--iteration', metavar='ITERATION',
        help="ITERATION.",
        type=int, default=10)
    parser.add_argument(
        '-max', '--max-trials', metavar='MAX-trials',
        help="Max-trials.",
        type=int, default=10)
    parser.add_argument(
        '-mr','--mr', metavar='MR',
        help="MUTATION.",
        type=float_range(0.0, 1.0), default=0.5)
    parser.add_argument(
        '-k', '--k-fold', metavar='K',
        help="K.",
        type=int,
        default=3)

def create_csv(dataset, selected_columns, args,ds):
    selected_column_indices = np.where(selected_columns == 1)[0]
    new_columns = dataset.columns[selected_column_indices].tolist() + [dataset.columns[-1]]
    output_dir = args.output
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f'abc_{os.path.basename(ds)}')
    dataset[new_columns].to_csv(output_file, index = False)

        
def run(args,ds):
    args = args

    try:
        dataset = pd.read_csv(ds)
    except BaseException as e:
        print('Exception: {}'.format(e))
        exit(1)
    

    if args.estimator == 'svm':
        clf = svm.SVC()
    if args.estimator == 'rf':
        clf = RandomForestClassifier(random_state = 0)
    if args.estimator == 'knn':
        clf = KNeighborsClassifier()
    if args.estimator == 'dt':
        clf = DecisionTreeClassifier()

    
    
    X = dataset.drop(args.class_column, axis=1)
    y = dataset[args.class_column]
    
    X_np = X.to_numpy()
    y_np = y.to_numpy()
    abc = ABC(X_np, y_np, clf, args.escorer, args.iteration, args.max_trials, args.mr, args.k_fold)
    features_selected = abc.feature_selection()
    create_csv(dataset, features_selected,args,ds)
