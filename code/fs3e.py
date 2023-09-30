#!/usr/bin/python3

import argparse
from termcolor import colored, cprint
from methods.utils import *
import argparse
import sys
import glob
import asyncio
from itertools import chain
import pandas as pd
import re
import seaborn as sns
import logging
import os
from importlib import import_module

class DefaultHelpParser(argparse.ArgumentParser):
    def error(self, message):
        global logger
        self.print_help()
        msg = colored(message, 'red')
        logger.error(msg)
        sys.exit(2)

def load_methods_args(parser, args):
    global methods_path
    global methods_type
    global methods_dict

    selected_methods = args.fs_methods
    selected_types = methods_types
    if args.ft_types:
        selected_types = args.ft_types

    for type in selected_types:
        methods = methods_dict[type]
        for mth in methods:
            if not selected_methods or mth in selected_methods:
                module = '.'.join([methods_path, type, mth, 'run'])
                model_instance = import_module(module)
                model_instance.add_arguments(parser)

def modify_choices(parser, dest, choices):
    for action in parser._actions:
        if action.dest == dest:
            action.choices = choices
            action.help += '. Choices: ' + str(choices)
            return
    else:
        raise AssertionError('argument {} not found'.format(dest))

def parse_args(argv):
    global methods_types
    global methods_dict

    action_base_parser = argparse.ArgumentParser(add_help = False)
    action_base_group = action_base_parser.add_mutually_exclusive_group(required = True)
    action_base_group.add_argument(
        '--ft-types', nargs = '+', metavar = 'TYPE',
        help = 'Methods of SELECTED Features Types. Choices: ' + str(methods_types),
        choices = methods_types, type = str)
    action_base_group.add_argument(
        '--all-ft-types', help = f'Methods of ALL Features Types',
        action = 'store_true')

    parser = DefaultHelpParser(formatter_class = argparse.RawTextHelpFormatter)
    parser._optionals.title = 'Optional Arguments'
    action_subparser = parser.add_subparsers(title = 'Available Actions', dest = 'action', metavar = '', required = True)
    action_list = action_subparser.add_parser(
        'list', help = 'List Features Selection Methods',
        parents = [action_base_parser])
    action_list._optionals.title = 'Optional Arguments'
    action_run = action_subparser.add_parser(
        'run', help = 'Run Features Selection Methods',
        parents = [action_base_parser])
    action_run._optionals.title = 'Optional Arguments'
    args_, _ = parser.parse_known_args()

    available_methods = list()
    selected_types = methods_types
    if args_.action == 'run':
        if args_.ft_types:
            selected_types = args_.ft_types
        #print(selected_types)
        for type in selected_types:
            #print(methods_dict[type])
            available_methods += methods_dict[type]

    action_run_group = action_run.add_mutually_exclusive_group(required = True)
    action_run_group.add_argument(
        '--fs-methods', nargs = '+', metavar = 'METHOD',
        help = 'Run Selected Methods. Choices: ' + str(available_methods),
        choices = available_methods, type = str)
    action_run_group.add_argument(
        '--all-fs-methods', help = f'Run ALL Methods',
        action = 'store_true')
    action_run.add_argument(
        '-d', '--datasets', nargs = '+', metavar = 'DATASET',
        help = 'One or More Datasets (csv Files). For All Datasets in Directory Use: [DIR_PATH]/*.csv',
        type = str,  required = True)
    action_run.add_argument('-c', '--class-column', type = str, default="class", metavar = 'CLASS_COLUMN',
        help = 'Name of the class column. Default: "class"')
    action_run.add_argument(
        '--verbose', help = "Show More Run Info.",
        action = 'store_true')
    action_run.add_argument(
        '--output', help = "Output File Directory. Default: ./results",
        type = str, default = './results')
    args_, _ = parser.parse_known_args()

    if args_.action == 'run':
        load_methods_args(action_run, args_)

    args = parser.parse_args(argv)
    return args

def get_dir_list(dir_path):
    l = list()
    for it in os.scandir(dir_path):
        if it.is_dir():
            l.append(it.name)
    if '__pycache__' in l:
        l.remove('__pycache__')
    return l

def list_methods(selected_methods_types):
    for type in selected_methods_types:
        dir_path = os.path.join(methods_path, type)
        methods_in_dir = get_dir_list(dir_path)
        f = open(os.path.join(dir_path, "about.desc"), "r")
        method_desc = f.read()
        print(colored("\n>>> " + method_desc, 'green'))
        for i in methods_in_dir:
            f = open(os.path.join(dir_path, i, "about.desc"), "r")
            method_desc = f.read()
            print(colored("\t" + method_desc, 'yellow'))
    exit(1)

def get_methods():
    global methods_types
    d = {}
    for type in methods_types:
        methods_type_path = os.path.join(methods_path, type)
        dir_list = get_dir_list(methods_type_path)
        d[type] = dir_list
    return d

def run_methods(args, type, mth, ds):
    print("Running Method", colored(mth, 'green'), "to Dataset", colored(ds, 'green'))
    module = '.'.join([methods_path, type, mth, 'run'])
    model_instance = import_module(module)
    model_instance.run(args,ds)



if __name__ == '__main__':
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global methods_path
    global methods_type
    global methods_dict
    global logger
    global all_results
    global predict_results
    global roc_results
    logger = logging.getLogger('FS3E')
    predict_results = list()
    methods_path = 'methods'
    methods_types = get_dir_list(methods_path)
    methods_dict = get_methods()
    all_results = pd.DataFrame()
    roc_results = pd.DataFrame()
    predict_results = list()

    #print(methods_dict)

    args = parse_args(sys.argv[1:])


    if args.action == 'list':
        if args.ft_types:
            list_methods(args.ft_types)
        elif args.all_ft_types:
            list_methods(methods_types)

    if args.verbose:
        logger.setLevel(logging.INFO)

    datasets_list = args.datasets
    for ds in datasets_list:
        try:
            print(f'carregando dataset {ds}')
            #dataset = pd.read_csv(ds)
        except BaseException as e:
            msg = colored("Exception: {}".format(e), 'red')
            logger.error(msg)
            exit(1)

        selected_methods = args.fs_methods
        selected_types = methods_types
        if args.ft_types:
            selected_types = args.ft_types

        for type in selected_types:
            methods = methods_dict[type]
            for mth in methods:
                if not selected_methods or mth in selected_methods:
                    run_methods(args, type, mth, ds)
