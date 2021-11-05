import concurrent.futures
import sys
import yaml

import lm4bld.experiment.nlp as experiment

CONF_FILE = "lm4bld.yml"

# YAML Terms
NITER = 'niter'
NFOLDS = 'nfolds'
MINORDER = 'minorder'
MAXORDER = 'maxorder'
PROJECTS = 'projects'
MAXJOBS = 'maxjobs'
CROSSPROJ_ORDER = 'crossproj_order'
TASK = 'task'
VERSIONS = 'normalize_version_strings'
PATHS = 'normalize_paths'
NEXT_TOKEN_ORDER = 'next_token_order'
NEXT_TOKEN_TEST_SIZE = 'next_token_test_size'
MIN_CANDIDATES = 'min_candidates'
MAX_CANDIDATES = 'max_candidates'

def process_experimental_config(fname):
    fhandle = open(fname, 'r')
    return yaml.load(fhandle, Loader=yaml.FullLoader)

def rq1(projname, executor, confdata):
    pomlist = confdata[PROJECTS][projname]
    maxjobs = confdata[MAXJOBS]
    minorder = confdata[MINORDER]
    maxorder = confdata[MAXORDER]
    nfolds = confdata[NFOLDS]
    niter = confdata[NITER]
    versions = confdata[VERSIONS]
    paths = confdata[PATHS]

    return experiment.CrossFoldExperiment(projname, executor, pomlist, minorder,
                                          maxorder, nfolds, niter, versions,
                                          paths)

def rq2(projname, executor, confdata):
    order = confdata[CROSSPROJ_ORDER]
    pomlist = confdata[PROJECTS][projname]

    return experiment.CrossProjectExperiment(projname, executor, pomlist, order,
                                             confdata[PROJECTS],
                                             confdata[VERSIONS],
                                             confdata[PATHS])

def rq4(projname, executor, confdata):
    pomlist = confdata[PROJECTS][projname]
    order = confdata[NEXT_TOKEN_ORDER]
    testSize = confdata[NEXT_TOKEN_TEST_SIZE]
    minCandidates = confdata[MIN_CANDIDATES]
    maxCandidates = confdata[MAX_CANDIDATES]
    versions = confdata[VERSIONS]
    paths = confdata[PATHS]

    return experiment.NextTokenExperiment(projname, executor, pomlist, order,
                                          testSize, minCandidates,
                                          maxCandidates, versions, paths) 

if __name__ == "__main__":
    confdata = process_experimental_config(CONF_FILE)
    my_task = confdata[TASK]
    maxjobs = confdata[MAXJOBS]

    if (maxjobs is None):
        maxjobs = os.cpu_count()

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=maxjobs)
    futures_list = list()

    for projname in confdata[PROJECTS]:
        exp = globals()[my_task](projname, executor, confdata)
        futures_list += exp.createFutures()

    for future in concurrent.futures.as_completed(futures_list):
        assert (future.done() and not future.cancelled()
                and future.exception() is None)

        print(future.result())

    executor.shutdown()
