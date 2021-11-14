import concurrent.futures
import importlib
import os
import sys

from lm4bld.experiment.nlp import CrossFoldExperiment
from lm4bld.experiment.nlp import CrossProjectTrainModelsExperiment
from lm4bld.experiment.nlp import NextTokenExperiment
from lm4bld.experiment.nlp import TokenizeExperiment

import lm4bld.experiment.config as config

def lookup_class(mod_name, cname):
    mod = importlib.import_module(mod_name)
    return getattr(mod, cname)

def main():
    # Kind of fragile, but...
    cfile = sys.argv[1] if len(sys.argv) == 2 else None
    conf = config.Config(cfile)

    maxjobs = conf.get_maxjobs() 
    exp_class = lookup_class("lm4bld.experiment.nlp", conf.get_task()) 

    if (maxjobs is None):
        maxjobs = os.cpu_count()

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=maxjobs)
    futures_list = list()

    for projname in conf.get_projects():
        exp = exp_class(projname, executor, conf)
        futures_list += exp.createFutures()

    for future in concurrent.futures.as_completed(futures_list):
        #assert (future.done() and not future.cancelled()
        #        and future.exception() is None)

        print(future.result())

    executor.shutdown()

if __name__ == "__main__":
    main()
