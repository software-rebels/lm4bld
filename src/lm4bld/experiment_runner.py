import sys
import yaml

from lm4bld import pom_nlp
CONF_FILE = "lm4bld.yml"

# YAML Terms
NITER = 'niter'
NFOLDS = 'nfolds'
MINORDER = 'minorder'
MAXORDER = 'maxorder'
PROJECTS = 'projects'
MAXJOBS = 'maxjobs'

def process_experimental_config(fname):
    fhandle = open(fname, 'r')
    return yaml.load(fhandle, Loader=yaml.FullLoader)

def main():
    confdata = process_experimental_config(CONF_FILE)

    for projname in confdata[PROJECTS]:
        pomlist = confdata[PROJECTS][projname]

        exp = pom_nlp.PomNLPMultirunExperiment(projname, pomlist, confdata[MINORDER],
                                               confdata[MAXORDER])

        exp.perform(confdata[NFOLDS], confdata[NITER], confdata[MAXJOBS])

if __name__ == "__main__":
    main()
