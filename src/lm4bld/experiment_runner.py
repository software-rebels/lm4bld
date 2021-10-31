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

def process_experimental_config(fname):
    fhandle = open(fname, 'r')
    return yaml.load(fhandle, Loader=yaml.FullLoader)

def print_results(results, t):
    for proj in results:
        for order in results[proj]:
            for res in results[proj][order]:
                print(f'{proj},{order},{t},{res}')

def main():
    confdata = process_experimental_config(CONF_FILE)

    unk_rates = {}
    ents = {}

    for projname in confdata[PROJECTS]:
        print(projname)
        pomlist = confdata[PROJECTS][projname]

        exp = pom_nlp.PomNLPMultirunExperiment(pomlist, confdata[MINORDER],
                                               confdata[MAXORDER])

        unk_rates[projname], ents[projname] = exp.perform(confdata[NFOLDS], confdata[NITER])

    print_results(unk_rates, "oov_rate")
    print_results(ents, "entropy")

if __name__ == "__main__":
    main()
