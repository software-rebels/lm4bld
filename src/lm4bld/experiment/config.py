import os
import yaml

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
POMLISTDIR = 'pomlistdir'
SRCLISTDIR = 'srclistdir'
PREFIX = 'prefix'
TOKENPREFIX = 'tokenprefix'
MODELPREFIX = 'modelprefix'
TESTRATIOTHRESHOLD = 'test_ratio_threshold'
PROCESS_POMS = 'process_poms'
PROCESS_SOURCE = 'process_source'

class Config:
    def __init__(self, conf_file):
        if conf_file is None:
            conf_file = CONF_FILE

        fhandle = open(conf_file, 'r')
        self.confdata = yaml.load(fhandle, Loader=yaml.FullLoader)
        fhandle.close()

    def get_task(self):
        return self.confdata[TASK]

    def get_maxjobs(self):
        return self.confdata[MAXJOBS]

    def get_projects(self):
        return self.confdata[PROJECTS]

    def get_minorder(self):
        return self.confdata[MINORDER]

    def get_maxorder(self):
        return self.confdata[MAXORDER]

    def get_crossproj_order(self):
        return self.confdata[CROSSPROJ_ORDER]

    def get_nfolds(self):
        return self.confdata[NFOLDS]

    def get_niter(self):
        return self.confdata[NITER]

    def get_versions(self):
        return self.confdata[VERSIONS]

    def get_paths(self):
        return self.confdata[PATHS]

    def get_next_token_order(self):
        return self.confdata[NEXT_TOKEN_ORDER]

    def get_next_token_test_size(self):
        return self.confdata[NEXT_TOKEN_TEST_SIZE]

    def get_min_candidates(self):
        return self.confdata[MIN_CANDIDATES]

    def get_max_candidates(self):
        return self.confdata[MAX_CANDIDATES]

    def get_prefix(self):
        return self.confdata[PREFIX]

    def get_tokenprefix(self):
        return self.confdata[TOKENPREFIX]

    def get_modelprefix(self):
        return self.confdata[MODELPREFIX]

    def get_testratiothreshold(self):
        return self.confdata[TESTRATIOTHRESHOLD]

    def get_process_source(self):
        return self.confdata[PROCESS_SOURCE]

    def get_process_poms(self):
        return self.confdata[PROCESS_POMS]

    def get_pomlist(self, projname):
        return f'{self.confdata[POMLISTDIR]}{os.path.sep}{projname}.txt'

    def get_srclist(self, projname):
        return f'{self.confdata[SRCLISTDIR]}{os.path.sep}{projname}.txt'
