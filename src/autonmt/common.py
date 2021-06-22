import sys

import yaml

def load_config(filename):
    with open(filename, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print('Invalid configuration')
            print(exc)
            sys.exit(1)

class CompleteConfig:
    def __init__(self, filename) -> None:
        self.filename = filename
        config = load_config(filename)
        self.raw = config
        self.expname = config['name']
        self.outputdir = config['output_dir'].rstrip('/')
        self.outputdir = f'{self.outputdir}/{self.expname}'
        self.outputdatadir = config['data']['output_dir'].rstrip('/')
        self.outputdatadir = f'{self.outputdatadir}/{self.expname}'
        self.datadir = config['data'].get('dir', '.').rstrip('/')
        self.scriptsdir = config['preprocessing'].get('scripts_dir', '.').rstrip('/')
        self.max_entries_per_corpus = int(config['data']['max_entries_per_corpus'])
        self.preprocdir = f'/tmp/{self.expname}'

class LanguagePair:
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def __str__(self):
        return f'{self.src}-{self.tgt}'

    def reversed(self):
        return LanguagePair(self.tgt, self.src)

class CorpusInfo:
    def __init__(self, corpus, datadir, path):
        self.name = corpus
        self.path = path
        self.complete_path = f'{datadir}/{path}' if path and path[0] != '/' else path

    def complete_path_for_pair(self, pair : LanguagePair):
        return f'{self.complete_path}.{pair}'

    def filename_prefix_for_pair(self, pair : LanguagePair):
        return self.complete_path_for_pair(pair).split('/')[-1]