import logging
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

class LoadAndPreprocessConfig:
    def __init__(self, raw) -> None:
        expname = raw['name']
        self.outputdatadir = raw['data']['output_dir'].rstrip('/')
        self.outputdatadir = f'{self.outputdatadir}/{expname}'
        self.preprocdir = f'/tmp/{expname}'
        self.src_lgs = raw['data']['src_lgs']
        self.tgt_lgs = raw['data']['tgt_lgs']
        self.corpora = raw['data']['corpora']
        self.datadir = raw['data']['dir'].rstrip('/')
        self.on_missing_data = raw['data'].get('on_missing_data', [])
        self.max_entries_per_corpus = int(raw['data']['max_entries_per_corpus'])
        self.preprocessing_steps = raw['preprocessing']['steps']

        class StepConfig:
            def __init__(this, step_raw) -> None:
                this.corpora = step_raw['corpora']
                this.script = step_raw['script']
        self.step_config = { step: StepConfig(raw[f'preprocessing_{step}']) for step in raw['preprocessing']['steps'] }
        self.scriptsdir = raw['preprocessing']['scripts_dir']
        self.final_files = raw['preprocessing']['final_files']

class TokenizeConfig:
    def __init__(self, raw) -> None:
        self.final_files = raw['preprocessing']['final_files']
        expname = raw['name']
        self.outputdatadir = raw['data']['output_dir'].rstrip('/')
        self.outputdatadir = f'{self.outputdatadir}/{expname}'

class BuildVocabConfig:
    def __init__(self, raw) -> None:
        expname = raw['name']
        self.outputdatadir = raw['data']['output_dir'].rstrip('/')
        self.outputdatadir = f'{self.outputdatadir}/{expname}'
        class VocabConfig:
            def __init__(this, vocab_raw) -> None:
                this.output = vocab_raw['save_to']
                this.files = vocab_raw['files']
        self.vocabs = { vocab: VocabConfig(vocab_raw) for vocab, vocab_raw in raw['vocab'].items() }

class SplitConfig:
    def __init__(self, raw) -> None:
        self.files = raw['splitting']['files']
        self.parts = raw['splitting']['parts']
        self.remain = raw['splitting']['remain']
        self.seed = int(raw['splitting']['seed'])
        expname = raw['name']
        self.outputdatadir = raw['data']['output_dir'].rstrip('/')
        self.outputdatadir = f'{self.outputdatadir}/{expname}'

class TrainConfig:
    def __init__(self, raw) -> None:
        expname = raw['name']
        self.outputdir = raw['train']['output_dir'].rstrip('/')
        self.outputdir = f'{self.outputdir}/{expname}'
        model_file = raw['model'].get('file')
        if model_file is not None:
            self.model_file = model_file
            self.custom_model = True
        else:
            self.model_type = raw['model']['type']
            self.custom_model = False
        self.model_config_file = raw['model']['config']
        self.options = raw['train']['options']

class ConfigFactory:
    configs = {
        'load': LoadAndPreprocessConfig,
        'tokenize': TokenizeConfig,
        'build_vocab': BuildVocabConfig,
        'split': SplitConfig,
        'train': TrainConfig
    }

    def __init__(self, raw) -> None:
        self.raw = raw

    def build_for(self, step):
        if step not in ConfigFactory.configs.keys():
            raise NotImplementedError(f'Factory cannot build config for step {step}')
        try:
            result = ConfigFactory.configs[step](self.raw)
        except KeyError as err:
            logging.warning(f'Missing config element: {err}')
            return None
        return result
