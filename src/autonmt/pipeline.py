import logging
import string
import sys

from .steps import LoadAndPreprocess, Tokenize, BuildVocab, Split, Train
from .configs import load_config, ConfigFactory

complete_pipeline = [
    ('load', LoadAndPreprocess),
    ('tokenize', Tokenize),
    ('build_vocab', BuildVocab),
    ('split', Split),
    ('train', Train)
]

available_steps = [name for name, _ in complete_pipeline]

name2step = { name: step for name, step in complete_pipeline }

class Pipeline:
    def __init__(self, config_filename, steps = available_steps, error_on_missing_config = False):
        self.config_factory = ConfigFactory(load_config(config_filename))
        self.steps = [(name, name2step[name]) for name in steps]
        self.error_on_missing_config = error_on_missing_config
    
    def run(self):
        logging.info(f'Pipeline steps: {", ".join(name for name, _ in self.steps)}')
        for i, (step_name, step) in enumerate(self.steps):
            logging.info(f'{string.ascii_uppercase[i]}) {step_name}')
            config = self.config_factory.build_for(step_name)
            if config is None:
                if self.error_on_missing_config:
                    logging.error(f'Missing configuration for step {step_name}')
                    sys.exit(1)
                else:
                    logging.warning(f'Missing configuration for step {step_name}, skipping step')
                    continue
            step(config).run()
