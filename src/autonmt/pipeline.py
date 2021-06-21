import logging
from .steps import LoadAndPreprocess, Tokenize, BuildVocab, Split
from .common import CompleteConfig

complete_pipeline = [
    ('load', LoadAndPreprocess),
    ('tokenize', Tokenize),
    ('build_vocab', BuildVocab),
    ('split', Split)
]

available_steps = [name for name, _ in complete_pipeline]

name2step = { name: step for name, step in complete_pipeline }

class Pipeline:
    def __init__(self, config_filename, steps = available_steps):
        self.config = CompleteConfig(config_filename)
        self.steps = [(name, name2step[name]) for name in steps]
    
    def run(self):
        logging.info(f'Pipeline steps: {", ".join(name for name, _ in self.steps)}')
        for i, (step_name, step) in enumerate(self.steps):
            logging.info(f'{i+1}) {step_name}')
            step(self.config).run()