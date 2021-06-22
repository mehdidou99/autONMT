import os
import logging
import shutil
import sys
import subprocess

from .common import CorpusInfo, LanguagePair
from .util import run_and_check_result, SubprocessError

class Step:
    def __init__(self, config):
        self.config = config

    def run(self):
        raise NotImplementedError()

class LoadAndPreprocess(Step):
    def prepare_directories(self):
        try:
            os.makedirs(self.config.outputdatadir)
            logging.info(f'Created output data directory {self.config.outputdatadir}')
        except FileExistsError:
            logging.warning(f'Output data directory {self.config.outputdatadir} already exists')

        if os.path.exists(self.config.preprocdir):
            shutil.rmtree(self.config.preprocdir)
        os.makedirs(self.config.preprocdir)

    def load_raw_corpus_for_pair(self, corpus_info : CorpusInfo, pair : LanguagePair):
        src = pair.src
        tgt = pair.tgt
        logging.info('Loading raw data')
        input_src = f'{corpus_info.complete_path_for_pair(pair)}.{src}'
        input_tgt = f'{corpus_info.complete_path_for_pair(pair)}.{tgt}'
        if not os.path.exists(input_src) or not os.path.exists(input_tgt):
            if 'try_reversed_lg_pairs' in self.config.on_missing_data:
                logging.info(f'\'{corpus_info.filename_prefix_for_pair(pair)}\' data not found, trying with reversed language pair')
                input_src = f'{corpus_info.complete_path_for_pair(pair.reversed())}.{src}'
                input_tgt = f'{corpus_info.complete_path_for_pair(pair.reversed())}.{tgt}'
                if not os.path.exists(input_src) or not os.path.exists(input_tgt):
                    msg_prefix = f'Neither \'{corpus_info.filename_prefix_for_pair(pair)}\' nor \'{corpus_info.filename_prefix_for_pair(pair.reversed())}\' data found'
                    if 'exit_error' in self.config.on_missing_data:
                        logging.error(f'{msg_prefix}, aborting')
                        sys.exit(1)
                    else:
                        logging.warning(f'{msg_prefix}, skipping corpus for this language pair')
                        return None
            else:
                msg_prefix = f'\'{corpus_info.filename_prefix_for_pair(pair)}\' data not found'
                if 'exit_error' in self.config.on_missing_data:
                    logging.error(f'{msg_prefix}, aborting')
                    sys.exit(1)
                else:
                    logging.warning(f'{msg_prefix}, skipping corpus for this language pair')
                    return None

        with open(f'{self.config.preprocdir}/{corpus_info.filename_prefix_for_pair(pair)}.{src}', 'w') as f_src_out, open(f'{self.config.preprocdir}/{corpus_info.filename_prefix_for_pair(pair)}.{tgt}', 'w') as f_tgt_out:
            try:
                run_and_check_result(['head', '-n', f'{self.config.max_entries_per_corpus}', input_src], stdout=f_src_out, stderr=subprocess.PIPE)
                run_and_check_result(['head', '-n', f'{self.config.max_entries_per_corpus}', input_src], stdout=f_tgt_out, stderr=subprocess.PIPE)
            except SubprocessError as err:
                logging.error(err.result.stderr.decode('utf-8'))
                sys.exit(1)

        return [f'{corpus_info.filename_prefix_for_pair(pair)}.{src}', f'{corpus_info.filename_prefix_for_pair(pair)}.{tgt}']

    def generate_corpus_for_pair(self, corpus, path, pair):
        corpus_info = CorpusInfo(corpus, self.config.datadir, path)
        logging.info(f'{corpus} ({corpus_info.complete_path})')

        corpus_files = self.load_raw_corpus_for_pair(corpus_info, pair)

        preproc_steps = self.config.preprocessing_steps
        used_preproc_steps = [step for step in preproc_steps if corpus in self.config.step_config[step].corpora]
        logging.info(f'Preprocessing steps for this corpus: {", ".join(used_preproc_steps) if used_preproc_steps else "None"}')
        for i, step in enumerate(used_preproc_steps):
            step_config = self.config.step_config[step]
            assert(corpus in step_config.corpora)
            logging.info(f'{i+1}) {step}')
            try:
                sub_result = run_and_check_result([f'{self.config.scriptsdir}/{step_config.script}', self.config.preprocdir] + corpus_files, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except SubprocessError as err:
                logging.error(f'Error during \'{step}\' preprocessing step for \'{corpus}\' corpus')
                logging.error(err.result.stderr.decode('utf-8'))
                sys.exit(1)
            corpus_files = sub_result.stdout.decode('utf-8').strip().split('\n')
        return corpus_files

    def concatenate_generated(self, final_corpora_files, final_output_files):
        num_files_per_corpus = len(final_output_files)
        if not all(len(final_corpus_files) == num_files_per_corpus for final_corpus_files in final_corpora_files):
            error_msg = f'At the end of the preprocessing, every corpus should have as many output files as there are in \'{self.config.filename} > preprocessing > final_output_files\' in order to concatenate them into those final files'
            logging.error(error_msg)
            sys.exit(1)

        logging.info('Concatenating all generated data')
        for output_filename, input_filenames in zip(final_output_files, zip(*final_corpora_files)):
            logging.info(output_filename)
            with open(f'{self.config.preprocdir}/{output_filename}', 'wb') as output:
                for input_filename in input_filenames:
                    with open(f'{self.config.preprocdir}/{input_filename}', 'rb') as input:
                        shutil.copyfileobj(input, output)

    def run(self):
        self.prepare_directories()

        final_corpora_files = []
        for src in self.config.src_lgs:
            for tgt in self.config.tgt_lgs:
                pair = LanguagePair(src, tgt)
                logging.info(f'Generating \'{pair}\' data')
                for corpus, path in self.config.corpora.items():
                    final_corpus_files = self.generate_corpus_for_pair(corpus, path, pair)
                    if final_corpus_files is not None:
                        final_corpora_files.append(final_corpus_files)

        final_output_files = self.config.final_files
        self.concatenate_generated(final_corpora_files, final_output_files)

        for output_filename in final_output_files:
            shutil.move(f'{self.config.preprocdir}/{output_filename}', f'{self.config.outputdatadir}/{output_filename}')

class Tokenize(Step):
    def run(self):
        final_output_files = self.config.final_files
        logging.info('Tokenizing generated data')

        logging.info('Training tokenizer')
        base_learn_command = '/nfs/SSALING-DATA/michon/Tokenizer/build/cli/subword_learn --mode conservative'.split(' ')
        inputs = [['-i', f'{self.config.outputdatadir}/{final_file}'] for final_file in final_output_files]
        inputs = [arg for pair in inputs for arg in pair]
        output = ['-o', f'{self.config.outputdatadir}/bpe.32k']
        trailing_options = '-- bpe --symbols 32000'.split(' ')
        run_and_check_result(base_learn_command + inputs + output + trailing_options)

        logging.info('Tokenizing')
        tokenize_command = '/nfs/SSALING-DATA/michon/Tokenizer/build/cli/tokenize --mode conservative --joiner_annotate'.split(' ')
        tokenize_command += ['--bpe_model', f'{self.config.outputdatadir}/bpe.32k']
        for final_file in final_output_files:
            with open(f'{self.config.outputdatadir}/{final_file}', 'rb') as tokenizer_input, \
                    open(f'{self.config.outputdatadir}/{final_file}.tok', 'wb') as tokenizer_output:
                run_and_check_result(tokenize_command, stdin=tokenizer_input, stdout=tokenizer_output)

class BuildVocab(Step):
    def run(self):
        logging.info(f'Building following vocabularies: {" ".join(self.config.vocabs.keys())}')
        base_command = 'onmt-build-vocab --size 32000'.split(' ')
        for i, (vocab_name, vocab_config) in enumerate(self.config.vocabs.items()):
            logging.info(f'{i+1}) {vocab_name}')
            output_filename = vocab_config.output
            output = ['--save_vocab', f'{self.config.outputdatadir}/{output_filename}']
            inputs_filenames = vocab_config.files
            inputs = [f'{self.config.outputdatadir}/{input_filename}' for input_filename in inputs_filenames]
            with open(f'{self.config.outputdatadir}/{output_filename}.log', 'wb') as log_file:
                run_and_check_result(base_command + output + inputs, stdout=log_file, stderr=log_file)

class Split(Step):
    def run(self):
        to_split = self.config.files
        logging.info(f'Splitting following files: {" ".join(to_split)}')
        parts_descs = [f'{name} ({val} lines)' for name, val in self.config.parts.items()]
        logging.info(f'Parts: {", ".join(parts_descs)}')
        logging.info(f'Remaining: {self.config.remain}')
        logging.info(f'Seed: {self.config.seed}')
        base_command = 'python3 /nfs/RESEARCH/crego/projects/corpora-tools/corpus/corpus-split-sets.py'.split(' ')
        inputs = [['-data', f'{self.config.outputdatadir}/{input_filename}'] for input_filename in to_split]
        inputs = [arg for pair in inputs for arg in pair]
        parts = [['-set', f'{name},{val}'] for name, val in self.config.parts.items()]
        parts = [arg for pair in parts for arg in pair]
        remain = ['-remain', self.config.remain]
        trailing_options = ['-v', '-seed', str(self.config.seed)]
        try:
            run_and_check_result(base_command + inputs + parts + remain + trailing_options, stderr=subprocess.PIPE)
        except SubprocessError as err:
            logging.error(f'Error during data splitting')
            logging.error(err.result.stderr.decode('utf-8'))
            sys.exit(1)

class Train(Step):
    def run(self):
        try:
            os.makedirs(self.config.outputdir)
            logging.info(f'Created output directory {self.config.outputdir}')
        except FileExistsError:
            logging.warning(f'Output directory {self.config.outputdir} already exists')
        current_env = os.environ.copy()
        current_env['CUDA_VISIBLE_DEVICES'] = 0
        command_base = ['onmt-main']
        model_spec = ['--model', self.config.model_file] if self.config.custom_model else ['--model_type', self.config.model_type]
        model_config = ['--config', self.config.model_config_file]
        command_suffix = '--auto_config train'.split(' ')
        options = ['--with_eval'] if 'eval' in self.config.options else []
        with open(f'{self.config.outputdir}/train.log', 'wb') as log_file:
            run_and_check_result(command_base + model_spec + model_config + command_suffix + options, stdout=log_file, stderr=log_file)
