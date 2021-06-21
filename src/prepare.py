#!/usr/bin/env python3

import argparse
import logging

from autonmt import available_steps, Pipeline

def main():
    parser = argparse.ArgumentParser(description='Run custom OpenNMT-tf pipeline')
    parser.add_argument('config', type=str, help='filename of the pipeline config file')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--steps', type=str, nargs='*', default=available_steps, help='steps to run in the pipeline')
    parser.add_argument('--skip', type=str, nargs='*', default=[], help='steps to skip in the pipeline')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    steps = [step for step in args.steps if step not in args.skip]
    Pipeline(args.config, steps).run()

if __name__ == '__main__':
    main()
