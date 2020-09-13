import argparse
import os
from datetime import datetime

class BaseOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('exp_name', type=str, default='test', help='Experiment name')

    
    def parse(self, fixed=None):
        if fixed is not None:
            args = self.parser.parse_args(fixed)
        else:
            args = self.parser.parse_args()
        
        return args
    

    def initialize(self, fixed=None):
        self.args = self.parse(fixed)

        return self.args
    

    def update(self, args, extra_options):

        for k, v in extra_options.items():
            setattr(args, k, v)