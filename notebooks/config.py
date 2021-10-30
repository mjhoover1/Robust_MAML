import os

PATH = os.getcwd() # Home directoy

EPSILON = 1e-8

if PATH is None:
    raise Exception('Configure your data folder location in config.py before continuing!')
