from io import open
import os
import glob
from torchloop import definitions
from torchloop.util import nlp_utils

def read_lines(filename):
    # read all file content if file not big utf-8
    with open(filename, encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    return [nlp_utils.unicodeToAscii(line) \
            for line in lines]

def find_files(path):
    return glob.glob(path)

# remove any ext or dir name expose only fname
def file_name(fname):
    return os.path.splitext(os.path.basename(\
            fname))[0]

def src_home():
    return definitions.src_home

def torchloop_home():
    return definitions.torchloop_home

def default_conf_dir():
    return os.path.join(src_home(), "conf")

def default_model_dir():
    return os.path.join(src_home(), "models")

def default_bg_dir():
    return os.path.join(src_home(), "data", "bgs")

def default_font_dir():
    return os.path.join(src_home(), "data", "chn_fonts")
