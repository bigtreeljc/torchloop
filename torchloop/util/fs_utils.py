from io import open
import glob

def readLines(file):
    # read all file content if file not big utf-8
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]
 
