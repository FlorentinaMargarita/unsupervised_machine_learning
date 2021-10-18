import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import re


def clean(s):
    # remove non-ASCII characters
    s = s.encode('ascii', errors='ignore').decode()
    # remove punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    # convert tabs and newlines to spaces
    s = re.sub('[\n\t]', ' ', s)
    # remove leading spaces
    s = re.sub('^\s+', '', s)
    # remove trailing spaces
    s = re.sub('\s+$', '', s)
    # squeeze multiple spaces into one
    s = re.sub(r'\s\s+', ' ', s)
    return s


def load_json_data(path, n):
    with open(path, 'rb') as json_file:
        for i, line in enumerate(json_file):
            if i >= n: return
            if i % 50000 == 0: print(f'{100*i/1955945.0:.0f}% line {i}')
            paper = json.loads(line)
            year = int(re.search(r'\d{4}', paper['versions'][0]['created']).group())
            yield year, paper['title'], paper['categories'], paper['abstract']

for row in load_json_data(r'C:\Users\flore\Downloads\WPy64-3950\notebooks\arxiv-metadata-oai-snapshot.json', 5):
    print(row) 