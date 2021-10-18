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


# read the JSON data file and yield the relevant columns
# for each line (metadata record).
def load_json_data(path, n):
    with open(path, 'rb') as json_file:
        # for each line in json_file, line number "i"
        for i, line in enumerate(json_file):
            # we were only asked to read n lines, so if we've read more,
            # we should stop.
            if i >= n: return
            # if we're at a multiple of 50k lines, let's print a progress
            # message with the number of lines read and what percent of
            # the total number of records it is.
            if i % 50000 == 0: print(f'{100*i/1955945.0:.0f}% line {i}')
            # parse the line as JSON.  "paper" is now a dict with the
            # paper metadata.
            paper = json.loads(line)
            # each metadata entry has a list of versions: when it was first
            # uploaded, and then version entries for any subsequent changes.
            # we use the year of the first version (upload) as the date
            # of the paper.
            # the "created" field has a full timestamp, e.g.
            # Tue, 24 Jul 2007 20:10:27 GMT
            # so we find the year by looking for any four digits.
            year = int(re.search(r'\d{4}', paper['versions'][0]['created']).group())
            # clean (handle non-ASCII characters, punctuation, and whitespace)
            # the title and abstract, and yield all the fields of interest.
            yield year, clean(paper['title']), paper['categories'], clean(paper['abstract'])

for row in load_json_data(r'C:\Users\flore\Downloads\WPy64-3950\notebooks\arxiv-metadata-oai-snapshot.json', 5):
    print(row) 