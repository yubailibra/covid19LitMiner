import collections
import sys
import copy
import re
import json
import spacy
import scispacy
import en_core_sci_lg
import glob
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from scispacy.abbreviation import AbbreviationDetector
#from scispacy.umls_linking import UmlsEntityLinker
from spacy.vocab import Vocab
import sklearn.metrics
from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
from compress_pickle import dump, load
import scipy.sparse
import pytextrank
from wmd import WMD
#from spacy.pipeline import Sentencizer
from spacy.tokens import Doc

from scireader import *

nlp = en_core_sci_lg.load()

abbreviation_pipe = AbbreviationDetector(nlp)
nlp.add_pipe(abbreviation_pipe)

nlp.add_pipe(WMD.SpacySimilarityHook(nlp), last=True)

root_path = '/home/duai/Desktop/Github/CORD-19-research-challenge'
jsonfiles = glob.glob(f'{root_path}/biorxiv_medrxiv/biorxiv_medrxiv/**/*.json', recursive=True)[0:10]

bank=PaperBank(model=nlp)

bank.read(jsonfiles)
bank.parse('abstract')


kw_covid19=['\\bcovid19','\\bcovid.19','\\bcoronavirus','\\bsars.*\\bcov','\\bsars cov','\\bmers.*\\bcov','\\bmers cov']
junk,hits_covid19,junk, junk,rs_covid19,junk,junk=scanPapersByKW(bank,kw_covid19,[],[],similarity_outcome=0.8,similarity_difference=0.99,similarity_design=0.9)