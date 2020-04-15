import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import sklearn.metrics
from scireader.utils import *


class PaperBank:
    def __init__(self, model):
        self.model = model
        self.text = []
        self.list_words = []
        # self.allwords=[]
        self.sent_tokens = []
        self.vocabulary = self.model.vocab  # Vocab() #

    def read(self, files):
        dict_text = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': []}
        for file in files:
            try:
                with open(file, 'r') as f:
                    text = json.load(f)
                    parsed = FileExtractor(text)
                    dict_text['paper_id'].append(parsed['paper_id'])
                    dict_text['title'].append(parsed['title'])
                    dict_text['abstract'].append(parsed['abstract'])
                    # dict_text['body_text'].append(parsed['title']+". "+parsed['abstract']+". "+parsed['body_text'])
                    dict_text['body_text'].append('')
                    dict_text['authors'].append(concatAuthors(parsed["authors"]))
            except:
                print('cannot read file:', file)
        self.text = pd.DataFrame(dict_text, columns=['paper_id', 'title', 'abstract', 'body_text', 'authors'])
        self.text = self.text.set_index('paper_id')

    def parse(self, textfile='abstract'):
        # dictionary: {named entity: vector} from paper abstracts
        # ne2vec=[ent2vector(text, self.model) for text in self.text[textfile]]
        # ne2vec=[enttoken2vector(text, self.model) for text in self.text[textfile]]
        ne2vec = [enttoken2vector(text, self.model) for text in self.text[textfile]]
        self.sent_tokens = [sublist[1] for sublist in ne2vec]

        # Per paper, save its named entities as a bag of words
        self.list_words = [[each[0] for each in sublist[0]] for sublist in ne2vec]

        vector_data = {each[0]: each[1] for sublist in ne2vec for each in sublist[0]}
        for word, vector in vector_data.items():
            self.vocabulary.set_vector(word, vector)

    def query(self, keyword, similarity, verbose=False):
        allwords = list(set([item for sublist in self.list_words for item in sublist]))
        keywords = [allwords[i] for i in range(len(allwords)) if re.search(keyword, allwords[i]) is not None]
        if verbose:
            print('keyword hits', keywords)
        synonyms_hits = self.vocabulary.vectors.most_similar(queries=np.array([self.vocabulary.get_vector(keyword)]),
                                                             sort=True, n=100)
        synonyms = [self.vocabulary.strings.__getitem__(synonyms_hits[0][0][i]) for i in range(len(synonyms_hits[0][0]))
                    if synonyms_hits[2][0][i] >= similarity]
        synonyms = list(set(synonyms).difference(set(keywords)))
        if verbose:
            print("\nadditional synonyms:", synonyms)
        keywords.extend(synonyms)

        ctVectorSingle = CountVectorizer(lowercase=True, analyzer=lambda l: l, vocabulary=keywords)
        termFreq = ctVectorSingle.fit_transform(self.list_words).toarray().sum(axis=1).flatten()

        df_termFreq = pd.Series(termFreq, index=self.text.index.values.tolist(), name='TermFreq')
        sortedTF = df_termFreq[termFreq.nonzero()[0]].sort_values(axis=0, ascending=False)
        if verbose:
            print(sortedTF.shape[0])
        sortedHitsByTF = self.text.loc[sortedTF.index.values.tolist(), ['title', 'abstract']]
        sortedHitsByTF['TermFreq'] = sortedTF
        return sortedHitsByTF

    def query_keywords(self, keyword, similarity=0.9, verbose=False):
        paperids = self.text.index.values.tolist()
        allwords = list(set([item for sublist in self.list_words for item in sublist]))
        allstitched = [' '.join(sublist) for sublist in self.list_words]
        keywords = [allwords[i] for i in range(len(allwords)) if re.search(keyword, allwords[i]) is not None]
        if verbose:
            print('\nkeyword hits', keywords)
        cleankeyword = re.sub("\s+", " ", re.sub(r'[\.\*\(\)\[\]]', ' ',
                                                 re.sub(r'\[-\\s\]', ' ', re.sub(r'\\b', '', keyword)))).lower().strip(
            '.- ')
        synonyms_hits = self.vocabulary.vectors.most_similar(
            queries=np.array([self.vocabulary.get_vector(cleankeyword)]), sort=True, n=100)
        synonyms = [self.vocabulary.strings.__getitem__(synonyms_hits[0][0][i]) for i in range(len(synonyms_hits[0][0]))
                    if synonyms_hits[2][0][i] >= similarity]
        synonyms = list(set(synonyms).difference(set(keywords)))
        if verbose:
            print("additional synonyms:", synonyms)
        keywords.extend(synonyms)
        # sitched
        fromStiched = [paperids[i] for i in range(len(allstitched)) if re.search(keyword, allstitched[i]) is not None]

        if len(keywords) == 0:
            if verbose:
                print(str(0) + ' keyword(' + keyword + ')+synonym' + ', ',
                      str(len(fromStiched)) + ' hit by fromStiched')
            fromStiched = [[h, keyword] for h in fromStiched]
            return fromStiched
        else:
            ctVectorSingle = CountVectorizer(lowercase=True, analyzer=lambda l: l, vocabulary=keywords)
            hitmat = ctVectorSingle.fit_transform(self.list_words).toarray()
            termFreq = hitmat.sum(axis=1).flatten()

            df_termFreq = pd.Series(termFreq, index=paperids, name='TermFreq')
            sortedTF = df_termFreq[termFreq.nonzero()[0]]  # .sort_values(axis=0, ascending=False)
            # hitmat=hitmat[np.where(termFreq>0),:]
            if verbose:
                print(str(sortedTF.shape[0]) + ' hit by keyword+synonym: ' + keyword + ', ',
                      str(len(fromStiched)) + ' hit by fromStiched')
            # sortedHitsByTF=self.text.loc[sortedTF.index.values.tolist(),['title','abstract']]
            # sortedHitsByTF['TermFreq']=sortedTF
            # finalhits=list(set(list(sortedHitsByTF.index)).union(set(fromStiched)))
            finalhits = list(set(sortedTF.index.values.tolist()).union(set(fromStiched)))
            finalhits = [[h, keyword] for h in finalhits]
        if (finalhits is None or len(finalhits) == 0):
            print(keyword)
        return finalhits

    def clustering_tfidf(self, min_df=3, combine=False, sim_thre=0.95, max_features=2 ** 18):
        ctVectorSingle = CountVectorizer(lowercase=True, analyzer=lambda l: l, min_df=min_df, max_features=max_features)
        X = ctVectorSingle.fit_transform(self.list_words)
        words = ctVectorSingle.get_feature_names()
        if combine:
            vectors = np.zeros((X.toarray().shape[1], 200))
            for i, word in enumerate(words):
                vectors[i, :] = self.model(word).vector
            sim = sklearn.metrics.pairwise.cosine_similarity(vectors)
            indices = np.where(sim > sim_thre)
            index_list = []
            word_list = []
            for i, i1 in enumerate(indices[0]):
                i2 = indices[1][i]
                if i2 > i1:
                    index_list.append([i1, i2])
                    word_list.append([words[i1], words[i2]])
            sort_list = similarity_sort(index_list)
            flatten_sort = list(sum(sort_list, []))
            x_post = np.zeros((X.toarray().shape[0], X.toarray().shape[1] - len(flatten_sort) + len(sort_list)))
            words_new = []
            list_tem = [i for i in range(len(words)) if i not in flatten_sort]
            c = '/'
            for n, idx in enumerate(list_tem):
                words_new.append(words[idx])
                x_post[:, n] = X.toarray()[:, idx]
            for m, combine_ids in enumerate(sort_list):
                word_new = c.join([words[index] for index in combine_ids])
                words_new.append(word_new)
                for index in combine_ids:
                    x_post[:, m - len(sort_list) - 1] += X.toarray()[:, index]
        else:
            x_post = X.toarray()
        tfidf = TfidfTransformer()
        tfidf_mat = tfidf.fit_transform(x_post)
        tfidf_array = tfidf_mat.toarray()
        return tfidf_array