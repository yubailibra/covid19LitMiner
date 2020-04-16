# tools

from spacy.tokens import Doc
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from scireader.utils import *
import re


def query_keywords(bank, keyword, similarity=0.9, verbose=False, display=False):
    paperids = bank.text.index.values.tolist()
    allwords = list(set([item for sublist in bank.list_words for item in sublist]))
    allstitched = [' '.join(sublist) for sublist in bank.list_words]
    popkeyword = '\\b' + '.*'.join(re.sub("-", ' ', keyword).split(' ')) + '\\b'
    keywords = [allwords[i] for i in range(len(allwords)) if re.search(popkeyword, allwords[i]) is not None]
    if verbose:
        print('\nkeyword hits:', keywords[0:min(20, len(keywords))], '...')
    cleankeyword = re.sub("\s+", " ", re.sub(r'[\.\*\(\)\[\]]', ' ',
                                             re.sub(r'\[-\\s\]', ' ', re.sub(r'\\b', '', popkeyword)))).lower().strip(
        '.- ')
    cleankeywords = [allwords[i] for i in range(len(allwords)) if re.search(keyword, allwords[i]) is not None]
    synonyms_hits = bank.vocabulary.vectors.most_similar(
        queries=np.array([bank.vocabulary.get_vector(cleankeyword)]), sort=True, n=100)
    synonyms = [bank.vocabulary.strings.__getitem__(synonyms_hits[0][0][i]) for i in range(len(synonyms_hits[0][0]))
                if synonyms_hits[2][0][i] >= similarity]
    synonyms = list(set(synonyms).difference(set(cleankeywords)))
    if verbose:
        print("\nsynonyms hits:", synonyms[0:min(20, len(synonyms))], '...')
    keywords.extend(list(set(synonyms).difference(set(keywords))))
    # sitched
    fromStiched = [paperids[i] for i in range(len(allstitched)) if re.search(keyword, allstitched[i]) is not None]

    if len(keywords) == 0:
        if verbose:
            1  # print(str(0) + ' keyword(' + keyword + ')+synonym' + ', ',
            #      str(len(fromStiched)) + ' hit by fromStiched')
        fromStiched = [[h, keyword] for h in fromStiched]
        return fromStiched
    else:
        ctVectorSingle = CountVectorizer(lowercase=True, analyzer=lambda l: l, vocabulary=keywords)
        hitmat = ctVectorSingle.fit_transform(bank.list_words).toarray()
        termFreq = hitmat.sum(axis=1).flatten()

        df_termFreq = pd.Series(termFreq, index=paperids, name='term.freq')
        sortedTF = df_termFreq[termFreq.nonzero()[0]].sort_values(axis=0, ascending=False)
        # hitmat=hitmat[np.where(termFreq>0),:]
        if display:
            # print(str(sortedTF.shape[0]) + ' hit by keyword+synonym: ' + keyword + ', ',
            #      str(len(fromStiched)) + ' hit by fromStiched')
            sortedHitsByTF = bank.text.loc[sortedTF.index.values.tolist(), ['title', 'abstract']]
            sortedHitsByTF['term.freq'] = sortedTF
            print("\n" + str(len(
                set(sortedTF.index.values.tolist()).union(set(fromStiched)))) + " papers found with keyword+synonym")
            print(sortedHitsByTF.head())
        finalhits = list(set(sortedTF.index.values.tolist()).union(set(fromStiched)))
        finalhits = [[h, keyword] for h in finalhits]
    if (finalhits is None or len(finalhits) == 0):
        print('no hit for keyword ' + cleankeyword)
    return finalhits


def scanPapersByKW(bank, kw_outcomes=[], kw_differences=[], kw_designs=[], similarity_outcome=0.99,
                   similarity_difference=0.99, similarity_design=0.9, verbose=False, analyze=False):
    ## Yunchen Yang: Arguments 'allwords' and 'allstitched' were deleted as they were not used here.
    hits_outcomes = []
    hits_differences = []
    hits_designs = []
    rs_outcomes = {}
    rs_differences = {}
    rs_designs = {}
    # try:

    for k in kw_outcomes:
        hits = query_keywords(bank, k, similarity_outcome, verbose=verbose)
        if (len(hits) > 0):
            hits_outcomes.extend([hit[0] for hit in hits])
            for hit in hits:
                if hit[0] not in rs_outcomes.keys():
                    rs_outcomes[hit[0]] = [k]
                else:
                    rs_outcomes[hit[0]].append(k)
    hits_outcomes = list(set(hits_outcomes))

    for k in kw_differences:
        hits = query_keywords(bank, k, similarity_difference, verbose=verbose)
        if (len(hits) > 0):
            hits_differences.extend([hit[0] for hit in hits])
            for hit in hits:
                if hit[0] not in rs_differences.keys():
                    rs_differences[hit[0]] = [k]
                else:
                    rs_differences[hit[0]].append(k)
    hits_differences = list(set(hits_differences))

    for k in kw_designs:
        hits = query_keywords(bank, k, similarity_design, verbose=verbose)
        if len(hits) > 0:
            hits_designs.extend([hit[0] for hit in hits])
            for hit in hits:
                if hit[0] not in rs_designs.keys():
                    rs_designs[hit[0]] = [k]
                else:
                    rs_designs[hit[0]].append(k)
    hits_designs = list(set(hits_designs))
    # except:
    #    print(k)
    ol1 = set(hits_outcomes).union(set(hits_differences))
    ol2 = ol1.union(set(hits_designs))
    finalhits = list(ol2)
    if analyze:
        return finalhits, hits_outcomes, hits_differences, hits_designs, rs_outcomes, rs_differences, rs_designs
    else:
        return finalhits


def sentSimilarity(bank, sent1, sent2, useCosine=False):
    if useCosine:
        doc1 = Doc(bank.vocabulary, words=my_token_analyzer(sent1, bank.model))
        doc2 = Doc(bank.vocabulary, words=my_token_analyzer(sent2, bank.model))
    else:
        doc1 = bank.model(sent1)
        doc2 = bank.model(sent2)
    wmddist = doc1.similarity(doc2)
    return wmddist


def queryBySentOnePaper(bank, qsent, sent_tokens, distance, useCosine=False, badwords=None):
    if useCosine:
        # qdoc=Doc(bank.vocabulary,words=list(nlp(qsent))))
        qdoc = Doc(bank.vocabulary, words=my_token_analyzer(qsent, bank.model))
    else:
        qdoc = bank.model(qsent)
    hits = []
    if badwords is not None:
        badwords_ = badwords
    else:
        badwords_ = ["copyright", 'copy right', 'preprint', 'biorxiv', 'medrxiv', 'bioRxiv', 'medRxiv',
                     'acknowledgement', 'acknowledgements', 'https', 'http', 'palabra clave']
    for tokens in sent_tokens:
        tokens = list(filter(lambda a: a not in badwords_, tokens))
        if len(tokens) > 2:
            refdoc = Doc(bank.vocabulary, words=tokens)
            try:
                wmddist = qdoc.similarity(refdoc)
                if useCosine:
                    if wmddist >= distance:
                        hits.append([' '.join(tokens), wmddist])
                else:
                    if wmddist <= distance:
                        hits.append([' '.join(tokens), wmddist])
            except:
                continue  # print(qsent,tokens)
                ## Yunchen Yang: replace number 1 with 'continue'
    return hits


def queryBySentAllPapers(bank, selpaperids, qsent, distance, useCosine=False):
    paperids = selpaperids  # list(bank.text.index)[pstart:pstop]
    allpaperids = list(bank.text.index)
    all_sent_tokens = [bank.sent_tokens[allpaperids.index(i)] for i in paperids]
    allhits = []
    for i in range(len(all_sent_tokens)):
        hits = queryBySentOnePaper(bank, qsent, all_sent_tokens[i], distance, useCosine=useCosine)
        if len(hits) > 0:
            allhits.append([paperids[i], hits])
        else:
            continue  # print(qsent,i)
            ## Yunchen Yang: replace number 1 with 'continue'
    return allhits


def scanPapersBySent(bank, selpaperids, sent_outcomes=[], sent_differences=[], sent_designs=[], distance=3,
                     useCosine=False, analyze=False):
    hits_outcomes = []
    hits_differences = []
    hits_designs = []
    rs_outcomes = {}
    rs_differences = {}
    rs_designs = {}
    # try:

    for qsent in sent_outcomes:
        qsent = cleanStr(qsent).lower().strip('.- ')
        if qsent not in rs_outcomes.keys():
            rs_outcomes[qsent] = []
        hits = queryBySentAllPapers(bank, selpaperids, qsent, distance, useCosine=useCosine)
        if (len(hits) > 0):
            hits_outcomes.extend([hit[0] for hit in hits])
            rs_outcomes[qsent].extend(hits)
            # print('found '+str(len(hits))+' papers related to topic: ', qsent)
    hits_outcomes = list(set(hits_outcomes))

    for qsent in sent_differences:
        qsent = cleanStr(qsent).lower().strip('.- ')
        if qsent not in rs_differences.keys():
            rs_differences[qsent] = []
        hits = queryBySentAllPapers(bank, selpaperids, qsent, distance, useCosine=useCosine)
        if (len(hits) > 0):
            hits_differences.extend([hit[0] for hit in hits])
            rs_differences[qsent].extend(hits)
            # print('found '+str(len(hits))+' papers related to topic: ', qsent)
    hits_differences = list(set(hits_differences))

    for qsent in sent_designs:
        qsent = cleanStr(qsent).lower().strip('.- ')
        if qsent not in rs_designs.keys():
            rs_designs[qsent] = []
        hits = queryBySentAllPapers(bank, selpaperids, qsent, distance, useCosine=useCosine)
        if (len(hits) > 0):
            hits_designs.extend([hit[0] for hit in hits])
            rs_designs[qsent].extend(hits)
            # print('found '+str(len(hits))+' papers related to topic: ', qsent)
    hits_designs = list(set(hits_designs))
    # except:
    #    print(k)
    ol1 = set(hits_outcomes).union(set(hits_differences))
    ol2 = ol1.union(set(hits_designs))
    finalhits = list(ol2)

    sorted_finalhits = {}
    for topic in rs_outcomes.keys():
        pid_outcomes = {}
        if len(rs_outcomes) > 0:
            pid_outcomes = dict(zip([each[0] for each in rs_outcomes[topic]],
                                    [min([sent2score[1] for sent2score in each[1]]) for each in rs_outcomes[topic]]))
            if useCosine:
                pid_outcomes = dict(zip([each[0] for each in rs_outcomes[topic]],
                                        [max([sent2score[1] for sent2score in each[1]]) for each in
                                         rs_outcomes[topic]]))

        pid_differences = {}
        if len(rs_differences) > 0:
            pid_differences = dict(zip([each[0] for each in rs_differences[topic]],
                                       [min([sent2score[1] for sent2score in each[1]]) for each in
                                        rs_differences[topic]]))
            if useCosine:
                pid_differences = dict(zip([each[0] for each in rs_differences[topic]],
                                           [max([sent2score[1] for sent2score in each[1]]) for each in
                                            rs_differences[topic]]))
        pid_designs = {}
        if len(rs_designs) > 0:
            pid_designs = dict(zip([each[0] for each in rs_designs[topic]],
                                   [min([sent2score[1] for sent2score in each[1]]) for each in rs_designs[topic]]))
            if useCosine:
                pid_designs = dict(zip([each[0] for each in rs_designs[topic]],
                                       [max([sent2score[1] for sent2score in each[1]]) for each in rs_designs[topic]]))

        hits2score = {}
        for hit in finalhits:
            bestScore = 1000
            if useCosine:
                bestScore = 0
            if hit in pid_outcomes.keys():
                currentScore = pid_outcomes[hit]
                if currentScore < bestScore:
                    bestScore = currentScore
                if useCosine:
                    if currentScore > bestScore:
                        bestScore = currentScore
            if hit in pid_differences.keys():
                currentScore = pid_differences[hit]
                if currentScore < bestScore:
                    bestScore = currentScore
                if useCosine:
                    if currentScore > bestScore:
                        bestScore = currentScore
            if hit in pid_designs.keys():
                currentScore = pid_designs[hit]
                if currentScore < bestScore:
                    bestScore = currentScore
                if useCosine:
                    if currentScore > bestScore:
                        bestScore = currentScore
            hits2score[hit] = bestScore
        # sorted_finalhits[topic]=[each[0] for each in list(sorted(hits2score.items(), key=lambda x: x[1]))]
        sorted_finalhits[topic] = list(sorted(hits2score.items(), key=lambda x: x[1]))
        if useCosine:
            sorted_finalhits[topic] = list(sorted(hits2score.items(), key=lambda x: x[1], reverse=True))

    if analyze:
        return sorted_finalhits, hits_outcomes, hits_differences, hits_designs, rs_outcomes, rs_differences, rs_designs
    else:
        return sorted_finalhits
