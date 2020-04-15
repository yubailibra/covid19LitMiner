from spacy.tokens import Doc
from scireader.utils import *


# todo: remember to fix abbrev in the input string


def prepKW(model, outcomes, differences, designs):  # ,kw_outcomes=None, kw_differences=None, kw_designs=None
    # if kw_outcomes is None:
    phrase_outcomes = [re.sub('/', ' ', each).lower().strip('.- ') for each in outcomes.split(',')]
    phrase_designs = [re.sub('/', ' ', each).lower().strip('.- ') for each in designs.split(',')]
    phrase_differences = [re.sub('/', ' ', each).lower().strip('.- ') for each in differences.split(',')]

    kw_outcomes = []
    for p in phrase_outcomes:
        kw_outcomes.extend(getEnt(p, model))
    kw_outcomes = list(set(kw_outcomes))
    # if kw_differences is None:
    kw_designs = []
    for p in phrase_designs:
        kw_designs.extend(getEnt(p, model))
    kw_designs = list(set(kw_designs))
    # if kw_designs is None:
    kw_differences = []
    for p in phrase_differences:
        kw_differences.extend(getEnt(p, model))
    kw_differences = list(set(kw_differences))
    return kw_outcomes, kw_differences, kw_designs


def scanPapersByKW(bank, kw_outcomes, kw_differences, kw_designs, similarity_outcome=0.9,
                   similarity_difference=0.95, similarity_design=0.9, verbose=False):
    ## Yunchen Yang: Arguments 'allwords' and 'allstitched' were deleted as they were not used here.
    hits_outcomes = []
    hits_differences = []
    hits_designs = []
    rs_outcomes = {}
    rs_differences = {}
    rs_designs = {}
    # try:

    for k in kw_outcomes:
        # hits=bank.query_keywords(k,similarity=0.8,verbose=False)
        hits = bank.query_keywords(k, similarity_outcome, verbose=verbose)
        if len(hits) > 0:
            hits_outcomes.extend([hit[0] for hit in hits])
            for hit in hits:
                if hit[0] not in rs_outcomes.keys():
                    rs_outcomes[hit[0]] = [k]
                else:
                    rs_outcomes[hit[0]].append(k)
    hits_outcomes = list(set(hits_outcomes))

    for k in kw_differences:
        # hits=bank.query_keywords(k,similarity=0.8,verbose=False)
        hits = bank.query_keywords(k, similarity_difference, verbose=verbose)
        if (len(hits) > 0):
            hits_differences.extend([hit[0] for hit in hits])
            for hit in hits:
                if hit[0] not in rs_differences.keys():
                    rs_differences[hit[0]] = [k]
                else:
                    rs_differences[hit[0]].append(k)
    hits_differences = list(set(hits_differences))

    for k in kw_designs:
        # hits=bank.query_keywords(k,similarity=0.8,verbose=False)
        hits = bank.query_keywords(k, similarity_design, verbose=verbose)
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
    return finalhits, hits_outcomes, hits_differences, hits_designs, rs_outcomes, rs_differences, rs_designs


def queryBySentOnePaper(bank, qsent, sent_tokens, distance, useCosine=False, badwords=None):
    if useCosine:
        # qdoc=Doc(bank.vocabulary,words=list(nlp(qsent))))
        qdoc = Doc(bank.vocabulary, words=my_token_analyzer(qsent))
    else:
        qdoc = bank.model(qsent)
    hits = []
    if badwords is not None:
        badwords_ = badwords
    else:
        badwords_ = ["copyright", 'copy right', 'preprint', 'biorxiv', 'medrxiv', 'bioRxiv', 'medRxiv',
                    'acknowledgement',
                    'acknowledgements', 'https', 'http', 'palabra clave']
    for tokens in sent_tokens:
        tokens = list(filter(lambda a: a not in badwords_, tokens))
        if len(tokens) > 2:
            refdoc = Doc(bank.vocabulary, words=tokens)
            try:
                wmddist = qdoc.similarity(refdoc)
                if wmddist <= distance:
                    hits.append([' '.join(tokens), wmddist])
            except:
                continue  # print(qsent,tokens)
            ## Yunchen Yang: replace number 1 with 'continue'
    return hits


def queryBySentAllPapers(bank, selpaperids, qsent, distance, verbose=False, useCosine=False):
    paperids = selpaperids  # list(bank.text.index)[pstart:pstop]
    allpaperids = list(bank.text.index)
    all_sent_tokens = [bank.sent_tokens[allpaperids.index(i)] for i in paperids]
    allhits = []
    for i in range(len(all_sent_tokens)):
        hits = queryBySentOnePaper(bank, qsent, all_sent_tokens[i], distance, verbose)
        if len(hits) > 0:
            allhits.append([paperids[i], hits])
        else:
            continue  # print(qsent,i)
        ## Yunchen Yang: replace number 1 with 'continue'
    return allhits


def scanPapersBySent(bank, selpaperids, sent_outcomes, sent_differences, sent_designs, distance=2, useCosine=False):
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
        hits = queryBySentAllPapers(bank, selpaperids, qsent, distance, verbose=False, useCosine=useCosine)
        if (len(hits) > 0):
            hits_outcomes.extend([hit[0] for hit in hits])
            rs_outcomes[qsent].extend(hits)
        print('done search ', qsent, len(hits))
    hits_outcomes = list(set(hits_outcomes))

    for qsent in sent_differences:
        qsent = cleanStr(qsent).lower().strip('.- ')
        if qsent not in rs_differences.keys():
            rs_differences[qsent] = []
        hits = queryBySentAllPapers(bank, selpaperids, qsent, distance, verbose=False, useCosine=useCosine)
        if (len(hits) > 0):
            hits_differences.extend([hit[0] for hit in hits])
            rs_differences[qsent].extend(hits)
        print('done search ', qsent, len(hits))
    hits_differences = list(set(hits_differences))

    for qsent in sent_designs:
        qsent = cleanStr(qsent).lower().strip('.- ')
        if qsent not in rs_designs.keys():
            rs_designs[qsent] = []
        hits = queryBySentAllPapers(bank, selpaperids, qsent, distance, verbose=False, useCosine=useCosine)
        if (len(hits) > 0):
            hits_designs.extend([hit[0] for hit in hits])
            rs_designs[qsent].extend(hits)
        print('done search ', qsent, len(hits))
    hits_designs = list(set(hits_designs))
    # except:
    #    print(k)
    ol1 = set(hits_outcomes).union(set(hits_differences))
    ol2 = ol1.union(set(hits_designs))
    finalhits = list(ol2)
    return finalhits, hits_outcomes, hits_differences, hits_designs, rs_outcomes, rs_differences, rs_designs