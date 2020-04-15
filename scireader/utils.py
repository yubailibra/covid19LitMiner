import collections
import re
import numpy as np


badpattern = re.compile('[%"]+')

Author = collections.namedtuple('Author', 'first middle last')
Citation = collections.namedtuple('Citation', 'title authors year venue volume issn pages other_ids')


def FileExtractor(text):
    text_dict = {}
    for section in ['paper_id', 'metadata', 'abstract', 'body_text', 'bib_entries']:  # "ref_entries",'back_matter'
        if section in text.keys():
            if section == 'paper_id':
                text_dict['paper_id'] = extractPaperId(text)
            if section == 'metadata':
                text_dict['title'] = extractTitle(text)
                text_dict['authors'] = extractAuthors(text)
            if section == 'abstract':
                text_dict['abstract'] = extractAbstract(text)
            if section == 'body_text':
                text_dict['body_text'] = extractBodyText(text)
            if section == 'bib_entries':
                text_dict['bib_entries'] = extractBibs(text)
    return text_dict


def extractPaperId(text):
    if "paper_id" in text.keys() and len(text["paper_id"]) > 0:
        return text['paper_id'].strip()
    else:
        return ''


def extractTitle(text):
    if "metadata" in text.keys() and 'title' in text['metadata'].keys() and len(text["metadata"]['title']) > 0:
        return badpattern.sub('', text['metadata']['title']).strip()
    else:
        return ''


def extractAuthors(text):
    if "metadata" in text.keys() and 'authors' in text['metadata'].keys() and len(text["metadata"]['authors']) > 0:
        authors = []
        for each in text['metadata']['authors']:
            first = middle = last = ''
            if len(each['first']) > 0:
                first = badpattern.sub('', each['first'].lower()).strip()
            if len(each['middle']) > 0:
                middle = badpattern.sub('', each['middle'][0].lower()).strip()
            if len(each['last']) > 0:
                last = badpattern.sub('', each['last'].lower()).strip()
            authors.append(Author(first=first, middle=middle, last=last))
        return authors
    elif 'authors' in text.keys() and len(text['authors']) > 0:
        authors = []
        for each in text['authors']:
            first = middle = last = ''
            if len(each['first']) > 0:
                first = badpattern.sub('', each['first'].lower()).strip()
            if len(each['middle']) > 0:
                middle = badpattern.sub('', each['middle'][0].lower()).strip()
            if len(each['last']) > 0:
                last = badpattern.sub('', each['last'].lower()).strip()
            authors.append(Author(first=first, middle=middle, last=last))
        return authors
    else:
        return []


def extractAbstract(text):
    abstract = ''
    if "abstract" in text.keys() and len(text["abstract"]) > 0:
        abstract = ' '.join([re.compile('[%"]+').sub('', entry['text'].strip()) for entry in text['abstract']])
        abstract = re.sub('\s+[0-9]+(\s+[0-9]+)*\s+', ' ', abstract).strip()
    return abstract


def extractBodyText(text):
    body = ''
    if "body_text" in text.keys() and len(text["body_text"]) > 0:
        body = ' '.join([re.compile('[%"]+').sub('', entry['text'].strip()) for entry in text['body_text']])
        body = re.sub('\s+[0-9]+(\s+[0-9]+)*\s+', ' ', body).strip()
    return body


def extractBibs(text):
    if "bib_entries" in text.keys() and len(text["bib_entries"]) > 0:
        allbib = []
        for each in text['bib_entries']:
            entry = text['bib_entries'][each]
            title = year = venue = volume = issn = pages = ''
            authors = []
            other_ids = {}
            if len(entry['title']) > 0:
                title = entry['title'].strip('"')
            if len(entry['authors']) > 0:
                authors = extractAuthors(entry)
            if entry['year'] != '' and entry['year'] != None:
                # print(entry['year'])
                year = entry['year']
            if len(entry['venue']) > 0:
                venue = badpattern.sub('', entry['venue'])
            if len(entry['volume']) > 0:
                volume = badpattern.sub('', entry['volume'])
            if len(entry['issn']) > 0:
                issn = badpattern.sub('', entry['issn'])
            if len(entry['pages']) > 0:
                pages = badpattern.sub('', entry['pages'])
            if len(entry['other_ids']) > 0:
                for eachtype in entry['other_ids'].keys():
                    other_ids[eachtype] = []
                    for each in entry['other_ids'][eachtype]:
                        other_ids[eachtype].append(each.strip('"'))
            allbib.append(
                Citation(title=title, authors=authors, year=year, venue=venue, volume=volume, issn=issn, pages=pages,
                         other_ids=other_ids))
        return allbib
    else:
        return []


def concatAuthors(authorlist):
    authors_str = ""
    for each in authorlist:
        authors_str = authors_str + each.first + ":" + each.middle + ":" + each.last + ", "
    return authors_str


def ent2vector(text, nlp):
    nes2vec = []
    vec = []
    try:
        doc2 = nlp((text))
        nes2vec = [[re.sub('[\(\)]', '', cleanStr(token.lemma_.lower())).strip('.- '), token.vector] for token in
                   doc2.ents if (re.search('[a-zA-Z]', token.lemma_.lower()) is not None) and len(
                re.sub('[\(\)]', '', cleanStr(token.lemma_.lower())).strip('.- ')) > 0]
        abrvs3 = [[cleanStr(token.lemma_.lower()).strip('.- '), token.vector] for token in doc2._.abbreviations if
                  (")" not in token.lemma_.lower()) and ("(" not in token.lemma_.lower()) and (
                              re.search('[a-zA-Z]', token.lemma_.lower()) is not None) and len(
                      cleanStr(token.lemma_.lower()).strip('.- ')) > 0]
        nes2vec.extend(abrvs3)
        currentWords = set([each[0] for each in nes2vec])
        abrvs2 = [[cleanStr(token._.long_form.lemma_.lower()).strip('.- '), token._.long_form.vector] for token in
                  doc2._.abbreviations if (token._.long_form.lemma_.lower() not in currentWords) and (
                              ")" not in token._.long_form.lemma_.lower()) and (
                              "(" not in token._.long_form.lemma_.lower()) \
                  and (re.search('[a-zA-Z]', token.lemma_.lower()) is not None) and (
                              re.search('\s+', token.lemma_.lower().strip()) is None) and (
                              re.search(token._.long_form.lemma_.lower().strip('.- '),
                                        text.lower()) is not None) and len(
                cleanStr(token._.long_form.lemma_.lower()).strip('.- ')) > 0]  #
        terms = [abrvs2[i][0] for i in range(len(abrvs2))]
        keepIndex = [terms.index(uniq) for uniq in set(terms)]
        [abrvs2[i] for i in keepIndex]
        nes2vec.extend([abrvs2[i] for i in keepIndex])
    except:
        print('cannot parse text:', text[:min(20, len(text))])
    return nes2vec


def enttoken2vector(text, nlp):
    nes2vec = []
    sent_tokens = []
    try:
        doc = nlp((text))

        abrvs3 = [[cleanStr(token.lemma_.lower()).strip('.- '), token.vector] for token in doc._.abbreviations if
                  (")" not in token.lemma_.lower()) and ("(" not in token.lemma_.lower()) and (
                              re.search('[a-zA-Z]', token.lemma_.lower()) is not None) and len(
                      cleanStr(token.lemma_.lower()).strip('.- ')) > 0]
        nes2vec.extend(abrvs3)
        currentWords = set([each[0] for each in nes2vec])

        ne2vec = [[re.sub('[\(\)]', '', cleanStr(token.lemma_.lower())).strip('.- '), token.vector] for token in
                  doc.ents if (re.search('[a-zA-Z]', token.lemma_.lower()) is not None) and len(
                re.sub('[\(\)]', '', cleanStr(token.lemma_.lower())).strip('.- ')) > 0 and re.sub('[\(\)]', '',
                                                                                                  cleanStr(
                                                                                                      token.lemma_.lower())).strip(
                '.- ') not in currentWords]
        nes2vec.extend(ne2vec)
        currentWords = set([each[0] for each in nes2vec])
        ## add in tokens
        sent_tokens = []
        for sent in doc.sents:
            tokens_per_sent = [re.sub('[\(\)]', '', cleanStr(token.lemma_.lower())).strip('.- ') for token in sent if
                               (token.pos_ not in ('PUNCT', 'NUM', 'SYM')) and (not token.is_bracket) and (
                                   not token.is_left_punct) and (not token.is_oov) and (not token.is_punct) and \
                               (not token.is_quote) and (not token.is_right_punct) and (not token.is_space) and (
                                   not token.is_stop) and (re.search('[a-zA-Z]', token.lemma_.lower()) is not None) and \
                               len(re.sub('[\(\)]', '', cleanStr(token.lemma_.lower()).strip('.- '))) > 1]
            if len(tokens_per_sent) > 0:
                sent_tokens.append(tokens_per_sent)
        token2vec = [[re.sub('[\(\)]', '', cleanStr(token.lemma_.lower())).strip('.- '), token.vector] for token in doc
                     if (token.pos_ not in ('PUNCT', 'NUM', 'SYM')) and (not token.is_bracket) and (
                         not token.is_left_punct) and (not token.is_oov) and (not token.is_punct) and \
                     (not token.is_quote) and (not token.is_right_punct) and (not token.is_space) and (
                         not token.is_stop) and (re.search('[a-zA-Z]', token.lemma_.lower()) is not None) and \
                     len(re.sub('[\(\)]', '', cleanStr(token.lemma_.lower()).strip('.- '))) > 1 and re.sub('[\(\)]', '',
                                                                                                           cleanStr(
                                                                                                               token.lemma_.lower())).strip(
                '.- ') not in currentWords]
        nes2vec.extend(token2vec)
        currentWords = set([each[0] for each in nes2vec])

        abrvs2 = [[cleanStr(token._.long_form.lemma_.lower()).strip('.- '), token._.long_form.vector] for token in
                  doc._.abbreviations if
                  (cleanStr(token._.long_form.lemma_.lower()).strip('.- ') not in currentWords) and (
                              ")" not in token._.long_form.lemma_.lower()) and (
                              "(" not in token._.long_form.lemma_.lower()) \
                  and (re.search('[a-zA-Z]', token.lemma_.lower()) is not None) and (
                              re.search('\s+', token.lemma_.lower().strip()) is None) and (
                              re.search(token._.long_form.lemma_.lower().strip('.- '),
                                        text.lower()) is not None) and len(
                      cleanStr(token._.long_form.lemma_.lower()).strip('.- ')) > 0]  #
        terms = [abrvs2[i][0] for i in range(len(abrvs2))]
        keepIndex = [terms.index(uniq) for uniq in set(terms)]
        # [abrvs2[i] for i in keepIndex]
        nes2vec.extend([abrvs2[i] for i in keepIndex])

    except:
        print('cannot parse text:', text[:min(20, len(text))])
    return nes2vec, sent_tokens


def entRelaxToken2vector(text, nlp):
    nes2vec = []
    sent_tokens = []
    try:
        doc = nlp((text))

        abrvs3 = [[cleanStr(token.lemma_.lower()).strip('.- '), token.vector] for token in doc._.abbreviations if
                  (")" not in token.lemma_.lower()) and ("(" not in token.lemma_.lower()) and (
                              re.search('[a-zA-Z]', token.lemma_.lower()) is not None) and len(
                      cleanStr(token.lemma_.lower()).strip('.- ')) > 0]
        nes2vec.extend(abrvs3)
        currentWords = set([each[0] for each in nes2vec])

        ne2vec = [[re.sub('[\(\)]', '', cleanStr(token.lemma_.lower())).strip('.- '), token.vector] for token in
                  doc.ents if (re.search('[a-zA-Z]', token.lemma_.lower()) is not None) and len(
                re.sub('[\(\)]', '', cleanStr(token.lemma_.lower())).strip('.- ')) > 0 and re.sub('[\(\)]', '',
                                                                                                  cleanStr(
                                                                                                      token.lemma_.lower())).strip(
                '.- ') not in currentWords]
        nes2vec.extend(ne2vec)
        currentWords = set([each[0] for each in nes2vec])
        ## add in tokens
        sent_tokens = []
        for sent in doc.sents:  # and (not token.is_stop)
            tokens_per_sent = [re.sub('[\(\)]', '', cleanStr(token.lemma_.lower())).strip('.- ') for token in sent if
                               (token.pos_ not in ('PUNCT', 'NUM', 'SYM')) and (not token.is_bracket) and (
                                   not token.is_left_punct) and (not token.is_oov) and (not token.is_punct) and \
                               (not token.is_quote) and (not token.is_right_punct) and (not token.is_space) and (
                                           re.search('[a-zA-Z]', token.lemma_.lower()) is not None) and \
                               len(re.sub('[\(\)]', '', cleanStr(token.lemma_.lower()).strip('.- '))) > 1]
            if len(tokens_per_sent) > 0:
                sent_tokens.append(tokens_per_sent)
        token2vec = [[re.sub('[\(\)]', '', cleanStr(token.lemma_.lower())).strip('.- '), token.vector] for token in doc
                     if (token.pos_ not in ('PUNCT', 'NUM', 'SYM')) and (not token.is_bracket) and (
                         not token.is_left_punct) and (not token.is_oov) and (not token.is_punct) and \
                     (not token.is_quote) and (not token.is_right_punct) and (not token.is_space) and (
                                 re.search('[a-zA-Z]', token.lemma_.lower()) is not None) and \
                     len(re.sub('[\(\)]', '', cleanStr(token.lemma_.lower()).strip('.- '))) > 1 and re.sub('[\(\)]', '',
                                                                                                           cleanStr(
                                                                                                               token.lemma_.lower())).strip(
                '.- ') not in currentWords]
        nes2vec.extend(token2vec)
        currentWords = set([each[0] for each in nes2vec])

        abrvs2 = [[cleanStr(token._.long_form.lemma_.lower()).strip('.- '), token._.long_form.vector] for token in
                  doc._.abbreviations if
                  (cleanStr(token._.long_form.lemma_.lower()).strip('.- ') not in currentWords) and (
                              ")" not in token._.long_form.lemma_.lower()) and (
                              "(" not in token._.long_form.lemma_.lower()) \
                  and (re.search('[a-zA-Z]', token.lemma_.lower()) is not None) and (
                              re.search('\s+', token.lemma_.lower().strip()) is None) and (
                              re.search(token._.long_form.lemma_.lower().strip('.- '),
                                        text.lower()) is not None) and len(
                      cleanStr(token._.long_form.lemma_.lower()).strip('.- ')) > 0]  #
        terms = [abrvs2[i][0] for i in range(len(abrvs2))]
        keepIndex = [terms.index(uniq) for uniq in set(terms)]
        # [abrvs2[i] for i in keepIndex]
        nes2vec.extend([abrvs2[i] for i in keepIndex])

    except:
        print('cannot parse text:', text[:min(20, len(text))])
    return nes2vec, sent_tokens


def my_sentence_analyzer(text, nlp):
    sentences = []
    sent_docs = []
    try:
        doc = nlp(text)
        sentences = [' '.join([re.sub('[\(\)]', '', cleanStr(token.lemma_.lower())).strip('.- ') for token in sent if
                               (token.pos_ not in ('PUNCT', 'NUM', 'SYM')) and (not token.is_bracket) and (
                                   not token.is_left_punct) and (not token.is_oov) and (not token.is_punct) and \
                               (not token.is_quote) and (not token.is_right_punct) and (not token.is_space) and (
                                   not token.is_stop) and (re.search('[a-zA-Z]', token.lemma_.lower()) is not None) and \
                               len(re.sub('[\(\)]', '', cleanStr(token.lemma_.lower()).strip('.- '))) > 1]) for sent in
                     doc.sents]
    except:
        print('cannot tokenize text:', text[:min(20, len(text))])
    return sentences


def my_tokenBySent_analyzer(text, nlp):
    sent_tokens = []
    try:
        doc = nlp(text)
        sent_tokens = [[re.sub('[\(\)]', '', cleanStr(token.lemma_.lower())).strip('.- ') for token in sent if
                        (token.pos_ not in ('PUNCT', 'NUM', 'SYM')) and (not token.is_bracket) and (
                            not token.is_left_punct) and (not token.is_oov) and (not token.is_punct) and \
                        (not token.is_quote) and (not token.is_right_punct) and (not token.is_space) and (
                            not token.is_stop) and (re.search('[a-zA-Z]', token.lemma_.lower()) is not None) and \
                        len(re.sub('[\(\)]', '', cleanStr(token.lemma_.lower()).strip('.- '))) > 1] for sent in
                       doc.sents]

    except:
        print('cannot tokenize text:', text[:min(20, len(text))])
    return sent_tokens

#todo: fix https/web site, copyright, preprint, bioaXiv staff, acknowledge etc


def cleanStr(text):
    cleaned = re.sub('[^a-zA-z0-9\s\.\-/\(\)]', '', re.sub('\d+[\.\d\s]*-[\s\d\.]*', '',
                                                           re.sub(r'\s+[\-\+]\d+[\.\d]*\b', '',
                                                                  re.sub('\d+[\.\d\s-]*kda', '',
                                                                         re.sub('\d+[\.\d\s-]*bp', '',
                                                                                re.sub('\d+[\.\d\s-]*kbp', '',
                                                                                       re.sub('/', ' ',
                                                                                              re.sub('\s\s', ' ',
                                                                                                     re.sub('\(', ' (',
                                                                                                            re.sub(
                                                                                                                '(copy.*right|preprint|\\bbiorxiv\\b|\\bbioRxiv\\b|\\bmedrxiv\\b|\\bmedRxiv\\b|acknowledgement|palabra.*clave|\\bhttps\\b|\\bhttp\\b)',
                                                                                                                '',
                                                                                                                text)))))))))).strip()
    return cleaned


def getEnt(text, nlp):
    nes = []
    try:
        doc2 = nlp((text))
        nes = [re.sub('[\(\)]', '', cleanStr(token.lemma_.lower())).strip('.- ') for token in doc2.ents if
               (re.search('[a-zA-Z]', token.lemma_.lower()) is not None) and len(
                   re.sub('[\(\)]', '', cleanStr(token.lemma_.lower())).strip('.- ')) > 0]
    except:
        print('cannot parse text:', text[:min(20, len(text))])
    return nes


def my_token_analyzer(text, nlp):  # re.sub('[<{\(\[]\s*\d+[\.\d/]*\s*[\]\)}>]','',
    tokens = []
    try:
        doc = nlp(text)
        # nlp(re.sub('[^a-zA-z0-9\s\.\-/]','',re.sub('\d+[\.\d\s]*-[\s\d\.]*','',re.sub(r'\s+[\-\+]\d+[\.\d]*\b','',re.sub('\d+[\.\d\s-]*kda','',re.sub('\d+[\.\d\s-]*bp','',re.sub('\d+[\.\d\s-]*kbp','',re.sub('\(',' (',text))))))).strip())
        tokens = [re.sub('[\(\)]', '', cleanStr(token.lemma_.lower())).strip('.- ') for token in doc if
                  (token.pos_ not in ('PUNCT', 'NUM', 'SYM')) and (not token.is_bracket) and (
                      not token.is_left_punct) and (not token.is_oov) and (not token.is_punct) and \
                  (not token.is_quote) and (not token.is_right_punct) and (not token.is_space) and (
                      not token.is_stop) and (re.search('[a-zA-Z]', token.lemma_.lower()) is not None) and \
                  len(re.sub('[\(\)]', '', cleanStr(token.lemma_.lower()).strip('.- '))) > 1]
    except:
        print('cannot tokenize text:', text[:min(20, len(text))])
    return tokens


def similarity_sort(index_list):
    iter_index = list(set([item for sublist in index_list for item in list(set(sublist))]))
    locate_mat = np.negative(np.ones(len(index_list)))
    indexing = dict()
    for idx in iter_index:
        indexing[idx] = []
    for i, idx_pair in enumerate(index_list):
        for idx_ in idx_pair:
            indexing[idx_].append(i)
    indexing_tuple = sorted(indexing.items(), key=lambda item: item[1])

    for i in range(len(indexing_tuple)):
        if len(indexing_tuple[i][1]) == 1:
            if locate_mat[indexing_tuple[i][1][0]] == -1:
                locate_mat[indexing_tuple[i][1][0]] = indexing_tuple[i][1][0]
        else:
            min_val = min(indexing_tuple[i][1])
            tem = [locate_mat[i_] for i_ in indexing_tuple[i][1]]
            tem_clean = [n for n in tem if n != -1]
            tem_id_clean = [ii for ii in indexing_tuple[i][1] if locate_mat[ii] != -1]
            if len(list(set(tem))) == 1 and list(set(tem))[0] == -1:
                locate_mat[indexing_tuple[i][1]] = min_val
            else:
                min_val = min(min(tem_clean), min(tem_id_clean))
                for iii in range(len(index_list)):
                    if locate_mat[iii] in tem_clean:
                        locate_mat[iii] = min_val

                locate_mat[indexing_tuple[i][1]] = min_val

    sort_list = []
    group_index = list(set(list(locate_mat)))
    for group_ct, group_id in enumerate(group_index):
        ids = np.where(locate_mat == group_id)[0]
        sort_list.append(list(set([item_ for id_ in ids for item_ in index_list[id_]])))
    return sort_list
