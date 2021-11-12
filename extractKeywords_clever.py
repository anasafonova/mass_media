from newspaper import Article
import nltk
#import spacy
import pymorphy2
#from natasha import NamesExtractor
import ner
import pandas as pd
import numpy as np

def get_keywords(text, html, fname):
    extractor = ner.Extractor()
    #return ['aaaaa', 'bbbbb', 'ccccc']

    # nlp = spacy.load('ru2')
    # nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    # doc = nlp(text, regex=False)

    columns = ['Type', 'Span', 'Tokens', 'Normform', 'Block']
    matches_df = pd.DataFrame(columns=columns)
    count_df = pd.DataFrame(columns=['Block', 'PER', 'LOC', 'ORG'])

    sentences = nltk.tokenize.sent_tokenize(text)
    morph = pymorphy2.MorphAnalyzer()

    block_num = 0
    for s in sentences:
        print(s)
        counts = {'LOC': 0, 'PER': 0, 'ORG': 0}
        for m in extractor(s):
            counts[m.type] += 1
            word = ' '.join([t.text for t in m.tokens])
            p = [morph.parse(t.text)[0] for t in m.tokens]
            matches_df = matches_df.append({'Type': m.type, 'Span': m.span, 'Tokens': word,
                                            'Normform': ' '.join([pp.normal_form for pp in p]), 'Block': block_num}, ignore_index=True)

        count_df = count_df.append({'Block': block_num,
            'PER': counts['PER'], 'LOC': counts['LOC'], 'ORG': counts['ORG']}, ignore_index=True)
        block_num += 1

    freq = {}
    for nf in matches_df['Normform'].unique():
        nf_freq = 0
        for word in matches_df['Normform'].values:
            if (word == nf):
                nf_freq += 1
        freq[nf] = nf_freq

    print(freq)
    matches_df['Frequency'] = np.array([freq[nf] for nf in matches_df['Normform'].values])
    df_unique = pd.DataFrame(columns=matches_df.columns)

    for i in range(matches_df.shape[0]):
        if matches_df['Normform'].loc[i] not in df_unique['Normform'].values:
            dic = {}
            for col in matches_df.columns:
                dic[col] = matches_df[col].loc[i]
            df_unique = df_unique.append(dic, ignore_index=True)

    df_unique = df_unique.sort_values(by=['Frequency'], ascending=True)
    print(df_unique)
    df_unique.to_csv('%s_unique.csv' % fname, index=False)

    print(matches_df)
    matches_df.to_csv('%s_original.csv' % fname, index=False)

    print(count_df)
    count_df['HTML'] = html
    count_df.to_csv('%s_original_q.csv' % fname, index=False)

    df1 = df_unique[df_unique['Type'] == 'LOC']#.iloc[:3]
    df2 = df_unique[df_unique['Type'] == 'PER']#.iloc[:3]
    df3 = df_unique[df_unique['Type'] == 'ORG']#.iloc[:3]

    return matches_df, count_df, df_unique
    #({'LOC': df1['Tokens'].tolist(), 'PER': df2['Tokens'].tolist(), 'ORG': df3['Tokens'].tolist()}, matches_df)


def getKeywords(html, fname):
    article = Article(html)
    article.download()
    article.parse()
    article.nlp()
    functors_pos = {'CONJ', 'ADV-PRO', 'CCONJ', 'PART', 'PR', 'S-PRO', 'NONLEX', 'PUNCT', 'ADP',
                    'SPACE'}  # function words

    extractor = ner.Extractor()
    keywords = []
    morph = pymorphy2.MorphAnalyzer()

    print(article.text, '\n\n\n')

    nlp = spacy.load('ru2')
    nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    print("NLP pipeline: {}".format(nlp.pipe_names))
    doc = nlp(article.text)

    print('DeepPavlov:\n')

    columns = ['Type', 'Span', 'Tokens', 'Block']
    matches_df = pd.DataFrame(columns=columns)
    count_df = pd.DataFrame(columns=['Block', 'PER', 'LOC', 'ORG'])

    block_num = 0
    for s in doc.sents:
        print(s)
        counts = {'LOC': 0, 'PER': 0, 'ORG': 0}
        for m in extractor(s.string):
            #matches_df.loc[matches_df.index.max() + 1] = [m.type, m.span, m.tokens]
            counts[m.type] += 1
            matches_df = matches_df.append({'Type': m.type, 'Span': m.span, 'Tokens': ' '.join([t.text for t in m.tokens]), 'Block': block_num}, ignore_index=True)

        count_df = count_df.append({'Block': block_num,
            'PER': counts['PER'], 'LOC': counts['LOC'], 'ORG': counts['ORG']}, ignore_index=True)
        block_num += 1

    print(matches_df)
    matches_df.to_csv('%s_original.csv' % fname, index=False)

    print(count_df)
    count_df['HTML'] = html
    count_df.to_csv('%s_original_q.csv' % fname, index=False)

    df1 = matches_df[matches_df['Type'] == 'LOC'].iloc[:3]
    df2 = matches_df[matches_df['Type'] == 'PER'].iloc[:3]
    df3 = matches_df[matches_df['Type'] == 'ORG'].iloc[:3]

    return df1['Tokens'].tolist() + df2['Tokens'].tolist() + df3['Tokens'].tolist()

    # matches = {'LOC': [],
    #            'PER': [],
    #            'ORG': []
    #            }

    #for s in extractor.tokenizer(article.text):
    #    print('Token | ', s)

    # print(extractor.corpus)
    #
    # matchescnt = 0
    # for m in extractor(article.text):
    #     print(m)
    #     matches[m.type].append(m)
    #     matchescnt += 1
    #
    # print(matches)
    # print(mcnt, matchescnt)

    for m in extractor(article.text):
        print([token.text for token in m.tokens], ' | ', m.type)
        for token in m.tokens:
            cnt = 0
            t = morph.parse(token.text)[0]
            for word in keywords:
                p = morph.parse(word)[0]
                if p.normal_form == t.normal_form:
                    break
                else:
                    cnt += 1
            if (cnt == len(keywords)):
                keywords.append(token.text)

    print(' '.join(keywords))
    print('\n\n\nSpaCy:\n')
#    return keywords

    nlp = spacy.load('ru2')
    nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    print("NLP pipeline: {}".format(nlp.pipe_names))
    doc = nlp(article.text)
 #   for s in doc.sents:
 #       for t in s:
 #           if t.pos_ not in functors_pos:
#                print(type(t.tag_), t.tag_)
#                print('lemma "{}" with pos "{}" and tag "{}"'.format(t.lemma_, t.pos_, t.tag_[t.tag_.find(
 #                   'Case=') + 5:t.tag_.find('Case=') + 8]))
                #allwords += t.lemma_ + ' '
        #print(list(['lemma "{}" with pos "{}" and tag "{}"'.format(t.lemma_, t.pos_, t.tag_[t.tag_.find('Case=') + 5:t.tag_.find('Case=') + 8]) for t in s]))
        #    allwords.append(t.lemma_)

    #doc2 = nlp(allwords)

    for entity in doc.ents:
        print(entity.label_, ' | ', entity.text)

    print('\n\n\nNatasha:\n')
    extractor = NamesExtractor()
    matches = extractor(article.text)
    for match in matches:
        print(match.span, match.fact)

    return keywords

    #return [entity.text for entity in doc.ents]

    allwords = nltk.word_tokenize(article.text)
    morph = pymorphy2.MorphAnalyzer()
    normwords = []
    for word in allwords:
        p = morph.parse(word)[0]
        normwords.append(p.normal_form)

    #print(*[word for word, pos in nltk.pos_tag(article.keywords, lang='rus')
    #        if pos not in functors_pos])
    #for word, pos in nltk.pos_tag(normwords, lang='rus'):
    #    print(word,pos)

    words = [word for word, pos in nltk.pos_tag(normwords, lang='rus')
                if pos not in functors_pos]
    freq = {}
    for word in words:
        if word in freq:
            freq[word] += 1
        else:
            freq[word] = 1
    min_size = min(7, len(freq))
    keywords = sorted(freq.items(),
                      key=lambda x: (x[1], x[0]),
                      reverse=True)
    keywords = keywords[:7] #+ keywords[int(len(freq)/2):min_size + int(len(freq)/2)]
    keywords = dict((x, y) for x, y in keywords)
    keywords_form = []

    for k in keywords:
        articleScore = keywords[k] * 1.0 / max(len(words), 1)
        keywords[k] = articleScore * 1.5 + 1

    for k in list(keywords.keys()):
        for word in allwords:
            p = morph.parse(word)[0]
            if k == p.normal_form:
                keywords_form.append(word)
                break

    return [entity.text for entity in doc.ents[:3]] + keywords_form #list(keywords.keys())

def main():
    kw = getKeywords('https://ria.ru/20190930/1559283116.html')
    article = Article('https://ria.ru/20190930/1559283116.html')
    article.download()
    article.parse()
    article.nlp()
    words = nltk.word_tokenize(article.text)
    #functors_pos = {'CONJ', 'ADV-PRO', 'CONJ', 'PART', 'PR', 'S-PRO'}  # function words
    #print(*[word for word, pos in nltk.pos_tag(kw, lang='rus')
    #        if pos not in functors_pos])
    print(nltk.pos_tag(words, lang='rus'))

#main()