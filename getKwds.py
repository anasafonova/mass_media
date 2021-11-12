import nltk
import pymorphy2
import ner
import pandas as pd
import numpy as np

from deeppavlov import configs, build_model


ner_model = build_model(configs.ner.ner_rus)

ner_model([['Example', 'sentence']])
ner_model(['Example sentence'])


def get_all(text, href):
    extractor = ner.Extractor()
    columns = ['Type', 'Span', 'Tokens', 'Normform', 'Block']
    matches_df = pd.DataFrame(columns=columns)

    sentences = nltk.tokenize.sent_tokenize(text)
    morph = pymorphy2.MorphAnalyzer()

    block_num = 0
    counts_list = []
    for s in sentences:
        #print(s)
        counts = {'LOC': 0, 'PER': 0, 'ORG': 0}
        for m in extractor(s):
            counts[m.type] += 1
            word = ' '.join([t.text for t in m.tokens])
            p = [morph.parse(t.text)[0] for t in m.tokens]
            matches_df = matches_df.append({'Type': m.type, 'Span': m.span, 'Tokens': word,
                                            'Normform': ' '.join([pp.normal_form for pp in p]), 'Block': block_num},
                                           ignore_index=True)

        counts_list.append(counts)
        block_num += 1

    per = []
    loc = []
    org = []
    for dic in counts_list:
        summ = dic['PER'] + dic['LOC'] + dic['ORG']
        for i in range(0, summ):
            per.append(dic['PER'])
            loc.append(dic['LOC'])
            org.append(dic['ORG'])

    matches_df['PER'] = per
    matches_df['LOC'] = loc
    matches_df['ORG'] = org

    freq = {}
    for nf in matches_df['Normform'].unique():
        nf_freq = 0
        for word in matches_df['Normform'].values:
            if (word == nf):
                nf_freq += 1
        freq[nf] = nf_freq

    #print(freq)
    matches_df['Frequency'] = np.array([freq[nf] for nf in matches_df['Normform'].values])
    matches_df['HREF'] = href
    matches_df['AText'] = text
    #print(matches_df)
    return matches_df

def extract_unique(matches_df):
    df_unique = pd.DataFrame(columns=matches_df.columns)

    for i in range(matches_df.shape[0]):
        if matches_df['Normform'].loc[i] not in df_unique['Normform'].values:
            dic = {}
            for col in matches_df.columns:
                dic[col] = matches_df[col].loc[i]
            df_unique = df_unique.append(dic, ignore_index=True)

    df_unique = df_unique.sort_values(by=['Frequency'], ascending=True)
    #print(df_unique)
    return df_unique


df = get_all('Президент России Владимир Путин и президент США Дональд Трамп в Хельсинки. Архивное фото', '')
extract_unique(df)



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
        #print(s)
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

    #print(freq)
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