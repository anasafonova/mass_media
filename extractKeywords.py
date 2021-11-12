from newspaper import Article
import nltk
import pymorphy2


def getKeywords(html):
    article = Article(html)
    article.download()
    article.parse()
    article.nlp()
    print(article.keywords)
    allwords = nltk.word_tokenize(article.text)
    morph = pymorphy2.MorphAnalyzer()
    normwords = []
    for word in allwords:
        p = morph.parse(word)[0]
        normwords.append(p.normal_form)
    functors_pos = {'CONJ', 'ADV-PRO', 'CONJ', 'PART', 'PR', 'S-PRO', 'NONLEX'}  # function words
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
    keywords = keywords[:3] + keywords[int(len(freq)/2):min_size + int(len(freq)/2)]
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

    return keywords_form #list(keywords.keys())

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