from newspaper import Article


def getData(html):
    article = Article(html, language='ru')
    try:
        article.download()
        article.parse()
        article.nlp()
    except:
        pass
    return article


def main():
    article = getData('https://ria.ru/20190930/1559283116.html')
    print(article.text)

# main()
