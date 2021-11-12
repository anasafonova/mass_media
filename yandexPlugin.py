import yandex_search
from extractKeywords import getKeywords
from compareArticles import build_terms, build_tfidf, cosine_similarity
from parseArticle import getData

keywords = getKeywords('https://ria.ru/20190930/1559283116.html')
#print(keywords)
search_string = ' '.join(keywords)
print('Search:', search_string)

yandex = yandex_search.Yandex(api_user='asalsalsalt', api_key='03.678339024:53ead9d60e0be4c19813cdee1518f882')
search_results = yandex.search(search_string).items
yandex_dic = {}
yandex_list = []
for result in search_results[:30]:
    if (type(result['url']) != None) and (result['url'] != ''):
        print('DOC', len(yandex_list))
        print(result['url'])
        print('\n\n\n')
        try:
            yandex_res = getData(result['url']).text
            if (len(yandex_res.strip()) != 0):
                yandex_dic[result['url']] = yandex_res
                yandex_list.append(yandex_dic[result['url']])
        except:
            pass

tf_idf_total = []
terms = build_terms(yandex_list)

for document in yandex_list:
    tf_idf_total.append(build_tfidf(yandex_list, document, terms))

# print(terms.keys())
# for doc_rating in tf_idf_total:
#    print(doc_rating)

query = getData('https://ria.ru/20190930/1559283116.html').text
# print("QUERY:", query)
query_tfidf = build_tfidf(yandex_list, query, terms)
similarity_dic = {}
for index, document in enumerate(tf_idf_total):
    similarity_dic[index] = cosine_similarity(query_tfidf, document)
    # print("Similarity with DOC", index, "=", cosine_similarity(query_tfidf, document))

similarity = sorted(similarity_dic.items(),
                    key=lambda x: (x[1], x[0]),
                    reverse=True)
similarity = dict((x, y) for x, y in similarity)

for key in similarity.keys():
    print("Similarity with DOC", ("%02d" % key), "=", similarity[key])