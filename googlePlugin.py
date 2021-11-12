from google import google
from getKwds import get_all
from newspaper import Article
import pandas as pd

#keywords = getKeywords('https://ria.ru/20190930/1559283116.html', 'org')
#print(keywords)
#search_string = ' '.join(keywords)
#print('Search:', search_string)

def start_search(search_string, g_max):
    num_page = g_max #math.ceil(g_max / 10.)
    search_results = google.search(search_string, num_page, lang='ru')
    print(search_results)
    all_matches = pd.DataFrame()
    g_num = 0
    for result in search_results:
        print(result.link)
        if (type(result.link) != None) and (result.link != ''):
            try:
                article = Article(result.link)
                article.download()
                article.parse()
                article.nlp()
                matches_df = get_all(article.text, result.link)
                matches_df['Doc'] = g_num
                matches_df['HREF'] = result.link
                matches_df['AText'] = article.text
                all_matches = all_matches.append(matches_df, ignore_index=True)
                g_num += 1
            except:
                pass
        if g_num == g_max:
            break

    return all_matches



# def aaa:
#
#     for result in search_results:
#         if (type(result.link) != None) and (result.link != ''):
#             print('DOC', len(google_list))
#             print(result.link)
#             print('\n\n\n')
#             try:
#                 google_res = getData(result.link).text
#                 if (len(google_res.strip()) != 0):
#                     google_dic[result.link] = google_res
#                     google_list.append(google_dic[result.link])
#         except:
#             #print('Cannot download data')
#             pass
#
# tf_idf_total = []
# terms = build_terms(google_list)
#
# for document in google_list:
#     tf_idf_total.append(build_tfidf(google_list, document, terms))
#
# #print(terms.keys())
# #for doc_rating in tf_idf_total:
# #    print(doc_rating)
#
# query = getData('https://ria.ru/20190930/1559283116.html').text
# #print("QUERY:", query)
# query_tfidf = build_tfidf(google_list, query, terms)
# similarity_dic = {}
# for index, document in enumerate(tf_idf_total):
#     similarity_dic[index] = cosine_similarity(query_tfidf, document)
#     #print("Similarity with DOC", index, "=", cosine_similarity(query_tfidf, document))
#
# similarity = sorted(similarity_dic.items(),
#                   key=lambda x: (x[1], x[0]),
#                   reverse=True)
# similarity = dict((x, y) for x, y in similarity)
#
# for key in similarity.keys():
#     print("Similarity with DOC", ("%02d" % key), "=", similarity[key])
#
# #getKeywords(google_list[0], 'org2')
#
#
