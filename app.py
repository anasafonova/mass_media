import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from googlePlugin import start_search
import pandas as pd
import plotly.graph_objs as go
from extractKeywords_clever import get_keywords

article_text = u'''Президент России Владимир Путин и президент США Дональд Трамп в Хельсинки. Архивное фото

Президент России Владимир Путин и президент США Дональд Трамп в Хельсинки

МОСКВА, 30 сен — РИА Новости. В Кремле прокомментировали возможность публикации стенограмм бесед между президентами России и США Владимиром Путиным и Дональдом Трампом.

Как пояснил журналистам пресс-секретарь российского лидера Дмитрий Песков, это "возможно только по взаимному согласию сторон".

По его словам, подобные публикации дипломатической практикой не предусматриваются.

"Поэтому если будут американцы (давать. — Прим. ред.) какие-то сигналы, будем обсуждать", — добавил пресс-секретарь.

Говоря о публикации в США расшифровки телефонного разговора Трампа с президентом Украины Владимиром Зеленским, Песков отметил, что это внутреннее дело Америки. Россия в эти процессы не вмешивается и не имеет права вмешиваться, добавил представитель Кремля.

Скандал в Вашингтоне

Ранее в СМИ появилась информация, что Трамп в разговоре с Зеленским, который состоялся в июле, в обмен на выделение помощи Киеву просил расследовать деятельность Хантера Байдена — сына бывшего вице-президента США Джо Байдена.

Байден-старший лидирует в гонке за выдвижение в кандидаты в президенты США и может стать конкурентом Трампа на выборах.

Разговор Трампа и Зеленского стал поводом для начала демократами процедуры импичмента.

Впоследствии Белый дом опубликовал стенограмму переговоров. Американский президент действительно призывал возобновить расследование дела Байдена-младшего, однако в беседе нет указаний, что Трамп просил об этом в обмен на гарантии помощи Киеву.

В то время, когда Байден-старший был вице-президентом и курировал отношения с Киевом, его сына ввели в правление крупнейшей частной газовой компании Burisma Holdings.

Согласно банковским данным, с весны 2014-го по осень 2015 года на счета американской фирмы Хантера Rosemont Seneca Partners LLC ежемесячно поступало по 166 тысяч долларов со счетов Burisma. 
'''

colors = {
    'background': '#FFFFFF',
    'text': '#111111'
}

matches_df = pd.DataFrame(columns=['Type', 'Span', 'Tokens', 'Normform', 'Block', 'Doc', 'HTML', 'AText'])
unique_df = pd.DataFrame(columns=['Type', 'Span', 'Tokens', 'Normform', 'Block', 'Frequency'])

matches = pd.DataFrame(columns=['Type', 'Span', 'Tokens', 'Normform', 'Block', 'Doc', 'HTML', 'AText'])
counts = pd.DataFrame(columns=['Block', 'PER', 'LOC', 'ORG', 'Doc', 'HTML', 'AText'])
unique = pd.DataFrame()

search_df = pd.DataFrame(columns=['Block', 'PER', 'LOC', 'ORG', 'Doc', 'HTML', 'AText'])
max_per = 0
max_loc = 0
max_org = 0
sens = []
docs = []

head = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.H1(
                    children='''ОТ. Платформа''',
                    style={
                        'textAlign': 'center',
                        'color': colors['text']
                        }
                    )
                ),
                dbc.Col(
                    html.Div(children='Поиск статей',
                    style={
                    'textAlign': 'center',
                    'color': colors['text']
                    }
                    )
                )
            ]
        )
    ]
)

analyzer = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Textarea(
                            id='search-txt',
                            rows=20,
                            placeholder='Введите текст статьи...',
                            value=article_text,
                            style={'width': '30%'}
                        ),
                        html.Button(
                            'Получить ключевые слова',
                            id='get-keywds',
                            style={'horizontal-align': 'center',
                                   'vertical-align': 'top'}
                        ),
                        dcc.Textarea(
                            readOnly=True,
                            id='keywds',
                            rows=20,
                            placeholder='Ключевые слова...',
                            value='',
                            style={'width': '30%'}
                        ),

                    ]

                )
            ]
        )
    ]
)

searchcntr = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Input(
                            readOnly=True,
                            id='search-input',
                            placeholder='',
                            type='text',
                            value='',
                            style={'width': '75%'}
                        ),
                        html.Button('Начать поиск', id='search-button', style={'vertical-align': 'top'})
                    ]
                ),
                dbc.Col(
                    [
                        html.Div(children='Количество редких PER сущностей в запросе', style={'width': '30%'}),
                        dcc.Input(
                            id='per2',
                            placeholder='Персоны',
                            type='number',
                            min=0,
                            max=max_per,
                            value=0,
                            style={'width': '30%'}
                        )
                    ]
                ),
                dbc.Col(
                    [
                        html.Div(children='Количество редких LOC сущностей в запросе', style={'width': '30%'}),
                        html.Div(children='Количество редких ORG сущностей в запросе', style={'width': '30%'})
                    ]
                ),
                dbc.Col(
                    [
                        dcc.Input(
                            id='per',
                            placeholder='Персоны',
                            type='number',
                            min=0,
                            max=max_per,
                            value=0,
                            style={'width': '30%'}
                        ),
                        dcc.Input(
                            id='loc',
                            placeholder='Места',
                            type='number',
                            min=0,
                            max=max_loc,
                            value=0,
                            style={'width': '30%'}
                        ),
                        dcc.Input(
                            id='org',
                            placeholder='Организации',
                            type='number',
                            min=0,
                            max=max_org,
                            value=0,
                            style={'width': '30%'}
                        )
                    ]
                ),
                dbc.Col(
                    dcc.Textarea(
                        readOnly=True,
                        id='article-text',
                        rows=10,
                        placeholder='Найденный текст',
                        value='',
                        style={'width': '30%'}
                    )
                )
            ]
        )
    ]
)

display = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="Doc",
                                options=[{
                                    'label': 'Result %i' % i,
                                    'value': i
                                } for i in docs],
                                value=0),
                        ],
                        style={'width': '25%',
                        'display': 'inline-block'}
                    ),
                ),
                dbc.Col(
                    dcc.Graph(
                        id='textline'
                    )
                )
            ]
        )
    ]
)

app = dash.Dash()
app.config['suppress_callback_exceptions'] = True

# Import Data
# df = pd.read_csv("search_original_q.csv")
# sens = df['Block'].tolist()
#
# search_df = pd.read_csv("search_unique.csv")
# max_per = search_df[search_df['Type'] == 'PER'].shape[0]
# max_loc = search_df[search_df['Type'] == 'LOC'].shape[0]
# max_org = search_df[search_df['Type'] == 'ORG'].shape[0]
#
# results_df = pd.DataFrame(columns=['Block', 'PER', 'LOC', 'ORG', 'Doc'])
# for i in range(0, 7):
#     df2 = pd.read_csv("%i_original_q.csv" % i)
#     df2['Doc'] = i
#     results_df = results_df.append(df2)
#
# docs = results_df["Doc"].unique()

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Article TextLines',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    html.Div(id='hidden-div', style={'display': 'none'}),
    dcc.Textarea(
        id='search-txt',
        rows=15,
        placeholder='Введите текст статьи...',
        value=article_text,
        style={'width': '30%'}
    ),
    html.Button('Получить ключевые слова', id='get-keywds', style={'vertical-align': 'top'}),
    dcc.Textarea(
        readOnly=True,
        id='keywds',
        rows=10,
        placeholder='Ключевые слова...',
        value='',
        style={'width': '30%'}
    ),
    dcc.Input(
        id='per',
        placeholder='Персоны',
        type='number',
        min=0,
        max=max_per,
        value=0
    ),
    dcc.Input(
        id='loc',
        placeholder='Места',
        type='number',
        min=0,
        max=max_loc,
        value=0
    ),
    dcc.Input(
        id='org',
        placeholder='Организации',
        type='number',
        min=0,
        max=max_org,
        value=0
    ),
    dcc.Input(
        readOnly=True,
        id='search-input',
        placeholder='',
        type='text',
        value='',
        style={'width': '50%'}
    ),
    html.Button('Начать поиск', id='search-button', style={'vertical-align': 'top'}),
    dcc.Textarea(
        readOnly=True,
        id='article-text',
        rows=10,
        placeholder='Найденный текст',
        value='',
        style={'width': '30%'}
    ),
    html.Div(
        [
            dcc.Dropdown(
                id="Doc",
                options=[{
                    'label': 'Result %i' % i,
                    'value': i
                } for i in docs],
                value=0),
        ],
        style={'width': '25%',
               'display': 'inline-block'}),
    dcc.Graph(
        id='textline'
        # figure={
        #     'data': [
        #         go.Bar(name='Person', x=sens, y=df['PER'].tolist()),
        #         go.Bar(name='Location', x=sens, y=df['LOC'].tolist()),
        #         go.Bar(name='Organization', x=sens, y=df['ORG'].tolist()),
        #         go.Bar(name='Person', x=sens2, y=[-v for v in df2['PER'].tolist()]),
        #         go.Bar(name='Location', x=sens2, y=[-v for v in df2['LOC'].tolist()]),
        #         go.Bar(name='Organization', x=sens2, y=[-v for v in df2['ORG'].tolist()])
        #     ],
        #     'layout': {
        #         'barmode': 'relative',
        #         'plot_bgcolor': colors['background'],
        #         'paper_bgcolor': colors['background'],
        #         'font': {
        #             'color': colors['text']
        #         }
        #     }
        # }
    )
])

app.layout = html.Div([head, analyzer, searchcntr, display])

@app.callback(
    dash.dependencies.Output('search-input', 'value'),
    [dash.dependencies.Input('per', 'value'),
     dash.dependencies.Input('loc', 'value'),
     dash.dependencies.Input('org', 'value')]
)
def show_search_string(per, loc, org):
    if unique_df.empty:
        return ''
    df1 = unique_df[unique_df['Type'] == 'PER'].iloc[:per]
    if per is None:
        df1 = pd.DataFrame(columns=unique_df.columns)
    df4 = unique_df[unique_df['Type'] == 'LOC'].iloc[:loc]
    if loc is None:
        df4 = pd.DataFrame(columns=unique_df.columns)
    df3 = unique_df[unique_df['Type'] == 'ORG'].iloc[:org]
    if org is None:
        df3 = pd.DataFrame(columns=unique_df.columns)
    keywords = df1['Tokens'].values.tolist() + df4['Tokens'].values.tolist() + df3['Tokens'].values.tolist()
    search_string = ' '.join(keywords)
    return search_string


@app.callback(
    dash.dependencies.Output('keywds', 'value'),
    [dash.dependencies.Input('get-keywds', 'n_clicks')],
    state=[dash.dependencies.State(component_id='search-txt', component_property='value')]
)
def get_keywds(_n_clicks, value):
    global matches_df, search_df, unique_df
    matches_df, search_df, unique_df = get_keywords(value, '', 'search')
    search_string = 'Персоны:\n'
    search_string += '\n'.join(unique_df[unique_df['Type'] == 'PER']['Tokens'].tolist())
    search_string += '\n\nМеста:\n'
    search_string += '\n'.join(unique_df[unique_df['Type'] == 'LOC']['Tokens'].tolist())
    search_string += '\n\nОрганизации:\n'
    search_string += '\n'.join(unique_df[unique_df['Type'] == 'ORG']['Tokens'].tolist())
    global sens
    sens = search_df['Block'].tolist()
    global max_per, max_loc, max_org
    max_per = matches_df[matches_df['Type'] == 'PER'].shape[0]
    max_loc = matches_df[matches_df['Type'] == 'LOC'].shape[0]
    max_org = matches_df[matches_df['Type'] == 'ORG'].shape[0]
    return search_string


@app.callback(
    dash.dependencies.Output('hidden-div', 'value'),
    [dash.dependencies.Input('search-button', 'n_clicks')],
    state=[dash.dependencies.State(component_id='search-input', component_property='value')]
)
def search(_n_clicks, search_string):
    print('Search callback')
    if (search_string != ''):
        print('Im here')
        global matches, counts, unique
        matches, counts, unique = start_search(search_string, 10)
        print('Im still here')
        global docs
        docs = counts["Doc"].unique()
        print(matches)
        idx = matches[matches['Doc'] == 0].index.tolist()
        if matches.empty:
            return ''
        return matches['AText'].loc[idx[0]]
    return ''

@app.callback(
    dash.dependencies.Output('article-text', 'value'),
    [dash.dependencies.Input('Doc', 'value'),
     dash.dependencies.Input('hidden-div', 'value')])
def update_text(doc, _val):
    idx = matches[matches['Doc'] == doc].index.tolist()
    return matches['AText'].loc[idx[0]]

@app.callback(
    dash.dependencies.Output('Doc', 'options'),
    [dash.dependencies.Input('hidden-div', 'value')]
)
def update_dropdown(_text):
    global docs
    docs = matches["Doc"].unique()
    return [{
                'label': 'Result %i' % i,
                'value': i
            } for i in docs]

@app.callback(
    dash.dependencies.Output('per', 'max'),
    [dash.dependencies.Input('keywds', 'value')]
)
def per_max(_kwds):
    global max_per
    max_per = matches_df[matches_df['Type'] == 'PER'].shape[0]
    return max_per

@app.callback(
    dash.dependencies.Output('loc', 'max'),
    [dash.dependencies.Input('keywds', 'value')]
)
def loc_max(_kwds):
    global max_loc
    max_loc = matches_df[matches_df['Type'] == 'LOC'].shape[0]
    return max_loc

@app.callback(
    dash.dependencies.Output('org', 'max'),
    [dash.dependencies.Input('keywds', 'value')]
)
def org_max(_kwds):
    global max_org
    max_org = matches_df[matches_df['Type'] == 'ORG'].shape[0]
    return max_org

@app.callback(
    dash.dependencies.Output('textline', 'figure'),
    [dash.dependencies.Input('Doc', 'value'),
     dash.dependencies.Input('hidden-div', 'value')]
)
def update_graph(doc, _val):
    # if doc == 0:
    #     df_plot = results_df.copy()
    # else:

    df_plot = counts[counts['Doc'] == doc]
    global sens
    sens = search_df['Block'].tolist()
    sens2 = df_plot['Block'].tolist()

    return {
        'data': [
            go.Bar(name='Person', x=sens, y=search_df['PER'].tolist()),
            go.Bar(name='Location', x=sens, y=search_df['LOC'].tolist()),
            go.Bar(name='Organization', x=sens, y=search_df['ORG'].tolist()),
            go.Bar(name='Person', x=sens2, y=[-v for v in df_plot['PER'].tolist()]),
            go.Bar(name='Location', x=sens2, y=[-v for v in df_plot['LOC'].tolist()]),
            go.Bar(name='Organization', x=sens2, y=[-v for v in df_plot['ORG'].tolist()])
        ],
        'layout': {
            'title': 'NERs for original article and result article',
            'barmode': 'relative',
            'plot_bgcolor': colors['background'],
            'paper_bgcolor': colors['background'],
            'font': {
                'color': colors['text']
            }
        }
    }

if __name__ == '__main__':
    app.run_server() #debug=True)
