import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from googlePlugin import start_search
import pandas as pd
import plotly.graph_objs as go
from getKwds import get_all, extract_unique

article_text = u'''Президент России Владимир Путин и президент США Дональд Трамп в Хельсинки. Архивное фото'''

article_text2 = u'''Президент России Владимир Путин и президент США Дональд Трамп в Хельсинки. Архивное фото

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

table_header = [
    html.Thead(html.Tr([html.Th("Ключевое слово"), html.Th("Тип")]))
]

row1 = html.Tr([html.Td("Arthur")])
row2 = html.Tr([html.Td("Ford")])
row3 = html.Tr([html.Td("Zaphod")])
row4 = html.Tr([html.Td("Astra")])

table_body = [html.Tbody([])]

navbar = dbc.NavbarSimple(
    #children=[
    #    dbc.NavItem(dbc.NavLink("ОТ.Платформа", href="#")),
        # dbc.DropdownMenu(
        #     nav=True,
        #     in_navbar=True,
        #     label="Menu",
        #     children=[
        #         dbc.DropdownMenuItem("Entry 1"),
        #         dbc.DropdownMenuItem("Entry 2"),
        #         dbc.DropdownMenuItem(divider=True),
        #         dbc.DropdownMenuItem("Entry 3"),
        #     ],
        # ),
    #],
    brand="ОТ.Платформа",
    brand_href="#",
    sticky="top",
    color="primary"
)

body = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Br(),
                        html.H2("Поиск новости"),
                        dbc.Textarea(id='article_textarea', className="mb-3", bs_size="lg",
                                     placeholder="Введите текст новости...", value=article_text2),
                        html.Br(),
                        dbc.Button("Получить ключевые слова", color="primary", id='keywords_button'),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        dbc.Table(table_header + table_body, bordered=True, id='keywords_table', responsive=True, striped=True)
                    ]
                ),
            ]
        ),
        dbc.Row(
            [
                html.Div(id='hidden_in', style={'display': 'none'}),
                html.Div(id='hidden_out', style={'display': 'none'})
            ]
        )
    ]
)

search_layer = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                dbc.InputGroup(
                                    [dbc.InputGroupAddon("Желаемое количество результатов", addon_type="prepend"),
                                     dbc.Input(id='res', type="number", value=1, min=1, step=1)],
                                ),
                                dbc.InputGroup(
                                    [dbc.InputGroupAddon("PER", addon_type="prepend"),
                                     dbc.Input(id='per', type="number", min=0, max=0, step=1)],
                                ),
                                html.Br(),
                                dbc.InputGroup(
                                    [dbc.InputGroupAddon("LOC", addon_type="prepend"),
                                     dbc.Input(id='loc', type="number", min=0, max=0, step=1)]
                                ),
                                html.Br(),
                                dbc.InputGroup(
                                    [dbc.InputGroupAddon("ORG", addon_type="prepend"),
                                     dbc.Input(id='org', type="number", min=0, max=0, step=1)],
                                ),
                            ]
                        )
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.P("Поисковый запрос"),
                                dbc.Input(id='search_input', type="text", placeholder='Текст поискового запроса...', disabled=True),
                            ],
                            id="styled-text-input",
                        ),
                        dbc.Button("Поиск", color="primary", id='search_button')
                    ]
                ),
            ]
        ),
    ]

)

display = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.ListGroup(
                            [
                                dbc.ListGroupItem(
                                    [
                                        dbc.ListGroupItemHeading("", id='orig_list'),
                                        dbc.ListGroupItemText("Число уникальных NER'ов в оригинальной статье"),
                                    ],
                                    color="primary"
                                ),
                                dbc.ListGroupItem(
                                    [
                                        dbc.ListGroupItemHeading("", id='new_list'),
                                        dbc.ListGroupItemText("Число уникальных NER'ов в найденной статье"),
                                    ],
                                    color="secondary"
                                ),
                                dbc.ListGroupItem(
                                    [
                                        dbc.ListGroupItemHeading("", id='add_list'),
                                        dbc.ListGroupItemText("Число привнесённых NER'ов"),
                                    ],
                                    color="primary"
                                ),
                                dbc.ListGroupItem(
                                    [
                                        dbc.ListGroupItemHeading("", id='del_list'),
                                        dbc.ListGroupItemText("Число ушедших NER'ов"),
                                    ],
                                    color="secondary"
                                )
                            ]
                        )
                    ]
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id="article_dropdown",
                                ),
                            ],
                        ),
                        html.Br(),
                        html.H2("Textlines"),
                        dcc.Graph(
                            id='textline',
                            figure={}
                        ),
                        dbc.Textarea(id='output_article_textarea', className="mb-3", bs_size="lg",
                                     placeholder="Текст найденной статьи...", readOnly=True),
                    ]
                ),
            ]
        )
    ],
    className="mt-4",
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SIMPLEX])

app.layout = html.Div([navbar, body, search_layer, display])

@app.callback(
    dash.dependencies.Output('hidden_in', 'children'),
    [dash.dependencies.Input('keywords_button', 'n_clicks')],
    state=[dash.dependencies.State(component_id='article_textarea', component_property='value')]
)
def get_keywds(_n_clicks, text):
    df = get_all(text, '')
    return df.to_json(orient='split')

@app.callback(
    dash.dependencies.Output('keywords_table', 'children'),
    [dash.dependencies.Input('hidden_in', 'children')]
)
def update_keywords_table(df_json):
    in_df = pd.read_json(df_json, orient='split')
    df = extract_unique(in_df)
    #print(df)
    wds = df['Tokens'].tolist()
    typs = df['Type'].tolist()
    kwds_lst = []
    for i in range(df.shape[0]):
        kwds_lst.append(html.Tr([html.Td(wds[i]), html.Td(typs[i])]))
    kw_body = [html.Tbody(kwds_lst)]
    return table_header + kw_body

@app.callback(
    dash.dependencies.Output('per', 'max'),
    [dash.dependencies.Input('hidden_in', 'children')]
)
def per_max(df_json):
    in_df = pd.read_json(df_json, orient='split')
    df = extract_unique(in_df)
    max_per = df[df['Type'] == 'PER'].shape[0]
    return max_per

@app.callback(
    dash.dependencies.Output('loc', 'max'),
    [dash.dependencies.Input('hidden_in', 'children')]
)
def loc_max(df_json):
    in_df = pd.read_json(df_json, orient='split')
    df = extract_unique(in_df)
    max_loc = df[df['Type'] == 'LOC'].shape[0]
    return max_loc

@app.callback(
    dash.dependencies.Output('org', 'max'),
    [dash.dependencies.Input('hidden_in', 'children')]
)
def org_max(df_json):
    in_df = pd.read_json(df_json, orient='split')
    df = extract_unique(in_df)
    max_org = df[df['Type'] == 'ORG'].shape[0]
    return max_org

@app.callback(
    dash.dependencies.Output('search_input', 'value'),
    [dash.dependencies.Input('per', 'value'),
     dash.dependencies.Input('loc', 'value'),
     dash.dependencies.Input('org', 'value'),
     dash.dependencies.Input('hidden_in', 'children')]
)
def update_search_string(per, loc, org, df_json):
    in_df = pd.read_json(df_json, orient='split')
    df = extract_unique(in_df)
    if df.empty:
        return ''
    df1 = df[df['Type'] == 'PER'].iloc[:per]
    if per is None:
        df1 = pd.DataFrame(columns=df.columns)
    df4 = df[df['Type'] == 'LOC'].iloc[:loc]
    if loc is None:
        df4 = pd.DataFrame(columns=df.columns)
    df3 = df[df['Type'] == 'ORG'].iloc[:org]
    if org is None:
        df3 = pd.DataFrame(columns=df.columns)
    search_string = ' '.join(df1['Tokens'].values.tolist() + df4['Tokens'].values.tolist() + df3['Tokens'].values.tolist())
    return search_string

@app.callback(
    dash.dependencies.Output('hidden_out', 'children'),
    [dash.dependencies.Input('search_button', 'n_clicks')],
    state=[dash.dependencies.State(component_id='search_input', component_property='value')]
)
def search(_n_clicks, search_string):
    print('Search callback')
    #if search_string:
    if search_string != '':
        print('Im here')
        df = start_search(search_string, 10)
            #print('Im still here')
            #print(df)
        #docs = df["Doc"].unique()
        #idx = df[df['Doc'] == 0].index.tolist()
 #           if df.empty:
 #               return None
            #print(df)
        return df.to_json(orient='split')
    #return ''

@app.callback(
    dash.dependencies.Output('article_dropdown', 'options'),
    [dash.dependencies.Input('hidden_out', 'children')]
)
def update_dropdown(df_json):
    out_df = pd.read_json(df_json, orient='split')
    #print(out_df)
    docs = out_df["Doc"].unique()
    return [{'label': 'Статья %i' % i,
                'value': i
                } for i in docs]
    #[dbc.DropdownMenuItem("Результат %i" % i) for i in docs]

@app.callback(
    dash.dependencies.Output('output_article_textarea', 'value'),
    [dash.dependencies.Input('hidden_out', 'children'),
     dash.dependencies.Input('article_dropdown', 'value')]
)
def update_out_text(df_json, drdown):
    out_df = pd.read_json(df_json, orient='split')
    #print(out_df)
    idx = out_df[out_df['Doc'] == drdown].index.tolist()
    #print(drdown)
    if drdown is not None:
        return out_df['AText'].loc[idx[drdown]]
    return ''

@app.callback(
    dash.dependencies.Output('orig_list', 'children'),
    [dash.dependencies.Input('hidden_in', 'children')]
)
def update_ner1(df_in_json):
    in_df = pd.read_json(df_in_json, orient='split')
    df = extract_unique(in_df)
    return '%i' % df.shape[0]

@app.callback(
    dash.dependencies.Output('new_list', 'children'),
    [dash.dependencies.Input('hidden_out', 'children'),
     dash.dependencies.Input('article_dropdown', 'value')]
)
def update_ner2(df_json, drdown):
    out_df = pd.read_json(df_json, orient='split')
    df_1 = out_df[out_df['Doc'] == drdown]
    df = extract_unique(df_1)
    return '%i' % df.shape[0]

@app.callback(
    dash.dependencies.Output('add_list', 'children'),
    [dash.dependencies.Input('hidden_in', 'children'),
     dash.dependencies.Input('hidden_out', 'children'),
     dash.dependencies.Input('article_dropdown', 'value')
     ]
)
def update_ner3(df_in_json, df_json, drdown):
    in_df = pd.read_json(df_in_json, orient='split')
    df = extract_unique(in_df)
    out_df = pd.read_json(df_json, orient='split')
    df_1 = out_df[out_df['Doc'] == drdown]
    df_2 = extract_unique(df_1)
    count = 0
    for text in df_2['Normform'].tolist():
        if text not in df['Normform'].tolist():
            count += 1
    return '%i' % count

@app.callback(
    dash.dependencies.Output('del_list', 'children'),
    [dash.dependencies.Input('hidden_in', 'children'),
     dash.dependencies.Input('hidden_out', 'children'),
     dash.dependencies.Input('article_dropdown', 'value')
     ]
)
def update_ner4(df_in_json, df_json, drdown):
    in_df = pd.read_json(df_in_json, orient='split')
    df = extract_unique(in_df)
    out_df = pd.read_json(df_json, orient='split')
    df_1 = out_df[out_df['Doc'] == drdown]
    df_2 = extract_unique(df_1)
    count = 0
    if df_2.empty:
        return '0'
    for text in df['Normform'].tolist():
        if text not in df_2['Normform'].tolist():
            count += 1
    return '%i' % count

@app.callback(
    dash.dependencies.Output('textline', 'figure'),
    [dash.dependencies.Input('hidden_out', 'children'),
     dash.dependencies.Input('hidden_in', 'children'),
     dash.dependencies.Input('article_dropdown', 'value')]
)
def update_graph(df_json, df_in_json, drdown):
    out_df = pd.read_json(df_json, orient='split')
    #print(out_df)
    in_df = pd.read_json(df_in_json, orient='split')
    #print(in_df)
    df_plot = out_df[out_df['Doc'] == drdown]

    sens = in_df['Block'].unique().tolist()
    sens2 = df_plot['Block'].unique().tolist()
    sens_df = pd.DataFrame(columns=in_df.columns)
    sens2_df = pd.DataFrame(columns=df_plot.columns)
    for s1 in sens:
        idx = in_df[in_df['Block'] == s1].index.tolist()
        sens_df = sens_df.append(in_df.loc[idx[0]])

    for s2 in sens2:
        idx = df_plot[df_plot['Block'] == s2].index.tolist()
        sens2_df = sens2_df.append(df_plot.loc[idx[0]])
    #sens_df = in_df[[in_df['Block'] == s1 for s1 in sens]]
    #sens2_df = out_df[[out_df['Block'] == s2 for s2 in sens2]]

    return {
        'data': [
            go.Bar(name='Person', x=sens, y=sens_df['PER'].tolist()),
            go.Bar(name='Location', x=sens, y=sens_df['LOC'].tolist()),
            go.Bar(name='Organization', x=sens, y=sens_df['ORG'].tolist()),
            go.Bar(name='Person', x=sens2, y=[-v for v in sens2_df['PER'].tolist()]),
            go.Bar(name='Location', x=sens2, y=[-v for v in sens2_df['LOC'].tolist()]),
            go.Bar(name='Organization', x=sens2, y=[-v for v in sens2_df['ORG'].tolist()])
        ],
        'layout': {
            'barmode': 'relative'
        }
    }

if __name__ == "__main__":
    app.run_server()


dbc.DropdownMenuItem("A button", id="dropdown-button"),
dbc.DropdownMenuItem(
    "Internal link", href="/l/components/dropdown_menu"
),
dbc.DropdownMenuItem(
    "External Link", href="https://github.com"
),
dbc.DropdownMenuItem(
    "External relative",
    href="/l/components/dropdown_menu",
    external_link=True,
),