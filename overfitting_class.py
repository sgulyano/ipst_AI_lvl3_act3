import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


##
data = load_breast_cancer()

X = data['data'][:, :2]
y = data['target']

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42)

train_acc = []
test_acc = []
for i in range(1,16):
    clf = tree.DecisionTreeClassifier(max_depth=i)
    clf = clf.fit(x_train, y_train)

    acc_tr = accuracy_score(y_train, clf.predict(x_train))
    acc_te = accuracy_score(y_test, clf.predict(x_test))

    train_acc.append(acc_tr)
    test_acc.append(acc_te)


def get_fig(depth=4, show_dec_bound=True):
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    clf = clf.fit(x_train, y_train)

    acc_tr = accuracy_score(y_train, clf.predict(x_train))
    acc_te = accuracy_score(y_test, clf.predict(x_test))

    tree.plot_tree(clf)

    plot_step = 0.1
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    colorscale = [[0, 'peachpuff'], [1, 'lightcyan']]
    if show_dec_bound:
        fig = go.Figure(data=go.Heatmap(
            z=Z,
            x=np.arange(x_min, x_max, plot_step),
            y=np.arange(y_min, y_max, plot_step),
            colorscale=colorscale,
            colorbar=dict(),
            showscale=False
            # colorbar=dict(nticks=10, ticks='outside',
            #               ticklen=5, tickwidth=1,
            #               showticklabels=True,
            #               tickangle=0, tickfont_size=12)
        ),
            layout=go.Layout(
            xaxis=dict(range=[x_min, x_max]),
            xaxis_title=data.feature_names[0],
            yaxis=dict(range=[y_min, y_max]),
            yaxis_title=data.feature_names[1],
        )
        )
    else:
        fig = go.Figure(layout=go.Layout(
            xaxis=dict(range=[x_min, x_max]),
            xaxis_title=data.feature_names[0],
            yaxis=dict(range=[y_min, y_max]),
            yaxis_title=data.feature_names[1],
        ))

    colors = ['red', 'blue']
    for i, color in enumerate(colors):
        idx = np.where(y_train == i)
        fig.add_trace(go.Scatter(x=x_train[idx, 0].squeeze(), y=x_train[idx, 1].squeeze(),
                                 mode='markers',
                                 name=data.target_names[i],
                                 marker_color=color,
                                 opacity=0.8))
    for i, color in enumerate(colors):
        idx = np.where(y_test == i)
        fig.add_trace(go.Scatter(x=x_test[idx, 0].squeeze(), y=x_test[idx, 1].squeeze(),
                                 mode='markers',
                                 name=data.target_names[i] + ' test',
                                 marker_color=color,
                                 opacity=0.3))
    return fig, acc_tr, acc_te


fig, acc_tr, acc_te = get_fig()


app.layout = html.Div([
    html.H1(children='Overfitting/Underfitting in Classification'),

    html.Div(children='''
        ในแบบฝึกหัดนี้ ให้นักเรียนลองเปลี่ยนค่า hyperparamter depth ของ Decision Tree แล้วดูว่าเมื่อใดเกิด overfitting/underfitting
    '''),
    html.Div(children=[
        # dcc.Markdown('### ชุดข้อมูล'),
        # dcc.Dropdown(
        #     options=[
        #         {'label': 'มะเร็งเต้านม', 'value': 'breast_cancer'},
        #     ],
        #     value='breast_cancer'
        # ),

        dcc.Markdown('### Max Depth'),
        dcc.Slider(
            id='depth-slider-id',
            min=1,
            max=16,
            marks={i: '{}'.format(i) for i in [1, 4, 7, 10, 13, 16]},
            value=4,
        ),

        dcc.Graph(figure=go.Figure([go.Scatter(x=list(range(1,16)), y=train_acc, mode='markers', name="Train"), 
                              go.Scatter(x=list(range(1,16)), y=test_acc, mode='lines+markers', name="Test")],
                             layout=go.Layout(
            title='ข้อมูลที่ใช้ Train โมเดล',
            xaxis=dict(range=[0, 17]),
            xaxis_title='Depth',
            yaxis=dict(range=[0, 1.1]),
            yaxis_title='Accuracy',
        )))
    ],
        style={'width': '40%', 'display': 'inline-block', 'vertical-align': 'top'}
    ),

    html.Div(children=[
        dcc.Graph(id='graph-id', figure=fig),
        html.Div([
            html.Div(id='accuracy-train-id', children=f'Train Accuracy = {acc_tr:.3f}'),
            html.Div(id='accuracy-test-id', children=f'Test Accuracy = {acc_te:.3f}')
        ],
            style={'textAlign': 'center'}
        )
    ],
        style={'width': '50%', 'display': 'inline-block'}
    )
])


@app.callback(
    [Output(component_id='graph-id', component_property='figure'),
     Output(component_id='accuracy-train-id', component_property='children'),
     Output(component_id='accuracy-test-id', component_property='children')],
    [Input(component_id='depth-slider-id', component_property='value')]
)
def update_under_div(depth):
    fig, acc_tr, acc_te = get_fig(depth)
    return [fig, f'Train Accuracy = {acc_tr:.3f}', f'Test Accuracy = {acc_te:.3f}']


if __name__ == '__main__':
    app.run_server(debug=True)
