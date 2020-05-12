import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import HoverTool, ColumnDataSource, Slider, RadioButtonGroup, Div, Paragraph
from bokeh.plotting import figure
from bokeh.transform import factor_cmap, factor_mark
from bokeh.palettes import Category20_4
from sklearn import tree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

## load data
data = load_breast_cancer()

X = data['data'][:350, :2]
y = data['target'][:350]

# get min/max data
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


y_set = ['มะเร็ง/เนื้อร้าย', 'เนื้องอก']
y_map_value = {k:v for k, v in enumerate(y_set)}
y_train_name = [y_map_value[y]+ ' train' for y in y_train]
y_test_name = [y_map_value[y]+ ' test' for y in y_test]
y_labels = ['มะเร็ง/เนื้อร้าย train', 'มะเร็ง/เนื้อร้าย test', 'เนื้องอก train', 'เนื้องอก test']

plot_step = 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

def getZ(depth=1):
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    clf = clf.fit(x_train, y_train)

    acc_tr = accuracy_score(y_train, clf.predict(x_train))
    acc_te = accuracy_score(y_test, clf.predict(x_test))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    return Z.reshape(xx.shape), acc_tr, acc_te

Z, acc_tr, acc_te = getZ()

# Set up data
source = ColumnDataSource(data=dict(x=np.concatenate((x_train[:,0], x_test[:,0])), 
                                    y=np.concatenate((x_train[:,1], x_test[:,1])), 
                                    c=y_train_name + y_test_name))
bound_source = ColumnDataSource({'value': [Z]})

# Set up plot
plot = figure(plot_height=400, plot_width=600, title='ข้อมูลผู้ป่วยมะเร็งเต้านม',
              tools="crosshair,pan,reset,save,wheel_zoom", 
              x_range=[x_min, x_max], y_range=[y_min, y_max],
              x_axis_label='รัศมีเฉลี่ย',
              y_axis_label='ความขรุขระ')

img = plot.image('value', source=bound_source, x=x_min, y=y_min, dw=x_max-x_min, dh=y_max-y_min, 
                 palette=('#d2e2ff', '#ffe0cc'))

idx = y_train == 0
plot.scatter(x=x_train[idx,0], y=x_train[idx,1], legend_label='มะเร็ง/เนื้อร้าย train', 
             fill_alpha=0.6, size=10, marker='hex', color='#1f77b4',
             name='มะเร็ง/เนื้อร้าย train')
plot.scatter(x=x_train[~idx,0], y=x_train[~idx,1], legend_label='เนื้องอก train', 
             fill_alpha=0.6, size=10, marker='triangle', color='#ff7f0e',
             name='เนื้องอก train')

idx = y_test == 0
plot.scatter(x=x_test[idx,0], y=x_test[idx,1], legend_label='มะเร็ง/เนื้อร้าย test', 
             fill_alpha=0.6, size=10, marker='hex', color='#aec7e8',
             name='มะเร็ง/เนื้อร้าย test')
plot.scatter(x=x_test[~idx,0], y=x_test[~idx,1], legend_label='เนื้องอก test', 
             fill_alpha=0.6, size=10, marker='triangle', color='#ffbb78',
             name='เนื้องอก test')


hover = HoverTool(names=['มะเร็ง/เนื้อร้าย train', 'มะเร็ง/เนื้อร้าย test', 'เนื้องอก train', 'เนื้องอก test'], 
                  tooltips="""
    <div>
        <span style="font-size: 16px; font-weight: bold;">Index :</span><span style="font-size: 16px;">$index</span><br>
        <span style="font-size: 16px; font-weight: bold;">(X,Y) :</span><span style="font-size: 16px;">(@x, @y)</span><br>
        <span style="font-size: 16px; font-weight: bold;">Desc  :</span><span style="font-size: 16px;">$name</span><br>
    </div>
    """)
plot.add_tools(hover)
plot.legend.click_policy="hide"

# Set up dashboard
title = Div(text="""<H1>โมเดลที่เฉพาะเจาะจงเกินไป VS โมเดลที่ง่ายเกินไป (Overfitting/Underfitting in Classification)</H1>""")
desc = Paragraph(text="""ในแบบฝึกหัดนี้ ให้นักเรียนลองเปลี่ยนค่าตัวแปร depth ของ Decision Tree แล้วดูว่าเมื่อใดก่อให้เกิดโมเดลที่เฉพาะเจาะจงเกินไป (overfitting) และ โมเดลที่ง่ายเกินไป (underfitting) 
โดยการวาดกราฟเส้นระหว่างความแม่นยำกับค่าตัวแปร depth ของ Decision Tree ของทั้ง training data และ test data """)
header = column(title, desc, sizing_mode="scale_both")

# Set up widgets
text = Div(text="<H3>ตัวแปร</H3>")
depth_slider = Slider(title="Max Depth", value=1, start=1, end=10, step=1)
acc_txt = Div(text="<H3>ความแม่นยำ</H3>")
acc_tr_txt = Div(text=f'ความแม่นยำ บน training data = {acc_tr:.3f}')
acc_te_txt = Div(text=f'ความแม่นยำ บน test data = {acc_te:.3f}')


# Set up layouts and add to document
inputs = column(text, depth_slider, acc_txt, acc_tr_txt, acc_te_txt)
body = row(inputs, plot, width=800)

def update_data(attrname, old, new):
    # Get the current slider values
    depth = depth_slider.value
    Z, acc_tr, acc_te = getZ(depth)
    bound_source.data = {'value': [Z]}
    acc_tr_txt.text = f'ความแม่นยำ บน training data = {acc_tr:.3f}'
    acc_te_txt.text = f'ความแม่นยำ บน test data = {acc_te:.3f}'
    
depth_slider.on_change('value', update_data)

curdoc().add_root(column(header, body))
curdoc().title = "โมเดลสำหรับจำแนกที่เฉพาะเจาะจง/ง่ายเกินไป"