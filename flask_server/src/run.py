import os
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, url_for, redirect
import plotly.express as px
import ensembles
import validator as vld


UPLOAD_FOLDER = r'./uploads_data'

app = Flask(__name__, template_folder='./templates/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

mut_data = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/model_training', methods=['POST'])
def model_training():
    if request.method == 'POST':
        global mut_data

        model_name = request.form['algorithm']

        call_dict = {}
        if 'n_estimators' in request.form and request.form['n_estimators'] != '':
            try:
                call_dict['n_estimators'] = vld.validation(request.form['n_estimators'],
                                                           vld.is_int(),
                                                           vld.in_range(1, 10_000, '[', ']'))
            except TypeError as err:
                return render_template('index.html', n_estimators=err)

        if 'max_depth' in request.form and request.form['max_depth'] != '':
            try:
                call_dict['max_depth'] = vld.validation(request.form['max_depth'],
                                                        vld.is_str({'-1': None}),
                                                        vld.is_int(),
                                                        vld.in_range(1, 10_000, '[', ']'))
            except TypeError as err:
                return render_template('index.html', max_depth=err)

        if 'feature_subsample_size' in request.form and request.form['feature_subsample_size'] != '':
            try:
                call_dict['feature_subsample_size'] = vld.validation(request.form['feature_subsample_size'],
                                                                     vld.is_float(),
                                                                     vld.in_range(0, 1, '(', ']'))

            except TypeError as err:
                return render_template('index.html', feature_subsample_size=err)

        if 'learning_rate' in request.form and request.form['learning_rate'] != '':
            try:
                call_dict['learning_rate'] = vld.validation(request.form['learning_rate'],
                                                            vld.is_float(),
                                                            vld.in_range(0, 1, '(', ']'))
            except TypeError as err:
                return render_template('index.html', learning_rate=err)

        try:
            file = request.files['trainDataset']
            if '.csv' not in file.filename:
                raise Exception('Необходим .csv файл')

            filename = secure_filename(file.filename)
            train_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(train_path)

        except Exception as err:
            return render_template('index.html', trainDataset=err)

        try:
            val_path = None
            if 'validationDataset' in request.files and request.files['validationDataset'].filename != '':
                file = request.files['validationDataset']
                if '.csv' not in file.filename:
                    raise Exception('Необходим .csv файл')

                filename = secure_filename(file.filename)
                val_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(val_path)

        except Exception as err:
            return render_template('index.html', validationDataset=err)

        mut_data = train_model(train_path=train_path,
                               val_path=val_path,
                               model_name=model_name,
                               call_dict=call_dict)

        return redirect(url_for('model_eval'))


@app.route('/model_eval')
def model_eval():
    global mut_data

    return render_template('model_eval.html',
                           model=mut_data['model'],
                           plot=plot_render(mut_data['history']),
                           history=mut_data['history'],
                           train=mut_data['train_df'],
                           train_path=mut_data['train_path'],
                           val_path=mut_data['val_path'],
                           val=mut_data['val_df'],
                           list=list)


def plot_render(history: dict):
    df = pd.DataFrame(history)
    if 'val_score' in history:
        col = {"train_score": "Обучение", "val_score": "Валидация"}
        y = ["Обучение", "Валидация"]
    else:
        col = {"train_score": "Обучение"}
        y = ["Обучение"]

    df.rename(columns=col, inplace=True)
    df['iter'] = np.arange(df.shape[0])

    fig = px.line(df, x='time', y=y,
                  hover_data=['iter'],
                  labels={'iter': 'Итерация',
                          'value': 'RMSE',
                          'time': 'Время обучения, сек',
                          'variable': 'Кривая'})

    fig.update_layout(
        margin=dict(t=25, l=0, b=0, r=0)
    )

    return fig.to_html(full_html=False)


def train_model(train_path, val_path, model_name, call_dict):
    ret_dict = {'val_path': val_path, 'train_path': train_path}

    train = pd.read_csv(train_path)
    ret_dict['train_df'] = train
    X_train, y_train = train.loc[:, train.columns != 'y'].values, train.loc[:, 'y'].values

    X_val, y_val = None, None
    ret_dict['val_df'] = None
    if val_path is not None:
        val = pd.read_csv(val_path)
        ret_dict['val_df'] = val
        X_val, y_val = val.loc[:, val.columns != 'y'].values, val.loc[:, 'y'].values

    model = None
    if model_name == 'Gradient Boosting Regression':
        model = ensembles.GradientBoostingMSE(**call_dict)
    elif model_name == 'Random Forest Regression':
        model = ensembles.RandomForestMSE(**call_dict)

    history = model.fit(X_train, y_train, X_val, y_val, trace=True)
    ret_dict['model'] = model
    ret_dict['history'] = history

    return ret_dict


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
