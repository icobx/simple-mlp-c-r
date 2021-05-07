import numpy as np
import pandas as pd

import scipy.stats as stats

from sklearn import preprocessing, model_selection, metrics
import sklearn.neural_network as neural
import sklearn.linear_model as linear


import matplotlib.pyplot as plt


def train_classifier(model, df):
    dataset = df.values
    # print(df.shape)

    X = dataset[:, 0:6]
    Y = dataset[:, 6]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.3)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm = metrics.confusion_matrix(y_pred, y_test)

    print("Training set score: %f" % model.score(X_test, y_test))
    print("Training set loss: %f" % model.loss_)

    print(metrics.classification_report(y_test, y_pred))
    print(cm)

    plt.plot(model.loss_curve_, label='training loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def train_regressor(model, df):
    dataset = df.values

    X = dataset[:, 0:4]
    Y = dataset[:, 4]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.30)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    print('MLP      | Mean Squared Error: ', metrics.mean_squared_error(y_pred, y_train))
    print('MLP      | R2:                 ', model.score(X_test, y_test))

    return model


def train_lregressor(model, df):
    dataset = df.values

    X = dataset[:, 0:4]
    Y = dataset[:, 4]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.30)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Linear   | Mean Squared Error: ', metrics.mean_squared_error(y_pred, y_test))
    print('Linear   | R2:                 ', model.score(X_test, y_test))

    return model


def setup_classifier():
    return neural.MLPClassifier(
        hidden_layer_sizes=(20, 50, 10),
        max_iter=200,
        activation='relu',
        solver='adam',
        random_state=1,
    )


def setup_regressor():
    return neural.MLPRegressor(
        hidden_layer_sizes=(80, 140, 100),
        activation="relu",
        learning_rate_init=0.001,
        max_iter=200,
        random_state=9
    )


def replace_strings(df):
    df['cholesterol'] = pd.Categorical(df['cholesterol'])
    df['cholesterol'] = df.cholesterol.cat.codes

    df['glucose'] = pd.Categorical(df['glucose'])
    df['glucose'] = df.glucose.cat.codes

    df['gender'] = pd.Categorical(df['gender'])
    df['gender'] = df.gender.cat.codes

    return df


def fix_data(df):
    del df['id']
    df.dropna(inplace=True)

    df = df[(df['ap_lo'] > 0)]
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

    return df


def normalization(df):
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=df.columns)

    print(df.corr())
    plt.matshow(df.corr())
    plt.show()
    return df


def get_bmi_df(df):
    df_bmi = df[:
             ]
    weight = df_bmi.pop('weight')
    height = df_bmi.pop('height')

    height = height.astype(float)
    height = height.div(100)

    df_bmi['bmi'] = weight / (height*height)

    # clean by BMI
    # df_bmi = df_bmi[(np.abs(stats.zscore(df_bmi['bmi'])) < 3)] #.all(axis=1)
    df_bmi = df_bmi[(df_bmi['bmi'] < 44) & (df_bmi['bmi'] > 15)]
    print(df_bmi.shape)
    return df_bmi


def main():
    df = pd.read_csv('data/srdcove_choroby.csv')

    df = replace_strings(df=df)

    df = fix_data(df=df)

    df_bmi = get_bmi_df(df)

    df = normalization(df=df)
    df_bmi = normalization(df=df_bmi)

    # drop low impact
    temp = df[['smoke', 'alco', 'gender', 'height', 'active']]
    df.drop(columns=temp.columns, inplace=True)

    temp = df_bmi[['smoke', 'alco', 'active', 'cholesterol', 'glucose', 'gender']]
    df_bmi.drop(columns=temp.columns, inplace=True)

    # model = setup_classifier()
    # train_classifier(model, df)

    model_r = setup_regressor()
    model_r = train_regressor(model_r, df_bmi)

    model_lr = linear.LinearRegression()
    model_lr = train_lregressor(model_lr, df_bmi)
    

if __name__ == '__main__':
    main()

