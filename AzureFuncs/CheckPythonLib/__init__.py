import logging

import azure.functions as func

import numpy as np
import pandas as pd
from scipy import misc
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        arr = np.array([2,3,4])
        s = pd.Series([1, 3, 5, np.nan, 6, 8])
        face = misc.face()

        clf = RandomForestClassifier(random_state=0)
        X = [[ 1,  2,  3], [11, 12, 13]]
        y = [0, 1]
        clf.fit(X, y)
        preX = clf.predict(X) 
        pre4 = clf.predict([[4, 5, 6], [14, 15, 16]])

        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
        ])
        predictions = model(x_train[:1]).numpy()

        return func.HttpResponse(f"HelloMac, {name}. numpy={arr}. pandas={s}. scipy={face} scikit={preX}{pre4}. tensorflow={predictions}.")
 
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
