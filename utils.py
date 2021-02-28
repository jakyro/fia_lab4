import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


def get_data_csv(cols):
    columns = ["col1", "col2", "complexAge", "totalRooms", "totalBedrooms", "complexInhabitants", "apartmentsNr",
               "col8", "medianComplexValue"]
    return pd.read_csv("apartmentComplexData.txt", names=columns, usecols=cols)


def train_model(columns_name):
    x = get_data_csv(columns_name)
    y = get_data_csv(["medianComplexValue"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    return lr, x_test, y_test


def save_model(model, filename):
    pickle.dump(model, open(f"models/{filename}", 'wb'))


def load_model(filename):
    return pickle.load(open(f"models/{filename}", 'rb'))


def predict(model, values):
    return model.predict(values)


def name_pkl_model(form_values):
    return '_'.join(form_values)


def remove_empty_key(dict_values):
    return {k: v for k, v in dict_values.items() if v and v.strip()}


def get_model(model_name):
    if os.path.exists(model_name):
        model = load_model(model_name)
    else:
        cols = model_name.split("_")
        model, _, _ = train_model(cols)
        save_model(model, model_name)
    return model