from sklearn import metrics
from utils import get_data_csv, train_model, predict

df = get_data_csv("apartmentComplexData.txt")


def test_accuracy_all_columns():
    x = df.iloc[:, 0:8]
    y = df.iloc[:, 8]
    model, x_test, y_test = train_model(x, y)
    y_pred = predict(model, x_test)

    accuracy = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)

    assert accuracy >= 0.6
    assert mae >= 50826
    assert mse >= 4825755044


def test_accuracy_basic_columns():
    x = df.iloc[:, 2:7]
    y = df.iloc[:, 8]
    model, x_test, y_test = train_model(x, y)
    y_pred = predict(model, x_test)

    accuracy = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)

    assert accuracy >= 0.15
    assert mae >= 82573
    assert mse >= 11207166061


def test_nan_values():
    sum_nan_columns = df.isnull().sum().tolist()
    is_not_nan_columns = all(col == 0 for col in sum_nan_columns)

    assert is_not_nan_columns
