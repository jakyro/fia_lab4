from sklearn import metrics
from utils import train_model, predict, get_data_csv


def test_accuracy_all_columns():
    columns = ["col1", "col2", "complexAge", "totalRooms", "totalBedrooms", "complexInhabitants", "apartmentsNr",
               "col8"]
    model, x_test, y_test = train_model(columns)
    y_pred = predict(model, x_test)

    accuracy = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)

    assert accuracy >= 0.6
    assert mae >= 50826
    assert mse >= 4825755044


def test_accuracy_basic_columns():
    columns = ["complexAge", "totalRooms", "totalBedrooms", "complexInhabitants", "apartmentsNr"]
    model, x_test, y_test = train_model(columns)
    y_pred = predict(model, x_test)

    accuracy = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)

    assert accuracy >= 0.15
    assert mae >= 82573
    assert mse >= 11207166061


def test_nan_values():
    columns = ["col1", "col2", "complexAge", "totalRooms", "totalBedrooms", "complexInhabitants", "apartmentsNr",
               "col8", "medianComplexValue"]
    df = get_data_csv(columns)
    sum_nan_columns = df.isnull().sum().tolist()
    is_not_nan_columns = all(col == 0 for col in sum_nan_columns)

    assert is_not_nan_columns
