import os
from flask import Flask, render_template, request
from utils import load_model, predict, name_pkl_model, train_model, save_model, remove_empty_key

app = Flask(__name__)


@app.route('/')
def homepage():
    return render_template("index.html")


@app.route('/predict', methods=["POST"])
def predict_price():
    form_result = request.form.to_dict()
    processed_form = remove_empty_key(form_result)
    model_name = name_pkl_model(processed_form)
    data = list(processed_form.values())

    if os.path.exists(model_name):
        model = load_model(model_name)
    else:
        cols = model_name.split("_")
        model, _, _ = train_model(cols)
        save_model(model, model_name)

    predicted_price = predict(model, [data])

    return render_template("prediction.html", prediction=predicted_price[0][0])


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
