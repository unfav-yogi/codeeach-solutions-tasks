# Task 3 – Flask Financial Prediction App with Dropdown Lists


from flask import Flask, request, jsonify, render_template_string
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import joblib

app = Flask(__name__)

data = pd.read_excel(
    r"C:\Users\yoeshwar\OneDrive\Desktop\internships\code alpha\financial dashboard\Financial Sample.xlsx"
)

print(data.head())

original_data = data.copy()

label_encoders = {}

for col in data.select_dtypes(include='object').columns:

    le = LabelEncoder()

    data[col] = le.fit_transform(
        data[col].astype(str)
    )

    label_encoders[col] = le

for col in data.select_dtypes(include='datetime64[ns]').columns:

    data[col] = data[col].astype('int64')


data.dropna(inplace=True)

target_column = data.columns[-1]

X = data.drop(columns=[target_column])

y = data[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print("\nModel Trained Successfully")
print("Mean Absolute Error :", mae)

joblib.dump(model, "financial_model.pkl")

column_options = {}

for column in X.columns:

    if column in original_data.columns:

        unique_values = original_data[column].dropna().unique().tolist()

        column_options[column] = unique_values[:20]

html_page = """
<!DOCTYPE html>
<html>
<head>
    <title>Financial Prediction App</title>

    <style>

        body {
            font-family: Arial;
            padding: 30px;
            background-color: #f4f4f4;
        }

        h1 {
            color: #333;
        }

        form {
            background: white;
            padding: 20px;
            border-radius: 10px;
            width: 400px;
        }

        select, input {
            width: 100%;
            padding: 10px;
            margin-top: 5px;
            margin-bottom: 15px;
        }

        button {
            padding: 10px 20px;
            font-size: 16px;
        }

    </style>

</head>
<body>

    <h1>Financial Prediction System</h1>

    <form action="/predict" method="post">

        {% for column in columns %}

            <label>{{column}}</label><br>

            <select name="{{column}}" required>

                {% for option in options[column] %}

                    <option value="{{option}}">{{option}}</option>

                {% endfor %}

            </select>

        {% endfor %}

        <button type="submit">Predict</button>

    </form>

</body>
</html>
"""

@app.route('/')
def home():

    return render_template_string(
        html_page,
        columns=X.columns,
        options=column_options
    )

@app.route('/predict', methods=['POST'])
def predict():

    input_data = []

    for column in X.columns:

        value = request.form[column]

        if column in label_encoders:

            value = label_encoders[column].transform([str(value)])[0]

        else:

            value = float(value)

        input_data.append(value)

    prediction = model.predict([input_data])[0]

    return f"""

    <h1>Prediction Result</h1>

    <h2>{prediction}</h2>

    <a href="/">Go Back</a>

    """

@app.route('/api/predict', methods=['POST'])
def api_predict():

    data_json = request.get_json()

    input_data = [list(data_json.values())]

    prediction = model.predict(input_data)[0]

    return jsonify({
        "prediction": float(prediction)
    })

if __name__ == '__main__':

    app.run(debug=True)


