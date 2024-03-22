from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

#########################################################################
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/prediction' , methods = ['GET' , 'POST'])
def prediction():
    if request.method == 'POST':
        rev = request.form.get('Review')

        data_point = np.array([rev])

        model = joblib.load('models/logistic_regression_model.pkl')

        prediction = model.predict(data_point)

        if prediction[0] == 1:
            result = "Positive Review"
        else:
            result = "Negative Review"

        return render_template('Result.html', prediction=result)

    return render_template('Result.html', prediction="")

########################################################################


if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")