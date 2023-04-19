from flask import Flask, redirect, url_for, render_template, request, jsonify
import pickle
import numpy as np
app = Flask(__name__, template_folder='template')
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def welcome():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if prediction[0] == 1:
        return render_template('index.html', prediction_text='The person is diabetic.')
    else:
        return render_template('index.html', prediction_text='The person is Non-diabetic.')


if __name__ == '__main__':
    app.run(debug=True)
