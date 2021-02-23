from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

# loading model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    
    # input from HTML Form
    features = [int(x) for x in request.form.values()]
    final = [np.array(features)]
    print("Features:", features)
    print("Final:", final)

    # precting from input
    prediction = model.predict_proba(final)
    output = '{0: .{1}f}'.format(prediction[0][1], 2)

    if output > str(0.5):
        return render_template('index.html', pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output))
    else:
        return render_template('index.html', pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output))

if __name__ == '__main__':
    app.run(debug = True)