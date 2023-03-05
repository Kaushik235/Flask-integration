import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle
import jsonify

app = Flask(__name__)
model = pickle.load(open('classifier.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('./test.html')

@app.route('/predict',methods=['POST'])
def predict():
    feature_list = request.form.to_dict()
    feature_list = list(feature_list.values())
    feature_list = list(map(int, feature_list))
    final_features = np.array(feature_list).reshape(1, 11) 
    
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    if output == 1:
        text = "Not Exited"
    else:
        text = "Exited"

    return render_template('test.html', prediction_text='Customer  {}'.format(text))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



if  __name__ == '__main__':
    app.run(debug=True,port=5500)