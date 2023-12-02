import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os 


appUT =Flask(__name__)
modelUT=pickle.load(open('modelUT.pkl','rb'))

@appUT.route('/')
def home():
    return render_template('index.html')
@appUT.route('/predict', methods = ['POST'])
def predict():
    input_data = request.get_json()

    int_features = input_data.get('features', [])
    final_features = [np.array(int_features)]

    if (final_features[0] < 70):
        response_data = {
            'prediction_text': 'Haha lol no branch change for you'
        }

    else :
        prediction = modelUT.predict(final_features)

        output = round(prediction[0], 2)
        if output == 1:
            output = "CSE"
        if output == 2:
            output = "CSE AIML"
        if output == 3:
            output = "CSE DS"
        if output == 0:
            output = "CS"
        if output == 4:
            output = "CS  IT"
        if output == 5:
            output = "IT"

        response_data = {
            'prediction_text': 'Your Branch might be upgraded to :  {}'.format(output)
        }
            
    return jsonify(response_data)

@appUT.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = modelUT.predict([np.array(list(data.values()))])


    output = prediction

    
    return jsonify(output)        

if __name__ == "__main__":
    appUT.run(host='0.0.0.0', port=int(os.environ.get('PORT', 4000)), debug=True)