from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np 



from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)
app = application


#Route for homepage
@app.route('/')
def index():
    return render_template('index.html')


#predictions
@app.route('/predict', methods=['POST','GET'])
def predict_datapoint():
    if request.method == 'POST':
        data = request.form
        custom_data = CustomData(
            gender=data['gender'],
            race_ethnicity=data['ethnicity'],
            parental_level_of_education=data['parental_level_of_education'],
            lunch=data['lunch'],
            test_preparation_course=data['test_preparation_course'],
            reading_score=int(data['reading_score']),
            writing_score=int(data['writing_score'])
        )
        features = custom_data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(features)
        return render_template('home.html', results=prediction[0])

    else:
        return render_template('home.html')
    
if __name__ == "__main__":
    app.run(host='0.0.0.0')
