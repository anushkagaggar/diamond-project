from flask import Flask, request, render_template, jsonify
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline
import boto3
import pickle
import os
from mangum import Mangum

bucket_name = "diamond-model-bucket"
model_file = "model.pkl"

s3 = boto3.client('s3')

# Download if not already present
if not os.path.exists(model_file):
    s3.download_file(bucket_name, model_file, model_file)

# Load the model
with open(model_file, "rb") as f:
    model = pickle.load(f)

# Define Flask app
application = Flask(__name__)
app = application

# Define routes
@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data = CustomData(
            carat=float(request.form.get('carat')),
            depth=float(request.form.get('depth')),
            table=float(request.form.get('table')),
            x=float(request.form.get('x')),
            y=float(request.form.get('y')),
            z=float(request.form.get('z')),
            cut=request.form.get('cut'),
            color=request.form.get('color'),
            clarity=request.form.get('clarity')
        )
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)
        results = round(pred[0], 2)
        return render_template('form.html', final_result=results)

# Add Mangum handler **after app is defined**
handler = Mangum(app)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)