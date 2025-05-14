import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

# Load the trained model
model = joblib.load('my_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the request is JSON or form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # Extract features from the request
        features = []
        # Based on your model_training code, you have multiple features
        # Extract each feature from the request
        
        # List of expected features based on your training script
        # These should match the columns in your training data
        expected_features = [
            'mean', 'max', 'min', 'std', 'kurtosis', 'power',
            'spectralentropy', 'rms', 'zcr', 'rolloff',
            'centroid', 'bandwidth', 'flatness', 'contrast',
            'ber', 'snr', 'THD', 'f0', 'mfcc_mean1', 'mfcc_mean2',
            'mfcc_mean3', 'mfcc_mean4', 'mfcc_mean5', 'mfcc_mean6',
            'mfcc_mean7', 'mfcc_mean8', 'mfcc_mean9', 'mfcc_mean10',
            'mfcc_mean11', 'mfcc_mean12', 'mfcc_mean13', 'mfcc_std1',
            'mfcc_std2', 'mfcc_std3', 'mfcc_std4', 'mfcc_std5',
            'mfcc_std6', 'mfcc_std7', 'mfcc_std8', 'mfcc_std9',
            'mfcc_std10', 'mfcc_std11', 'mfcc_std12', 'mfcc_std13'
        ]
        
        # Extract each feature from the request
        for feature in expected_features:
            features.append(float(data.get(feature, 0)))
        
        # Convert to numpy array and reshape for prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)
        
        # In your model, the target appears to have 3 classes (1, 2, 3)
        # Map these to meaningful labels if needed
        class_labels = {
            1: "Class 1", 
            2: "Class 2", 
            3: "Class 3"
        }
        
        predicted_class = int(prediction[0])
        result = {
            'predicted_class': predicted_class,
            'class_label': class_labels.get(predicted_class, "Unknown")
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        # Expecting a CSV file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        # Read the uploaded CSV
        df = pd.read_csv(file)
        
        # Get features from the dataframe
        # Make sure the CSV has the same columns as your training data
        feature_columns = [col for col in df.columns if col != 'target']
        
        # Make predictions
        predictions = model.predict(df[feature_columns])
        
        # Create result dataframe
        results = pd.DataFrame({
            'predicted_class': predictions
        })
        
        # Convert to JSON
        return results.to_json(orient='records')
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
