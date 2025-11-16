from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import os

# Initialize the app
app = Flask(__name__)

# Path to the model folder
MODEL_PATH = os.path.join(os.getcwd(), 'model')

# Load model and scaler
with open(os.path.join(MODEL_PATH, 'wine_quality_model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_PATH, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read each feature explicitly
        features = [
            float(request.form['fixed_acidity']),
            float(request.form['volatile_acidity']),
            float(request.form['citric_acid']),
            float(request.form['residual_sugar']),
            float(request.form['chlorides']),
            float(request.form['free_sulfur_dioxide']),
            float(request.form['total_sulfur_dioxide']),
            float(request.form['density']),
            float(request.form['pH']),
            float(request.form['sulphates']),
            float(request.form['alcohol'])
        ]

        # Convert to numpy array and reshape
        final_features = np.array(features).reshape(1, -1)

        # Scale features
        final_features = scaler.transform(final_features)

        # Make prediction
        prediction = model.predict(final_features)[0]

        return render_template('index.html', prediction_text=f'Predicted Wine Quality: {prediction}')
    
    except ValueError:
        return render_template('index.html', prediction_text='Error: Please enter valid numeric values for all features.')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

# Optional API route
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(force=True)
        features = np.array(data['features']).reshape(1, -1)
        features = scaler.transform(features)
        prediction = model.predict(features)[0]
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    import os
    # Heroku assigns a dynamic port, default to 5000 for local testing
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


