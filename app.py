from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)
try:
    model = joblib.load('recovery_model2.pkl')
    print("Model Loaded ✅")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None 

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        features = np.array([
            float(data.get('Age', '0') or '0'),
            float(data.get('Height_cm', '0') or '0'),
            float(data.get('Training_Intensity', '0') or '0'),
            float(data.get('Training_Hours_Per_Week', '0') or '0'),
            float(data.get('Match_Count_Per_Week', '0') or '0'),
            float(data.get('Rest_Between_Events_Days', '0') or '0'),
            float(data.get('Load_Balance_Score', '0') or '0'),
            float(data.get('ACL_Risk_Score', '0') or '0'),
            1 if data.get('Gender', '').strip().lower() == 'female' else 0,
            1 if data.get('Gender', '').strip().lower() == 'male' else 0,
        ]).reshape(1, -1)
        prediction = model.predict(features)

        return jsonify({
            'predicted_recovery_days': float(prediction[0]),
            'message': f"Predicted recovery days: {float(prediction[0]):.2f}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    from waitress import serve 
    print("Starting server on http://0.0.0.0:4000 🚀")
    serve(app, host="0.0.0.0", port=4000)
