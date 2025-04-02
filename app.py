from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)
try:
    model = joblib.load('RecoveryTime_model.pkl')
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
        Age = float(data.get('Age', '0') or '0')
        Height_cm = float(data.get('Height_cm', '0') or '0')
        Weight_kg = float(data.get('Weight_kg', '0') or '0')
        Training_Intensity = float(data.get('Training_Intensity', '0') or '0')
        Training_Hours_Per_Week = float(data.get('Training_Hours_Per_Week', '0') or '0')
        Match_Count_Per_Week = float(data.get('Match_Count_Per_Week', '0') or '0')
        Rest_Between_Events_Days = float(data.get('Rest_Between_Events_Days', '0') or '0')
        Fatigue_Score = float(data.get('Fatigue_Score', '0') or '0')
        Performance_Score = float(data.get('Performance_Score', '0') or '0')
        Team_Contribution_Score = float(data.get('Team_Contribution_Score', '0') or '0')
        Load_Balance_Score = float(data.get('Load_Balance_Score', '0') or '0')
        ACL_Risk_Score = float(data.get('ACL_Risk_Score', '0') or '0')
        Injury_Indicator = float(data.get('Injury_Indicator', '0') or '0')
        Gender_Female = 1 if data.get('Gender_Female', '').strip().lower() == 'female' else 0
        Gender_Male = 1 if data.get('Gender_Male', '').strip().lower() == 'male' else 0
        Position_Center = 1 if data.get('Position_Center', '').strip().lower() == 'center' else 0
        Position_Forward = 1 if data.get('Position_Forward', '').strip().lower() == 'forward' else 0
        Position_Guard = 1 if data.get('Position_Guard', '').strip().lower() == 'guard' else 0
        Height_m = Height_cm / 100 
        BMI = Weight_kg / (Height_m ** 2) 
        Match_Rest_Ratio = Match_Count_Per_Week / Rest_Between_Events_Days if Rest_Between_Events_Days != 0 else 0  # Avoid division by zero
        Fatigue_Rest_Ratio = Fatigue_Score / Rest_Between_Events_Days if Rest_Between_Events_Days != 0 else 0  # Avoid division by zero
        Training_Load = Training_Intensity * Training_Hours_Per_Week 
        ACL_Injury_Interaction = Load_Balance_Score * ACL_Risk_Score 
        features = np.array([
            Age, Height_cm, Weight_kg, Training_Intensity, Training_Hours_Per_Week,
            Match_Count_Per_Week, Rest_Between_Events_Days, Fatigue_Score, Performance_Score,
            Team_Contribution_Score, Load_Balance_Score, ACL_Risk_Score, Injury_Indicator,
            Gender_Female, Gender_Male, Position_Center, Position_Forward, Position_Guard,
            ACL_Injury_Interaction, Height_m, BMI, Training_Load, Fatigue_Rest_Ratio, Match_Rest_Ratio
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
