from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

app = Flask(__name__)
CORS(app)

print("="*60)
print("LOADING MODEL AND PREPROCESSORS...")
print("="*60)

# Load preprocessor (handles scaling and encoding)
preprocessor = joblib.load(r"C:\Users\sandy\Desktop\DL Projects\Productivity\models\preprocessor.pkl")
print("✓ Preprocessor loaded")

# Load feature columns
feature_columns = joblib.load(r"C:\Users\sandy\Desktop\DL Projects\Productivity\models\feature_columns.pkl")
print("✓ Feature columns loaded")

# Load trained model
model = keras.models.load_model(
    r"C:\Users\sandy\Desktop\DL Projects\Productivity\models\productivity_mlp.keras"
)
print("✓ Model loaded successfully!")
print("="*60)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("\n" + "="*60)
        print("NEW PREDICTION REQUEST")
        print("="*60)
        print(f"Received data: {data}")
        
        # Validate required fields
        required_fields = [
            'age', 'daily_social_media_time', 'number_of_notifications',
            'work_hours_per_day', 'perceived_productivity_score', 'stress_level',
            'sleep_hours', 'screen_time_before_sleep', 'breaks_during_work',
            'coffee_consumption_per_day', 'days_feeling_burnout_per_month',
            'weekly_offline_hours', 'job_satisfaction_score', 'uses_focus_apps',
            'has_digital_wellbeing_enabled', 'gender', 'job_type',
            'social_platform_preference'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'success': False
                }), 400
        
        # Create input DataFrame with all base features
        input_data = pd.DataFrame([{
            'age': data['age'],
            'daily_social_media_time': data['daily_social_media_time'],
            'number_of_notifications': data['number_of_notifications'],
            'work_hours_per_day': data['work_hours_per_day'],
            'perceived_productivity_score': data['perceived_productivity_score'],
            'stress_level': data['stress_level'],
            'sleep_hours': data['sleep_hours'],
            'screen_time_before_sleep': data['screen_time_before_sleep'],
            'breaks_during_work': data['breaks_during_work'],
            'coffee_consumption_per_day': data['coffee_consumption_per_day'],
            'days_feeling_burnout_per_month': data['days_feeling_burnout_per_month'],
            'weekly_offline_hours': data['weekly_offline_hours'],
            'job_satisfaction_score': data['job_satisfaction_score'],
            'uses_focus_apps': int(bool(data['uses_focus_apps'])),
            'has_digital_wellbeing_enabled': int(bool(data['has_digital_wellbeing_enabled'])),
            'gender': data['gender'],
            'job_type': data['job_type'],
            'social_platform_preference': data['social_platform_preference']
        }])
        
        # Feature engineering (MUST match training script exactly!)
        input_data['interaction_social_x_number_of_notifications'] = (
            input_data['daily_social_media_time'] * input_data['number_of_notifications']
        )
        input_data['interaction_stress_burnout'] = (
            input_data['stress_level'] * input_data['days_feeling_burnout_per_month']
        )
        input_data['total_screen_time'] = (
            input_data['daily_social_media_time'] + input_data['screen_time_before_sleep']
        )
        input_data['wellbeing_score'] = (
            input_data['weekly_offline_hours'] - input_data['days_feeling_burnout_per_month']
        )
        
        print(f"✓ Input shape before preprocessing: {input_data.shape}")
        
        # Transform using preprocessor (handles imputation, scaling, encoding)
        input_processed = preprocessor.transform(input_data)
        print(f"✓ Input shape after preprocessing: {input_processed.shape}")
        
        # Make prediction
        prediction = model.predict(input_processed, verbose=0)
        raw_score = float(prediction[0][0])
        
        print(f"✓ Raw prediction: {raw_score:.2f}")
        
        # Clip to valid range (1-10) for safety
        was_clipped = (raw_score < 1.0 or raw_score > 10.0)
        final_score = np.clip(raw_score, 1.0, 10.0)
        
        if was_clipped:
            print(f"⚠ WARNING: Score clipped from {raw_score:.2f} to {final_score:.2f}")
        else:
            print(f"✓ Final score: {final_score:.2f}")
        
        print("="*60)
        
        return jsonify({
            'prediction': round(final_score, 2),
            'was_clipped': was_clipped,
            'raw_prediction': round(raw_score, 2),
            'success': True
        })
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*60)
        
        return jsonify({
            'error': str(e),
            'success': False
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'message': 'Productivity Predictor API is running!',
        'model_loaded': True
    })

if __name__ == '__main__':
    print("\n")
    print(" FLASK SERVER STARTING")
    print("="*60)
    print(" Server URL: http://localhost:5000")
    print(" Health check: http://localhost:5000/health")
    print("\n")
    app.run(debug=True, port=5000, host='0.0.0.0')