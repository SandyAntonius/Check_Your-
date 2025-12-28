import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import keras_tuner as kt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os

# Helper functions
def ln():
    print("\n\n\n ===========\n\n")

def msg(word):
    print(f"\n======== {word} =========\n")

# Load data
data = pd.read_csv(r"C:\Users\sandy\Desktop\DL Projects\Productivity\data\raws\social_media_vs_productivity.csv")

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 50)   
pd.set_option('display.max_colwidth', 50)

msg("Loading Dataset")
print(data.head())
ln()
msg("The Columns")
print(data.columns.to_list())
ln()
msg("The Shape")
print(data.shape)
ln()
msg("The Data Types")
print(data.dtypes)
ln()
msg("Which Columns have NaNs")
print(data.isna().sum())

# Handle Boolean columns (False/True -> 0/1)
bool_col = ['uses_focus_apps', 'has_digital_wellbeing_enabled']
data[bool_col] = data[bool_col].astype(int)

msg("After the conversion")
print(data.head())

# Drop rows where target is missing
data = data.dropna(subset=['actual_productivity_score'])

# Feature Engineering (done before split since it doesn't leak info)
def feature_engineering(data):
    msg("The Interactions")
    data['interaction_social_x_number_of_notifications'] = data['daily_social_media_time'] * data['number_of_notifications']
    data["interaction_stress_burnout"] = data["stress_level"] * data["days_feeling_burnout_per_month"]
    
    msg("The Aggregate")
    data['total_screen_time'] = (data['daily_social_media_time'] + data['screen_time_before_sleep'])
    data["wellbeing_score"] = (data["weekly_offline_hours"] - data["days_feeling_burnout_per_month"])
    return data

data = feature_engineering(data)
msg("After Adding the Feature Engineering")
print(data.head())




# Define features explicitly
label = "actual_productivity_score"
cat_cols = ['gender', 'job_type', 'social_platform_preference']
num_cols = ['age', 'daily_social_media_time', 'number_of_notifications',
            'work_hours_per_day', 'perceived_productivity_score', 'stress_level',
            'sleep_hours', 'screen_time_before_sleep', 'breaks_during_work', 
            'coffee_consumption_per_day', 'days_feeling_burnout_per_month', 
            'weekly_offline_hours', 'job_satisfaction_score',
            # your engineered features:
            'interaction_social_x_number_of_notifications',
            'interaction_stress_burnout', 'total_screen_time', 'wellbeing_score',
            # boolean features
            'uses_focus_apps', 'has_digital_wellbeing_enabled']

feature_cols = num_cols + cat_cols

# Ensure these columns exist in data
X = data[feature_cols].copy()
y = data[label].copy()

# Split first (prevent leakage)
x_train, x_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.50, random_state=42)

print("Train:", x_train.shape)
print("Validation:", x_val.shape)
print("Test:", x_test.shape)

# Build ColumnTransformer that imputes numeric, scales numeric, encodes categorical
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),   # fit on train only
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # protects against unseen nulls
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, num_cols),
    ('cat', categorical_pipeline, cat_cols)
], remainder='drop')

# Fit preprocessor on train ONLY
X_train_proc = preprocessor.fit_transform(x_train)
X_val_proc   = preprocessor.transform(x_val)
X_test_proc  = preprocessor.transform(x_test)

# Save preprocessor and feature metadata
os.makedirs(r"C:\Users\sandy\Desktop\DL Projects\Productivity\models", exist_ok=True)
joblib.dump(preprocessor, r"C:\Users\sandy\Desktop\DL Projects\Productivity\models\preprocessor.pkl")
joblib.dump(feature_cols, r"C:\Users\sandy\Desktop\DL Projects\Productivity\models\feature_columns.pkl")
print("Saved preprocessor and feature list")

# Use processed data
x_train_scaled = X_train_proc
x_val_scaled   = X_val_proc
x_test_scaled  = X_test_proc

# ✅ FIX #1: Use actual input shape instead of hardcoded 36
input_shape = x_train_scaled.shape[1]
print(f"\n✅ Input shape: {input_shape}")

#  Simpler model to reduce overfitting
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(input_shape,)),
    layers.Dropout(0.3),  # Increased dropout
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),  # Increased dropout
    layers.Dense(1)  # Regression output
])

# ✅ FIX #3: Lower learning rate to prevent overfitting
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # Reduced from 0.001
    loss='mse',
    metrics=['mae']
)

model.summary()

# ✅ FIX #4: More aggressive early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,  # Increased patience
    restore_best_weights=True
)

# ✅ FIX #5: Add ReduceLROnPlateau to help with overfitting
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001
)

history = model.fit(
    x_train_scaled, y_train,
    validation_data=(x_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Save model
model.save(r"C:\Users\sandy\Desktop\DL Projects\Productivity\models\productivity_mlp.keras")
model.save_weights(r"C:\Users\sandy\Desktop\DL Projects\Productivity\models\productivity_mlp.weights.h5")
print("Model saved")

# Predictions
y_pred_train = model.predict(x_train_scaled)
y_pred_val = model.predict(x_val_scaled)
y_pred_test = model.predict(x_test_scaled)

# Evaluation Function
def evaluate(true, pred, dataset_name=""):
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    
    print(f"\n📌 Results for {dataset_name}:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

# Show Results
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)
evaluate(y_train, y_pred_train, "Training Set")
evaluate(y_val, y_pred_val, "Validation Set")
evaluate(y_test, y_pred_test, "Test Set")

# Check for overfitting
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f"\n Overfitting Check:")
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Val RMSE: {val_rmse:.4f}")
print(f"Difference: {abs(train_rmse - val_rmse):.4f}")
if abs(train_rmse - val_rmse) < 0.5:
    print(" Good! Model is NOT overfitting significantly")
else:
    print(" Warning: Model might be overfitting")

# Loss Curve
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Predictions vs Actual Plot
plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title("Predicted vs Actual Productivity (Test Set)")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.grid(True)
plt.show()

# Sample predictions
print("\n" + "="*50)
print("SAMPLE PREDICTIONS")
print("="*50)
sample_indices = np.random.choice(len(y_test), 10, replace=False)
for idx in sample_indices:
    actual = y_test.iloc[idx]
    predicted = y_pred_test[idx][0]
    print(f"Actual: {actual:.2f} | Predicted: {predicted:.2f} | Error: {abs(actual-predicted):.2f}")

print("\n✅ MODEL TRAINING COMPLETE!")