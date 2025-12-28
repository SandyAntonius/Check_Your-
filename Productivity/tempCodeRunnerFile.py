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
