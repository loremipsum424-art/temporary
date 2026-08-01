import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------------------------------------
# 1. Generate Data (same polynomial curve + noise)
# ---------------------------------------------------------
x_data = np.linspace(-3, 3, 500).reshape(-1, 1)
y_data = x_data**3 - 2 * x_data**2 + x_data + np.random.randn(500, 1) * 2

# ---------------------------------------------------------
# 2. Split using sklearn (70% train, 15% val, 15% test)
# ---------------------------------------------------------
x_train, x_temp, y_train, y_temp = train_test_split(
    x_data, y_data, test_size=0.3, random_state=42
)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, random_state=42
)

# ---------------------------------------------------------
# 3. Define Model & R2 Metric
# ---------------------------------------------------------
def r2_metric(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return (1 - SS_res / (SS_tot + tf.keras.backend.epsilon()))

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(1,)))  # Wider first layer
model.add(Dense(128, activation='relu'))                 # Added extra deep layer
model.add(Dense(64, activation='relu'))                  # Added extra deep layer
model.add(Dense(32, activation='relu'))                  # Stepping down gradually
model.add(Dense(1))                                      # Output layer
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='mse',
    metrics=[r2_metric]
)

# ---------------------------------------------------------
# 4. Callbacks
# ---------------------------------------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=150, restore_best_weights=True, verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=50, min_lr=1e-6, verbose=1
)

# ---------------------------------------------------------
# 5. Train
# ---------------------------------------------------------
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=2000,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ---------------------------------------------------------
# 6. Evaluate
# ---------------------------------------------------------
test_loss, test_r2 = model.evaluate(x_test, y_test, verbose=1)

print("-" * 50)
print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test R2 Score: {test_r2:.4f}")
print("-" * 50)

# ---------------------------------------------------------
# 7. Custom Prediction
# ---------------------------------------------------------
custom_value = 2.5
custom_prediction = model.predict(np.array([[custom_value]]), verbose=1)[0][0]
true_value = custom_value**3 - 2*custom_value**2 + custom_value

print(f"Input X = {custom_value}")
print(f"Predicted Y = {custom_prediction:.4f}")
print(f"True Y = {true_value:.4f}")
print("-" * 50)

# ---------------------------------------------------------
# 8. Plot
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['loss'], label='Train Loss', color='blue')
ax1.plot(history.history['val_loss'], label='Validation Loss', color='red')
ax1.set_title('Train vs Validation Loss (MSE)')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['r2_metric'], label='Train R2', color='blue')
ax2.plot(history.history['val_r2_metric'], label='Validation R2', color='red')
ax2.set_title('Train vs Validation R2 Score')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('R2 Score')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
