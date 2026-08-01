import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

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
# Fixed variable names: split into train+val (85%) and test (15%)
x_trainWval, x_test, y_trainWval, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42
)



# ---------------------------------------------------------
# 3. Define Model 
# ---------------------------------------------------------


model = Sequential([
    Dense(64, activation="swish", input_shape=(1,)),  # Swish handles gradients better than ReLU
    Dense(32, activation="swish"),
    Dense(16, activation="swish"),
    Dense(1, activation="linear")
])
#compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
    loss="mse",
    metrics=[tf.keras.metrics.R2Score()]
)

# ---------------------------------------------------------
# 4. Callbacks
# ---------------------------------------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=150, restore_best_weights=True, verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=50, min_lr=1e-6, verbose=1
)

# ---------------------------------------------------------
# 5. Train
# ---------------------------------------------------------
history = model.fit(
    x_trainWval, y_trainWval,
    validation_split=0.2,
    epochs=300,                 
    batch_size=16,               
    shuffle=True,
    callbacks=[early_stop, reduce_lr]  
)

# ---------------------------------------------------------
# 6 & 7. Evaluate, Custom, and 50 Unseen Predictions
# ---------------------------------------------------------
print("-" * 50)
print(f"Test Loss / R2: {model.evaluate(x_test, y_test, verbose=0)}")

cv = 2.5
print(
    f"Custom X={cv} -> Pred: {model.predict(np.array([[cv]]), verbose=0)[0][0]:.4f}"
    f" | True: {cv**3 - 2*cv**2 + cv:.4f}"
)

np.random.seed(100)
x_rand = np.random.uniform(-3, 3, (50, 1))
y_rand_pred = model.predict(x_rand, verbose=0)
print("-" * 50)

# ---------------------------------------------------------
# 8. Plots (Combined & Corrected Keys)
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss Plot
ax1.plot(history.history["loss"], label="Train Loss", color="blue")
ax1.plot(history.history["val_loss"], label="Validation Loss", color="red")
ax1.set_title("Train vs Validation Loss (MSE)")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True)

# R2 Score Plot (Fixed key to match 'r2_score')
ax2.plot(history.history["r2_score"], label="Train R2", color="blue")
ax2.plot(history.history["val_r2_score"], label="Validation R2", color="red")
ax2.set_title("Train vs Validation R2 Score")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("R2 Score")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Unseen Predictions Scatter Plot
plt.figure(figsize=(9, 4))
x_smooth = np.linspace(-3, 3, 500).reshape(-1, 1)
plt.plot(x_smooth, x_smooth**3 - 2 * x_smooth**2 + x_smooth, "k--", label="True")
plt.scatter(x_rand, y_rand_pred, color="purple", label="50 Unseen Preds")
plt.legend()
plt.grid(True)
plt.show()
