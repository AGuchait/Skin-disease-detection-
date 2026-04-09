# ==============================
# SKIN DISEASE DETECTION (IMPROVED)
# ==============================

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# ----------- PARAMETERS -----------
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 20

# ----------- LOAD DATASET -----------
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42
)

class_names = train_data.class_names
num_classes = len(class_names)

print("Classes:", class_names)

# ----------- PREPROCESSING (IMPORTANT CHANGE) -----------
from tensorflow.keras.applications.efficientnet import preprocess_input

train_data = train_data.map(lambda x, y: (preprocess_input(x), y))
val_data = val_data.map(lambda x, y: (preprocess_input(x), y))

# ----------- DATA AUGMENTATION -----------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

train_data = train_data.map(
    lambda x, y: (data_augmentation(x, training=True), y)
)

# ----------- PERFORMANCE OPTIMIZATION -----------
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.prefetch(buffer_size=AUTOTUNE)
val_data = val_data.prefetch(buffer_size=AUTOTUNE)

# ----------- MODEL (BIG CHANGE HERE 🔥) -----------
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

base_model = EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # Freeze first

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# ----------- COMPILE -----------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ----------- CLASS WEIGHTS -----------
from sklearn.utils.class_weight import compute_class_weight

labels = np.concatenate([y for x, y in train_data], axis=0)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(class_weights))

# ----------- TRAIN (PHASE 1) -----------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights
)

# ----------- FINE-TUNING (IMPORTANT 🔥) -----------
print("\nStarting Fine-Tuning...")

base_model.trainable = True

# Freeze MOST layers, train last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# ----------- SAVE MODEL -----------
model.save("skin_disease_model_best.h5")
print("\nModel saved as skin_disease_model_best.h5")

# ----------- EVALUATION -----------
loss, accuracy = model.evaluate(val_data)
print(f"\nFinal Validation Accuracy: {accuracy * 100:.2f}%")

# ----------- PLOTS -----------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.legend()
plt.title('Loss')

plt.show()