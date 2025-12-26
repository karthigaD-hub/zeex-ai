import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

# -------------------------------
# CONFIG
# -------------------------------
IMG_SIZE = 128
BATCH = 32
EPOCHS = 25

TRAIN_DIR = r"D:\projects\zeex-fire-detect\data\processed\classification\train"
TEST_DIR  = r"D:\projects\zeex-fire-detect\data\processed\classification\test"

# -------------------------------
# 1. STRONG AUGMENTATION FOR TRAINING
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=40,
    zoom_range=0.25,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.25,
    brightness_range=[0.4, 1.6],
    channel_shift_range=35,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

# -------------------------------
# 2. TEST DATA (ONLY RESCALE)
# -------------------------------
test_datagen = ImageDataGenerator(rescale=1/255.0)

# -------------------------------
# 3. LOAD TRAIN & TEST DATA
# -------------------------------
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical"
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    shuffle=False
)

# -------------------------------
# 4. BUILD CNN MODEL FROM SCRATCH
# -------------------------------
model = Sequential([

    # BLOCK 1
    Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation="relu", padding="same"),
    MaxPooling2D(),
    Dropout(0.25),

    # BLOCK 2
    Conv2D(64, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(64, (3,3), activation="relu", padding="same"),
    MaxPooling2D(),
    Dropout(0.30),

    # BLOCK 3
    Conv2D(128, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(128, (3,3), activation="relu", padding="same"),
    MaxPooling2D(),
    Dropout(0.40),

    Flatten(),

    # FC LAYERS
    Dense(256, activation="relu"),
    Dropout(0.50),

    Dense(train_gen.num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# 5. TRAIN MODEL
# -------------------------------
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=test_gen
)

# -------------------------------
# 6. EVALUATION
# -------------------------------
pred_prob = model.predict(test_gen)
pred_classes = np.argmax(pred_prob, axis=1)
true_classes = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

print("\n------- CLASSIFICATION REPORT -------")
print(classification_report(true_classes, pred_classes, target_names=class_labels))

print("\n------- CONFUSION MATRIX -------")
print(confusion_matrix(true_classes, pred_classes))

# OPTIONAL: SAVE MODEL
model.save("fire_smoke_classifier_from_scratch.h5")
print("\nModel saved as fire_smoke_classifier_from_scratch.h5")
