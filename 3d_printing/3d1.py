import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Preprocessing function (NumPy version, no OpenCV required)
def grayscale_to_rgb(img):
    img = np.array(img, dtype=np.float32)
    if img.ndim == 3 and img.shape[-1] == 3:
        img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    img = np.expand_dims(img, axis=-1)
    img = np.repeat(img, 3, axis=-1)
    return img

# Use absolute path for robustness
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "base", "full_res")  # Adjust as needed

# Enhanced data augmentation with grayscale preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2,
    preprocessing_function=grayscale_to_rgb  # Convert to grayscale and repeat to RGB
)

train_gen = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

print("Class indices:", train_gen.class_indices)

# Save class indices
class_indices = train_gen.class_indices
inv_class_indices = {v: k for k, v in class_indices.items()}
with open('class_indices.pkl', 'wb') as f:
    pickle.dump(inv_class_indices, f)

# Model building with regularization
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(3, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=x)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)

# Training
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,
    callbacks=[early_stop, reduce_lr]
)

# Fine-tuning
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=3,
    callbacks=[early_stop, reduce_lr]
)

# Combine histories (optional, for plotting)
history = {
    'loss': history1.history['loss'] + history2.history['loss'],
    'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
    'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
    'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy']
}

# Print per-epoch metrics
print("\nTraining and Validation Metrics per Epoch:")
for epoch in range(len(history['loss'])):
    print(f"Epoch {epoch+1}:")
    print(f"  Training Loss: {history['loss'][epoch]:.4f}")
    print(f"  Training Accuracy: {history['accuracy'][epoch]:.4f}")
    print(f"  Validation Loss: {history['val_loss'][epoch]:.4f}")
    print(f"  Validation Accuracy: {history['val_accuracy'][epoch]:.4f}")
    print("---")

# Evaluate on validation set
print("\nEvaluating on Validation Set:")
val_loss, val_acc = model.evaluate(val_gen)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

# Save model
model.save('resnet50_3dprint_defect4.h5')

# Optional: Print model summary
print("\nModel Summary:")
model.summary()

# Optional: Plot training curves
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
print("Training curves saved as 'training_curves.png'")
