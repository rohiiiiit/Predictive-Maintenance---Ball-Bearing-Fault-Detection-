
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Data preparation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.002,
    height_shift_range=0.002,
    zoom_range=0.05,
    fill_mode='constant',
    cval=0
)

train_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/AI_PROJECT/DATASET/TRAINING',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
)

val_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/AI_PROJECT/DATASET/VALIDATION',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
)

# Model architecture
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('/content/drive/MyDrive/AI_PROJECT/best_model.h5',
                             monitor='val_accuracy',
                             save_best_only=True,
                             mode='max')

early_stop = EarlyStopping(monitor='val_loss',
                           patience=10,
                           restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              min_lr=1e-6)

# Training
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# Save final model
model.save('/content/drive/MyDrive/AI_PROJECT/final_ai_model.h5')
