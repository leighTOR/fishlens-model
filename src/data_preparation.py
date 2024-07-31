# src/data_preparation.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(data_dir, image_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator
