# src/train.py

import tensorflow as tf
from data_preparation import create_generators
from model import build_model

def train(data_dir, model_path, batch_size=32, epochs=50):
    train_generator, validation_generator = create_generators(data_dir, batch_size=batch_size)
    num_classes = train_generator.num_classes

    model = build_model(num_classes)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint]
    )

    return history

if __name__ == '__main__':
    data_dir = '../data'
    model_path = '../models/fish_disease_efficientnet.h5'
    train(data_dir, model_path)
