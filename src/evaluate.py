# src/evaluate.py

import tensorflow as tf
from data_preparation import create_generators

def evaluate(model_path, data_dir, batch_size=32):
    train_generator, validation_generator = create_generators(data_dir, batch_size=batch_size)

    model = tf.keras.models.load_model(model_path)

    loss, accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // batch_size)
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    data_dir = '../data'
    model_path = '../models/fish_disease_efficientnet.h5'
    evaluate(model_path, data_dir)
