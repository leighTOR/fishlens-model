# src/convert_to_tflite.py

import tensorflow as tf

def convert_to_tflite(model_path, tflite_model_path):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

if __name__ == '__main__':
    model_path = '../models/fish_disease_efficientnet.h5'
    tflite_model_path = '../models/fish_disease_efficientnet.tflite'
    convert_to_tflite(model_path, tflite_model_path)
