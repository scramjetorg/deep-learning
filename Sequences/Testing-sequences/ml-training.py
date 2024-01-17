#https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2022-04-01-tensorflow-audio-classifier/2022-04-01/#converting-data-into-a-spectrogram

import os
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np

from keras import layers, optimizers
from keras import models
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv2D, Dense, Flatten



def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels) 
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

def preprocess(file_path): #, label):
    # Load files into 16kHz mono
    wav = load_wav_16k_mono(file_path)
    # read the first 1 secs
    wav = wav[:16000]
    # If file < 1 sec pad it with zeros
    zero_padding = tf.zeros([16000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    # Use Short-time Fourier Transform
    spectrogram = tf.signal.stft(wav, frame_length=255, frame_step=128)
    # Convert to absolut values (no negatives)
    spectrogram = tf.abs(spectrogram)
    # Add channel dimension (needed by CNN later)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram #, label

def get_record(path):
    path_dataset = os.listdir(path)
    for folder in path_dataset:
        label = folder
        print(label)
        path_folder = os.path.join(path, folder)
        for file in os.listdir(path_folder):
            dir_index = path_dataset.index(folder)
            audio_dir = os.path.join(path_folder, file)
            spectrogram = preprocess(audio_dir) #, label)
            #print(label.dtype)
            yield spectrogram, dir_index

async def run(context, input):
    dataset_path = "../data/mini_speech_commands"
    dataset = tf.data.Dataset.from_generator(
        get_record,
        args=[dataset_path],
        output_signature=(
            tf.TensorSpec(shape=(124, 129, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)))



    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=12600)
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(tf.data.AUTOTUNE) #8

    train = dataset.take(160)
    test = dataset.skip(160).take(30)


    input_shape= (124, 129, 1)
    norm_layer = layers.Normalization()
    # Fit the state of the layer to the spectrograms
    # with `Normalization.adapt`.
    norm_layer.adapt(data=train.map(map_func=lambda spec, label: spec))

    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Downsample the input.
        layers.Resizing(32, 32),
        # Normalize.
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(), #learning_rate=0.001), 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), #from_logits=True),
        metrics=['accuracy'],
    )

    hist = model.fit(train, validation_steps=8, epochs=20, validation_data=test)
    
    return input