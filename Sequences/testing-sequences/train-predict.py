
import os
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio

from keras import layers
from keras import models
#from IPython import display


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

def get_spectrogram(file_path):
    # Load files into 16kHz mono
    wav = load_wav_16k_mono(file_path)
    # Only read the first 3 secs
    wav = wav[:16000]
    # If file < 3s add zeros
    zero_padding = tf.zeros([16000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    # Use Short-time Fourier Transform
    spectrogram = tf.signal.stft(wav, frame_length=255, frame_step=128)
    # Convert to absolut values (no negatives)
    spectrogram = tf.abs(spectrogram)
    # Add channel dimension (needed by CNN later)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram 


label_names = []
def get_record(path):
    path_dataset = os.listdir(path)
    for folder in path_dataset:
        label = folder
        print(label.decode("utf-8"))
        label = label.decode("utf-8")
        if label not in label_names:
           label_names.append(label)
        else:
           pass
        path_folder = os.path.join(path, folder)
        for file in os.listdir(path_folder):
            dir_index = path_dataset.index(folder)
            audio_dir = os.path.join(path_folder, file)
            spectrogram = get_spectrogram(audio_dir) 
            yield spectrogram, dir_index


dataset_path = "PATH/TO/FOLDER/mini_speech_commands"

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

history = model.fit(train, validation_steps=8, epochs=20, validation_data=test)

metrics = history.history
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Loss [CrossEntropy]')

plt.subplot(1,2,2)
plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
plt.legend(['accuracy', 'val_accuracy'])
plt.ylim([0, 100])
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')

model.save("model-01.keras")
print("model has been saved")

path_model = "PATH/TO/model-01.keras"
path_audio = "PATH/TO/WAV/FILE"

model = tf.keras.models.load_model(path_model)
model.summary()

spectrogram = get_spectrogram(path_audio)
spectrogram = spectrogram[np.newaxis, ...]

prediction = model(spectrogram)
print(f"Probability according to labels: {tf.nn.softmax(prediction[0])}")

# change according to labeled dataset
#labels = ['right', 'left', 'no', 'stop', 'down', 'go', 'up', 'yes', 'on', 'off']

# Find the class index with the highest probability
probabilities = tf.nn.softmax(prediction[0])
predicted_class_index = tf.argmax(probabilities).numpy()

# Get the corresponding label
predicted_label = label_names[predicted_class_index]
print(f"Predicted label: {predicted_label}")


# Prediction Method #2
predictions = model.predict(spectrogram)
prediction_max = tf.argmax(probabilities).numpy()
predicted_label = label_names[prediction_max]
print(f"second option: {predicted_label}")