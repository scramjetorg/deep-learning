
import os
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np

label_names = ['seven', 'bird', 'stop', 'go', 'up', 'bed', 'dog', 'one', 'cat', 'happy', 'down', 'off', 'tree', 'house', 'two', 'eight', 'sheila', 'nine', 'yes', 'three', 'wow', 'marvin', 'on', 'five', 'zero', 'four', 'no', 'six', 'left', 'right']

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

    zero_padding = tf.zeros([16000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)

    spectrogram = tf.signal.stft(wav, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

path_model = "speech-intent-large-eval-isolated"

eval_data = "/home/shared/data_eval"

model = tf.keras.models.load_model(path_model)
model.summary()

success = 0
path_dataset = os.listdir(eval_data)
for folder in path_dataset:
    label = folder
    if label not in label_names:
        label_names.append(label)
    else:
        pass
    
    print("Processed folder: ", folder)
    path_folder = os.path.join(eval_data, folder)
    for file in os.listdir(path_folder):
        dir_index = path_dataset.index(folder)
        audio_dir = os.path.join(path_folder, file)
        spectrogram = get_spectrogram(audio_dir)
        spectrogram = spectrogram[np.newaxis, ...]
        prediction = model(spectrogram)
        print(f"Probability according to labels: {tf.nn.softmax(prediction[0])}")

        # Find the class index with the highest probability
        probabilities = tf.nn.softmax(prediction[0])
        predicted_class_index = tf.argmax(probabilities).numpy()

        # Get the corresponding label
        predicted_label = label_names[predicted_class_index]
        print(f"Predicted label: {predicted_label}")


        # Prediction Method #2
        predictions = model.predict(spectrogram)
        prediction_max = tf.argmax(probabilities).numpy()
        predicted_label2 = label_names[prediction_max]
        print(f"second option: {predicted_label2}")
        
        if predicted_label ==  folder or predicted_label2 == folder:
            success += 1


print("Numer of correct predictions: ", success)

