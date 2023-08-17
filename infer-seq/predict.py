
from io import BytesIO
import struct
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from scramjet.streams import Stream

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

def get_spectrogram_from_buffer(buffer):
    zero_padding = tf.zeros([16000] - tf.shape(buffer), dtype=tf.float32)
    buffer = tf.concat([zero_padding, buffer],0)
    # Use Short-time Fourier Transform
    spectrogram = tf.signal.stft(buffer, frame_length=255, frame_step=128)
    # Convert to absolut values (no negatives)
    spectrogram = tf.abs(spectrogram)
    # Add channel dimension (needed by CNN later)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

def get_spectrogram(file_path):
    # Load files into 16kHz mono
    wav = load_wav_16k_mono(file_path)
    # Only read the first 3 secs
    wav = wav[:16000]
    # If file < 3s add zeros
    return get_spectrogram_from_buffer(wav)


label_names = ['happy', 'on', 'off']

path_model = "/package/model-01.keras"
path_audio = "happy.wav"

model = tf.keras.models.load_model(path_model)
model.summary()

# spectrogram = get_spectrogram(path_audio)
# spectrogram = spectrogram[np.newaxis, ...]

# prediction = model(spectrogram)
# print(f"Probability according to labels: {tf.nn.softmax(prediction[0])}")

# change according to labeled dataset
#labels = ['right', 'left', 'no', 'stop', 'down', 'go', 'up', 'yes', 'on', 'off']

# Find the class index with the highest probability
# probabilities = tf.nn.softmax(prediction[0])
# predicted_class_index = tf.argmax(probabilities).numpy()

# # Get the corresponding label
# predicted_label = label_names[predicted_class_index]
# print(f"Predicted label: {predicted_label}")

def analyse(buffer):
    print("Analyse method")
    spectrogram = get_spectrogram_from_buffer(buffer)
    spectrogram = spectrogram[np.newaxis, ...]
    prediction = model(spectrogram)
    print(f"Probability according to labels: {tf.nn.softmax(prediction[0])}")
    probabilities = tf.nn.softmax(prediction[0])
    predicted_class_index = tf.argmax(probabilities).numpy()
    # Get the corresponding label
    predicted_label = label_names[predicted_class_index]
    print(f"Predicted label: {predicted_label}")
    return predicted_label

buffer = BytesIO()

def process_chunk(chunk):
    buffer.write(chunk)
    a = "Insufficient data"
    if buffer.tell() > 32 * 1024:
        buffer.seek(-32 * 1024, 2)
        sample = buffer.read()
        sample = np.frombuffer(sample, dtype=np.float32)
        # sample = struct.unpack('<' + 'H' * (len(sample) //  2), sample)
        a = analyse(sample)
    print("value of a")
    print(a)
    return a

provides = {
    "provides": "predictions",
    "contentType": "application/octet-stream"
}

async def run(context, input):
    return input.map(lambda chunk: process_chunk(chunk).encode())

# Prediction Method #2
# predictions = model.predict(spectrogram)
# prediction_max = tf.argmax(probabilities).numpy()
# predicted_label = label_names[prediction_max]
# print(f"second option: {predicted_label}")
