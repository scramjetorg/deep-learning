
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from scramjet import streams
import asyncio



def load_wav_16k_mono(file_contents): #filename):
    # Load encoded wav file
#    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels) 
    # input audio file is binary
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

def get_audio_spectrogram(file_path): 
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

async def run(context, input):

    audio_file = await input.reduce(lambda a, b: a+b)
    audio = get_audio_spectrogram(audio_file)
    # Adding batch dimension using np.newaxis
    audio = audio[np.newaxis, ...]

    model = tf.keras.models.load_model("model-prediction.keras")
    prediction = model(audio)
    print(f"Probability according to labels: {tf.nn.softmax(prediction[0])}")
    # change according to labeled dataset
    labels = ['right', 'left', 'no', 'stop', 'down', 'go', 'up', 'yes', 'on', 'off']

    # Find the class index with the highest probability
    probabilities = tf.nn.softmax(prediction[0])
    predicted_class_index = tf.argmax(probabilities).numpy()

    # Get the corresponding label
    predicted_label = labels[predicted_class_index]
    print(f"Predicted label: {predicted_label}")

    return streams.Stream.read_from(f"Predicted label: {predicted_label}\n")


# si inst input - /PATH/TO/SAMPLE/AUDIO/off-16.36.15.wav -e -t application/octet-stream