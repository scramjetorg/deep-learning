
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from scramjet import streams
import asyncio
from io import BytesIO
import base64


def load_wav_16k_mono(file_contents): #filename):
    # Load encoded wav file
#    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels) 
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

def preprocess(file_path): 
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

def analyse(buffer):
    model = tf.keras.models.load_model("model-prediction.keras")
    max_probability = { "value": 0, "label": "" }
    label_names = ['right', 'left', 'no', 'stop', 'down', 'go', 'up', 'yes', 'on', 'off']

    if detect_silence(buffer):
        return ""

    spectrogram = get_spectrogram_from_buffer(buffer)
    spectrogram = spectrogram[np.newaxis, ...]
    prediction = model(spectrogram)
    probabilities = tf.nn.softmax(prediction[0])
    predicted_class_index = tf.argmax(probabilities).numpy()

    # Get the corresponding label
    if probabilities[predicted_class_index] > 0.04:
        predicted_label = label_names[predicted_class_index]
        print(f"Predicted label: {predicted_label} {probabilities[predicted_class_index]}")

        if max_probability["value"] < probabilities[predicted_class_index]:
            max_probability["value"] = probabilities[predicted_class_index]
            max_probability["label"] = predicted_label

        return predicted_label
    else:
        return ""

def detect_silence(waveform):
    # Calculate the energy of the waveform
    energy = tf.reduce_sum(tf.square(waveform))

    # Calculate the length of the waveform
    length = tf.cast(tf.shape(waveform)[0], dtype=tf.float32)

    # Calculate the average energy of the waveform
    average_energy = energy / length

    # Define a threshold value for silence detection
    threshold = 0.001

    # Check if the average energy is below the threshold value
    is_silence = tf.less(average_energy, threshold)

    return is_silence

def process_chunk(chunk):
    buffer = BytesIO()
    wav_header = "UklGRtB6AABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0Yax6AAA="
    MIN_SIZE = 32 * 1024

    # set pointer at the buffer end and write chunk
    buffer.seek(0, 2)
    buffer.write(chunk)

    a = ""
    print(f"\Total buffer size {buffer.tell()} bytes")

    # buffer read pointer position
    pointer = buffer.tell()

    if pointer >= MIN_SIZE:
        buffer.seek(-MIN_SIZE, 2)
        sound_data = buffer.read(MIN_SIZE)
    else:
        buffer.seek(0, 0)
        sound_data = buffer.read()
        #return ""

    print(f'Sample length: {len(sound_data)}')
    header = BytesIO(base64.b64decode(wav_header))
    # write data on header end
    header.seek(0, 2)
    header.write(sound_data)
    # seek to header byte 40 (data length)
    header.seek(40, 0)
    # write sound data length
    header.write((len(sound_data)).to_bytes(4,'little'))
    # rewind and read from start (header + sound_data)
    header.seek(0, 0)
    data = header.read()
    print(f"data length {len(data)}")

    wav, sample_rate = tf.audio.decode_wav(data, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    # wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    wav = wav[:16000]
    a = analyse(wav)
    return a


def split_non_silent_audio(audio_data, sample_width=2, silence_threshold=0.001, sample_rate=16000, chunk_duration=1): 
    """
    Split 16-bit PCM audio data into 1-second intervals excluding silence.
    Args:
        audio_data (bytes): Binary audio data.
        sample_width (int): Width of each audio sample in bytes (default is 2 for 16-bit PCM).
        silence_threshold (int): Silence threshold (default is 256 for 16-bit PCM audio).
        sample_rate (int): Number of samples per second (default is 44100 Hz).
        chunk_duration (int): Duration of each chunk in seconds (default is 1 second).
    Returns:
        list: List of non-silent 1-second audio chunks in binary format.
    """
    samples_per_chunk = sample_rate * chunk_duration
    non_silent_chunks = []

    for i in range(0, len(audio_data), sample_width * samples_per_chunk):
        chunk = audio_data[i:i + sample_width * samples_per_chunk]
        samples = [int.from_bytes(chunk[j:j + sample_width], byteorder='little', signed=True)
                   for j in range(0, len(chunk), sample_width)]

        if any(abs(sample) >= silence_threshold for sample in samples):
            non_silent_chunks.append(chunk)

    return non_silent_chunks


async def run(context, input):

    predictions = []
    chunk_size = 1024 * 32
    audio_file = await input.reduce(lambda a, b: a+b)

    many_audio = split_non_silent_audio(audio_file)
    print(f"number of audio files {len(many_audio)}")

    for i in many_audio:
        processing_result = process_chunk(i)
        predictions.append(processing_result)

    # remove empty elements from the list
    predictions = [predict for predict in predictions if predict.strip() != '']
    print("Sequence completed ...")
    print(f"Predicted labels: {predictions}\n")

    return streams.Stream.read_from(f"{predictions}\n")



