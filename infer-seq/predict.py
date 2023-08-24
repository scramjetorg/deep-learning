
import base64
from io import BytesIO
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np

wav_header = "UklGRtB6AABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0Yax6AAA="

label_names = ['seven', 'bird', 'stop', 'go', 'up', 'bed', 'dog', 'one', 'cat', 'happy', 'down', 'off', 'tree', 'house', 'two', 'eight', 'sheila', 'nine', 'yes', 'three', 'wow', 'marvin', 'on', 'five', 'zero', 'four', 'no', 'six', 'left', 'right']
max_probability = { "value": 0, "label": "" }

path_model = "speech-intent-large-eval-isolated.keras"
# path_audio = "on.wav"
# path_audio = "happy.wav"
path_audio = "happy-happy-on-off-mono.wav"
# path_audio = "honoff.wav"
# path_audio = "off-happy-of.wav"

model = tf.keras.models.load_model(path_model)
buffer = BytesIO()

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

def run(context, input):
    predictions = []
    chunk_size = 1024 * 32

    with input:
        first_chunk = True
        while True:
            if first_chunk:
                # drop WAV header
                data = input.read(chunk_size)[44:]
                first_chunk = False
            else:
                data = input.read(chunk_size)

            if not data:
                break

            processing_result = process_chunk(data)

            if (processing_result != ""):
                predictions.append(processing_result)

    print(predictions)
    print(f'Max probability {max_probability["value"]}')

# Remove this in Sequence
run(None, open(path_audio, 'rb'))

provides = {
    "provides": "predictions",
    "contentType": "application/octet-stream"
}
