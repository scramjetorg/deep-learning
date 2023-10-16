# Train model from input stream


import os
import boto3
#import tarfile
import asyncio
from scramjet import streams
import tarfile
#import pathlib
import time
from io import BytesIO
import base64


import tensorflow as tf
import tensorflow_io as tfio

from keras import layers
from keras import models
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


def create_model(input_shape, norm_layer):
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
    return model

# Save checkpoint
def save_checkpoint(model, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    checkpoint_manager.save()
    print(f"New checkpoint has been saved at {checkpoint_dir}")

# Load checkpoint if available
def load_checkpoint(model, checkpoint_dir):
    checkpoint = tf.train.Checkpoint(model=model)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint)
        print(f"Checkpoint '{latest_checkpoint}' restored!")



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

label_names = []
def get_record(audio_binary_files):

    labels = ['right', 'left', 'no', 'stop', 'down', 'go', 'up', 'yes', 'on', 'off']
#    audio_lst = split_audio(audio_binary_file)
    print(f"Length of Audio list: {len(audio_binary_files)}")
    for i in audio_binary_files:
        print(f"SHAPE: {i.shape()}")
#        label_num = 0
#        label_index = labels.index(label_num)
#        label_num += 1
        label_index = 0
        spectrogram = get_spectrogram_from_buffer(i)
        
        yield spectrogram, label_index
        label_index += 1


# Split audio binary into multiple audio binary list
def split_audio(audio_data, sample_width=2, silence_threshold=0.001, sample_rate=16000, chunk_duration=1): # silence_threshold=256, sample_rate=44100
    samples_per_chunk = sample_rate * chunk_duration
    non_silent_chunks = []

    for i in range(0, len(audio_data), sample_width * samples_per_chunk):
        chunk = audio_data[i:i + sample_width * samples_per_chunk]
        samples = [int.from_bytes(chunk[j:j + sample_width], byteorder='little', signed=True)
                   for j in range(0, len(chunk), sample_width)]

        if any(abs(sample) >= silence_threshold for sample in samples):
            non_silent_chunks.append(chunk)

    return non_silent_chunks


def process_chunk(chunk):
    buffer = BytesIO()
    wav_header = "UklGRtB6AABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0Yax6AAA="
    MIN_SIZE = 32 * 1024
    # set pointer at the buffer end and write chunk
    buffer.seek(0, 2)
    buffer.write(chunk)
    a = ""
    print(f"Total buffer size {buffer.tell()} bytes")
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
#    a = analyse(wav)
    return wav

print("[INFO:] Connecting to cloud")

def checkpoint_search(aws_key, aws_secret, bucket):
    session = boto3.session.Session()
    s3_client = session.client(
        service_name='s3',
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,

        )
    response = s3_client.list_objects(Bucket=bucket)
    print(response['ResponseMetadata']['HTTPStatusCode'])

    # search for a particular filename
    #response = s3_client.head_object(Bucket=bucket, Key='ckp_01.tar.gz')
    #print(response)

    bucket_lst = []
    try:
        for obj in response['Contents']:
            bucket_lst.append(obj['Key'])
            print(obj['Key'])
        print("Objects found in the bucket:")
        print(bucket_lst)
    except KeyError:
        print(f"No objects found in the bucket") 
    return bucket_lst


# download tar file from S3
def download_object(aws_key, aws_secret, bucket, object_name, downloaded_filename):
    session = boto3.session.Session()
    s3_client = session.client(
        service_name='s3',
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        )   
    #object_name = "ckp_01.tar.gz"
    #with open("ckpt_download.tar.gz", "wb") as file:
    with open(downloaded_filename, "wb") as file:
        try:
            response = s3_client.download_fileobj(bucket, object_name, file)
        except Exception as e:
            print(e)

#upload tar file to s3
def upload_object(aws_key, aws_secret, bucket, filename, uploaded_filename):
    session = boto3.session.Session()
    s3_client = session.client(
        service_name='s3',
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        )

    try:
        response = s3_client.upload_file(filename, bucket, uploaded_filename)
        print(f"Status code: {response}")
#    print(f"File {tar_file} uploaded to S3 bucket {bucket_name} as {access_key}")
    except Exception as e:
        print(f"Error: {e}")

        
# compress file into tar.gz
def make_tarfile(source_dir, tarfile_dir):
    with tarfile.open(tarfile_dir, "w:gz") as tar: # with will automatically close tar archive
        listdir = os.listdir(source_dir)
        for file in os.listdir(source_dir):
            file_path = os.path.join(source_dir, file)
            arcname = file
            tar.add(file_path, arcname=arcname) 
    print(f"Tar file created...")


# uncompress tar.gz file
def uncompress_tarfile(source_dir, desired_dir):
    try:
        dir = os.makedirs(desired_dir)
    except FileExistsError:
        pass
#    print(f" filename: {file.getnames()}")
    with tarfile.open(source_dir, "r") as tar:
        tar.extractall(desired_dir)
    print(f"Files extracted from tarfile completed...")


#checkpoint_search(key, secret, bucket)


async def run(context, input):
    key = ''
    secret = ''
    bucket = ''
    print(f"STARTING THE SEQUENCE")
    lst = checkpoint_search(key, secret, bucket)
    audio_file = await input.reduce(lambda a, b: a+b)


    #create directory if not existing
    try:
        path = "temp"
        dir = os.makedirs(path)
    except FileExistsError:
        pass

    #object = "model_ckpt.tar.gz"
    object = "checkpoint-10032023.tar.gz"
    #downloads = "/home/rnawfal/audio_model_01/model-seq/checkpoint_download.tar.gz"
    downloads = os.path.join(path, object) #"checkpoint_download.tar.gz")
    download_object(key, secret, bucket, object, downloads)

    #unzip tarfile
    #path_zipfile =  "/home/rnawfal/audio_model_01/model-seq/zip_checkpoint.tar.gz"
    source_dir = "temp/checkpoint-10032023.tar.gz"
    desired_dir = "temp_unzip"
    uncompress_tarfile(source_dir, desired_dir)

    lst = checkpoint_search(key, secret, bucket)

    # Dataset generator
#    dataset_path = "/home/rnawfal/data/mini_speech_commands"
    many_audio = split_audio(audio_file)
    audio_lst = []

    for i in many_audio:
        audio = process_chunk(i)
        audio_lst.append(audio)
    dataset_path = audio_lst
    dataset = tf.data.Dataset.from_generator(
        get_record,
        args=[dataset_path],
        output_signature=(
            tf.TensorSpec(shape=(124, 129, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)))

    # Buffer the dataset
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=10) #12600)
    dataset = dataset.batch(10) #32)
    dataset = dataset.prefetch(tf.data.AUTOTUNE) #8

    train = dataset.take(1) #160)
    test = dataset.take(1) # dataset.skip(160).take(30)

    print(f"Loading checkpoint to the Model...")
    # Model checkpoint load
    checkpoint_dir = "temp_unzip/cp.ckpt" 
    input_shape= (124, 129, 1)
    norm_layer = layers.Normalization()
    # with `Normalization.adapt`.
    norm_layer.adapt(data=train.map(map_func=lambda spec, label: spec))
    model = create_model(input_shape, norm_layer)
    load_checkpoint(model, checkpoint_dir)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(), #learning_rate=0.001), 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), #from_logits=True),
        metrics=['accuracy'],
    )

    history = model.fit(train, validation_steps=8, epochs=5, validation_data=test)

#    print(f"deleting temp_unzip folder")
#    os.rmdir("temp_unzip")
#    time.sleep(20)

    # save checkpoint after training
    checkpoint_dir = "checkpoint_folder"
    save_checkpoint(model, checkpoint_dir)

#    source_dir = "/home/rnawfal/audio_model_01/speech-to-intent/model_checkpoint"  
    source_dir = "temp_unzip"
    tarfile_dir = "temp/checkpoint.tar.gz" #"temp/zipfile01.tar.gz"
    make_tarfile(source_dir, tarfile_dir)

    up_object = "temp/checkpoint.tar.gz" #os.path.basename("temp/checkpoint_test_01.tar.gz")
    uploads = "checkpoint_01.tar.gz"
    upload_object(key, secret, bucket, up_object, uploads)

    return streams.Stream.read_from(f"{lst}\n") # should be a list

# delete if running as a Sequence

#path_audio = "/home/rnawfal/test-audio/multi-label-audio-02.wav"
#asyncio.run(run(None, path_audio))

# si inst input - deep-learning/audio-training-seq/onetime-predict/multi-label-audio-02.wav -e -t application/octet-stream