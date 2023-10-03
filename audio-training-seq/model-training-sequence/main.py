
import os
import boto3
import tarfile
import asyncio
from scramjet import streams

import tensorflow as tf
import tensorflow_io as tfio

from keras import layers
from keras import models
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


# Sequential model 
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
            spectrogram = preprocess(audio_dir) 
            yield spectrogram, dir_index

def checkpoint_search(aws_key, aws_secret, bucket):
    print("[INFO:] Connecting to cloud")
    session = boto3.session.Session()
    s3_client = session.client(
        service_name='s3',
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,

        )
    response = s3_client.list_objects(Bucket=bucket)
    print(response['ResponseMetadata']['HTTPStatusCode'])

    bucket_lst = []
    try:
        for obj in response['Contents']:
            bucket_lst.append(obj['Key'])
            print(obj['Key'])
        print("Objects found in the bucket:")
    except KeyError:
        print(f"No objects found in the bucket")
    print(bucket_lst) 
    return bucket_lst

# Download tar file from s3
def download_object(aws_key, aws_secret, bucket, object_name, downloaded_filename):
    session = boto3.session.Session()
    s3_client = session.client(
        service_name='s3',
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        )   
    with open(downloaded_filename, "wb") as file:
        response = s3_client.download_fileobj(bucket, object_name, file)
        print(f"Object file downloaded. Error code: {response}")

# Upload tar file to s3
def upload_object(aws_key, aws_secret, bucket, file_path, uploaded_filename):
    session = boto3.session.Session()
    s3_client = session.client(
        service_name='s3',
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        )

    try:
        response = s3_client.upload_file(file_path, bucket, uploaded_filename)
        print(f"Object file uploaded. Error code: {response}")
    except Exception as e:
        print(f"Error: {e}")

        
# Compress checkpoint files into a tar.gz format
def make_tarfile(source_dir, tarfile_dir):
    with tarfile.open(tarfile_dir, "w:gz") as tar: # with will automatically close tar archive
        listdir = os.listdir(source_dir)
        for file in os.listdir(source_dir):
            file_path = os.path.join(source_dir, file)
            arcname = file
            tar.add(file_path, arcname=arcname) 
    print(f"Tar file created...")


# Uncompress the checkpoint tar.gz file 
def uncompress_tarfile(source_dir, desired_dir):
    try:
        dir = os.makedirs(desired_dir)
    except FileExistsError:
        pass
    with tarfile.open(source_dir, "r") as tar:
        tar.extractall(desired_dir)
    print(f"Files extracted from tarfile completed...")


async def run(context, input, key, secret, bucket, object):

    dataset_path = "/PATH/TO/FOLDER/data/mini_speech_commands"

    key = key
    secret = secret
    bucket = bucket
    print(f"STARTING THE SEQUENCE")
    lst = checkpoint_search(key, secret, bucket)
    
    #create directory if not existing
    path = "temp"
    os.makedirs(path, exist_ok=True)

    object = object
    download_path = os.path.join(path, object) 
    download_object(key, secret, bucket, object, download_path)

    # Unzip tarfile
    file_path = "/".join([path, object])
    source_dir = file_path
    desired_dir = "temp_unzip"
    uncompress_tarfile(source_dir, desired_dir)

    # Dataset generator    
    dataset = tf.data.Dataset.from_generator(
        get_record,
        args=[dataset_path],
        output_signature=(
            tf.TensorSpec(shape=(124, 129, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)))
    
    # Buffer the dataset
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=12600)
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(tf.data.AUTOTUNE) 

    train = dataset.take(160)
    test = dataset.skip(160).take(30)

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
        optimizer=tf.keras.optimizers.Adam(), 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
        metrics=['accuracy'],
    )

    history = model.fit(train, validation_steps=8, epochs=20, validation_data=test)

    # save checkpoint after training
    checkpoint_dir = "checkpoints"
    save_checkpoint(model, checkpoint_dir)

    # Zip the checkpoint folder
    source_dir = "checkpoints"
    tarfile_dir = "temp/checkpoint.tar.gz"
    make_tarfile(source_dir, tarfile_dir)

    # Upload the new checkpoint tarfile
    path = "temp/checkpoint.tar.gz"
    upload_name = "checkpoint.tar.gz"
    upload_object(key, secret, bucket, path, upload_name)

    # list the new objects found on S3
    bucket_objects = checkpoint_search(key, secret, bucket)

    return streams.Stream.read_from(f"{bucket_objects}\n")



