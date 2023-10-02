
import os
import boto3
import tarfile
import asyncio
from scramjet import streams

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

    # Zip the checkpoint folder
    source_dir = "temp_unzip"
    tarfile_dir = "temp/checkpoint.tar.gz"
    make_tarfile(source_dir, tarfile_dir)

    # Upload the new checkpoint tarfile
    up_path = "temp/checkpoint.tar.gz"
    upload_name = "checkpoint.tar.gz"
    upload_object(key, secret, bucket, up_path, upload_name)

    # list the new objects found on S3
    bucket_objects = checkpoint_search(key, secret, bucket)

    return streams.Stream.read_from(f"{bucket_objects}\n")



