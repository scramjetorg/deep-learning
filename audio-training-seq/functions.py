import os
import tarfile
import boto3

key = ' '
secret = ' '
bucket = ' '
local_path = ' '
object_key = ' ' # checkpoint.tar.gz

# list all object in S3
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
    response = s3_client.head_object(Bucket=bucket, Key=object_key)
    print(response)

    bucket_lst = []
    if 'Contents' in response:
        print("Objects found in the bucket:")
        for obj in response['Contents']:
            bucket_lst.append(obj['Key'])
    else:
        return f"No objects found in the bucket"
    print(bucket_lst) 
    return bucket_lst

# get checkpoint from S3
def get_s3_object(aws_key, aws_secret, bucket, object_key, local_path):
    session = boto3.session.Session()
    s3_client = session.client(
        service_name='s3',
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        )
    response = s3_client.head_object(Bucket=bucket, Key=object_key)
    s3_client.download_file(bucket, object_key, local_path)

# upload object to S3
def upload_s3_object(aws_key, aws_secret, bucket, object_key, path_to_object):
    session = boto3.session.Session()
    s3_client = session.client(
        service_name='s3',
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        )
    try:
        response = s3_client.upload_file(path_to_object, bucket, object_key)
        print(f"Status code: {response}")
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
    os.makedirs(desired_dir, exist_ok=True)
    with tarfile.open(source_dir, "r") as tar:
        tar.extractall(desired_dir)
    print(f"Files extracted from tarfile completed...")

# delete S3 object
def delete_s3_object(aws_key, aws_secret, bucket, object_key):
    session = boto3.session.Session()
    s3_client = session.client(
        service_name='s3',
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        )
    try:
        response = s3_client.delete_object(Bucket=bucket, Key=object_key)
        if response['ResponseMetadata']['HTTPStatusCode'] == 204:
            print(f"Object {object_key} was not found")
        else:
            print(f"Object {object_key} deleted from S3 bucket {bucket}")
    except Exception as e:
        print(f"Error: {e}")


# return audio file size
# header = 44
# audio_bytes =  await input.read(header)
def get_audio_size(audio_bytes: bytes, start_index: int, end_index: int) -> int: 
        arr_bytes = audio_bytes[start_index:end_index]
        integer = int.from_bytes(arr_bytes, byteorder='little', signed=False)
        print(f"data size in bytes: {integer}\n")
        return integer
