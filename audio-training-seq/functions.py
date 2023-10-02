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

# zip checkpoint directory in a tar.gz file
def zip_tar(target_dir, zip_name, desired_dir):
    arcname = f"{desired_dir}/{zip_name}.tar.gz"
    with tarfile.open(arcname, "w:gz") as tar: # will automatically close tar archive
        for file in os.listdir(target_dir):
            file = os.path.join(target_dir, file)
            tar.add(os.path.dirname(file), arcname=arcname)

# unzip tar file
def unzip_tar(file_path, folder_name, desired_dir):
    # unzip the tar file
    file = tarfile.open(file_path)
    file.extractall(f"{desired_dir}/{folder_name}")
    file.close()

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