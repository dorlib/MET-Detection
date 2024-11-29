import boto3
import os
from botocore.exceptions import NoCredentialsError


class S3Client:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION")
        )
        self.bucket_name = os.getenv("S3_BUCKET_NAME")

    def upload_file(self, file_content, file_name):
        try:
            self.s3.put_object(Bucket=self.bucket_name, Key=file_name, Body=file_content)
            return f"https://{self.bucket_name}.s3.amazonaws.com/{file_name}"
        except NoCredentialsError:
            print("Credentials not available")
            return None
