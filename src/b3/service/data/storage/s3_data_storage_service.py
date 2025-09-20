import os
import logging
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Union, BinaryIO
from io import BytesIO

from .data_storage_handler import DataStorageHandler


class S3DataStorageService(DataStorageHandler):
    """
    S3 implementation of DataStorageHandler for AWS S3 storage.
    """

    def __init__(self, bucket_name: str = None, region_name: str = None):
        """
        Initialize S3 storage service.
        
        Args:
            bucket_name: S3 bucket name (defaults to environment variable S3_BUCKET_NAME)
            region_name: AWS region (defaults to environment variable AWS_REGION or 'us-east-1')
        """
        self.bucket_name = bucket_name or os.getenv('S3_BUCKET_NAME', 'asset-predict-model')
        self.region_name = region_name or os.getenv('AWS_REGION', 'us-east-1')
        
        if not self.bucket_name:
            raise ValueError("S3 bucket name must be provided either as parameter or S3_BUCKET_NAME environment variable")
        
        try:
            self.s3_client = boto3.client('s3', region_name=self.region_name)
            # Test connection by checking if bucket exists
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logging.info(f"S3 storage service initialized with bucket: {self.bucket_name}")
        except NoCredentialsError:
            raise ValueError("AWS credentials not found. Please configure AWS credentials.")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise ValueError(f"S3 bucket '{self.bucket_name}' not found")
            else:
                raise ValueError(f"Error accessing S3 bucket: {e}")

    def save_file(self, file_path: str, data: Union[bytes, BinaryIO], 
                  content_type: str = None) -> str:
        """
        Save data to S3.
        
        Args:
            file_path: S3 key where the file should be saved
            data: Data to save (bytes or file-like object)
            content_type: MIME type of the content (optional)
            
        Returns:
            str: S3 URL where the file was saved
        """
        try:
            normalized_path = self._normalize_path(file_path)
            
            # Convert file-like object to bytes if needed
            if hasattr(data, 'read'):
                data = data.read()
            
            # Determine content type if not provided
            if not content_type:
                content_type = self._get_content_type(normalized_path)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=normalized_path,
                Body=data,
                ContentType=content_type
            )
            
            s3_url = f"s3://{self.bucket_name}/{normalized_path}"
            logging.info(f"File saved to S3: {s3_url}")
            return s3_url
            
        except ClientError as e:
            logging.error(f"Error saving file to S3: {e}")
            raise

    def load_file(self, file_path: str) -> bytes:
        """
        Load data from S3.
        
        Args:
            file_path: S3 key of the file to load
            
        Returns:
            bytes: The file content
        """
        try:
            normalized_path = self._normalize_path(file_path)
            
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=normalized_path
            )
            
            data = response['Body'].read()
            logging.info(f"File loaded from S3: {normalized_path}")
            return data
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"File not found in S3: {file_path}")
            else:
                logging.error(f"Error loading file from S3: {e}")
                raise

    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in S3.
        
        Args:
            file_path: S3 key to check
            
        Returns:
            bool: True if file exists, False otherwise
        """
        try:
            normalized_path = self._normalize_path(file_path)
            self.s3_client.head_object(Bucket=self.bucket_name, Key=normalized_path)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                logging.error(f"Error checking file existence in S3: {e}")
                raise

    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from S3.
        
        Args:
            file_path: S3 key of the file to delete
            
        Returns:
            bool: True if file was deleted successfully, False otherwise
        """
        try:
            normalized_path = self._normalize_path(file_path)
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=normalized_path)
            logging.info(f"File deleted from S3: {normalized_path}")
            return True
        except ClientError as e:
            logging.error(f"Error deleting file from S3: {e}")
            return False

    def create_directory(self, dir_path: str) -> bool:
        """
        Create a directory in S3 (S3 doesn't have real directories, but we can create a placeholder).
        
        Args:
            dir_path: Path of the directory to create
            
        Returns:
            bool: True if directory was created successfully, False otherwise
        """
        try:
            normalized_path = self._normalize_path(dir_path)
            # Ensure path ends with /
            if not normalized_path.endswith('/'):
                normalized_path += '/'
            
            # Create a placeholder object to represent the directory
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=normalized_path,
                Body=b'',
                ContentType='application/x-directory'
            )
            
            logging.info(f"Directory created in S3: {normalized_path}")
            return True
        except ClientError as e:
            logging.error(f"Error creating directory in S3: {e}")
            return False

    def get_file_url(self, file_path: str) -> str:
        """
        Get the S3 URL for a file.
        
        Args:
            file_path: S3 key of the file
            
        Returns:
            str: S3 URL to access the file
        """
        normalized_path = self._normalize_path(file_path)
        return f"s3://{self.bucket_name}/{normalized_path}"

    def _get_content_type(self, file_path: str) -> str:
        """
        Determine content type based on file extension.
        
        Args:
            file_path: File path to analyze
            
        Returns:
            str: MIME content type
        """
        extension = os.path.splitext(file_path)[1].lower()
        
        content_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.csv': 'text/csv',
            '.json': 'application/json',
            '.joblib': 'application/octet-stream',
            '.pkl': 'application/octet-stream',
            '.pickle': 'application/octet-stream'
        }
        
        return content_types.get(extension, 'application/octet-stream')
