"""
Cloud Storage Integration
Backup and export storage to AWS S3, Google Cloud Storage, Azure Blob

Version 1.0: Phase 2.7.3 - Cloud Storage Integration
- AWS S3 support
- Google Cloud Storage support
- Azure Blob Storage support
- Automatic backup uploads
- Export data management
"""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import storage as gcs_storage
    from google.oauth2 import service_account
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

try:
    from azure.storage.blob import BlobServiceClient, BlobClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

from ..utils.logger import ModuleLogger


class CloudStorageManager:
    """Unified cloud storage management for multiple providers"""
    
    def __init__(self, config: Dict, logger: logging.Logger = None):
        """
        Initialize cloud storage manager
        
        Args:
            config: Configuration dictionary with cloud provider settings
            logger: Logger instance
        """
        self.config = config
        self.logger = ModuleLogger('CloudStorage', logger or logging.getLogger(__name__))
        
        # Initialize storage clients
        self.s3_client = None
        self.gcs_client = None
        self.azure_client = None
        
        # Load configurations
        self._init_aws_s3()
        self._init_gcs()
        self._init_azure()
        
        self.logger.info(f"Cloud Storage initialized: "
                        f"AWS={'✓' if self.s3_client else '✗'}, "
                        f"GCS={'✓' if self.gcs_client else '✗'}, "
                        f"Azure={'✓' if self.azure_client else '✗'}")
    
    def _init_aws_s3(self):
        """Initialize AWS S3 client"""
        if not AWS_AVAILABLE:
            self.logger.warning("boto3 not installed - AWS S3 unavailable")
            return
        
        try:
            aws_config = self.config.get('cloud_storage', {}).get('aws', {})
            
            if not aws_config.get('enabled', False):
                return
            
            # Get credentials
            access_key = aws_config.get('access_key_id') or os.getenv('AWS_ACCESS_KEY_ID')
            secret_key = aws_config.get('secret_access_key') or os.getenv('AWS_SECRET_ACCESS_KEY')
            region = aws_config.get('region', 'us-east-1')
            
            if access_key and secret_key:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    region_name=region
                )
                
                self.s3_bucket = aws_config.get('bucket_name', 'falconone-backups')
                self.s3_prefix = aws_config.get('prefix', 'falconone/')
                
                self.logger.info(f"AWS S3 initialized: bucket={self.s3_bucket}, region={region}")
            else:
                self.logger.warning("AWS credentials not found - S3 disabled")
        
        except Exception as e:
            self.logger.error(f"AWS S3 initialization failed: {e}")
            self.s3_client = None
    
    def _init_gcs(self):
        """Initialize Google Cloud Storage client"""
        if not GCS_AVAILABLE:
            self.logger.warning("google-cloud-storage not installed - GCS unavailable")
            return
        
        try:
            gcs_config = self.config.get('cloud_storage', {}).get('gcs', {})
            
            if not gcs_config.get('enabled', False):
                return
            
            # Get credentials
            credentials_path = gcs_config.get('credentials_path') or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            
            if credentials_path and Path(credentials_path).exists():
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
                self.gcs_client = gcs_storage.Client(
                    credentials=credentials,
                    project=gcs_config.get('project_id')
                )
                
                self.gcs_bucket_name = gcs_config.get('bucket_name', 'falconone-backups')
                self.gcs_prefix = gcs_config.get('prefix', 'falconone/')
                
                self.logger.info(f"GCS initialized: bucket={self.gcs_bucket_name}")
            else:
                self.logger.warning("GCS credentials not found - GCS disabled")
        
        except Exception as e:
            self.logger.error(f"GCS initialization failed: {e}")
            self.gcs_client = None
    
    def _init_azure(self):
        """Initialize Azure Blob Storage client"""
        if not AZURE_AVAILABLE:
            self.logger.warning("azure-storage-blob not installed - Azure unavailable")
            return
        
        try:
            azure_config = self.config.get('cloud_storage', {}).get('azure', {})
            
            if not azure_config.get('enabled', False):
                return
            
            # Get connection string
            connection_string = azure_config.get('connection_string') or os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            
            if connection_string:
                self.azure_client = BlobServiceClient.from_connection_string(connection_string)
                
                self.azure_container = azure_config.get('container_name', 'falconone-backups')
                self.azure_prefix = azure_config.get('prefix', 'falconone/')
                
                self.logger.info(f"Azure Blob initialized: container={self.azure_container}")
            else:
                self.logger.warning("Azure connection string not found - Azure disabled")
        
        except Exception as e:
            self.logger.error(f"Azure Blob initialization failed: {e}")
            self.azure_client = None
    
    def upload_file(self, local_path: str, remote_path: str = None,
                   provider: str = 'auto') -> Dict[str, Any]:
        """
        Upload file to cloud storage
        
        Args:
            local_path: Local file path
            remote_path: Remote file path (if None, uses local filename)
            provider: Cloud provider ('aws', 'gcs', 'azure', 'auto')
        
        Returns:
            Upload result dictionary
        """
        try:
            if not Path(local_path).exists():
                return {'success': False, 'reason': 'file_not_found'}
            
            # Determine remote path
            if remote_path is None:
                remote_path = Path(local_path).name
            
            # Determine provider
            if provider == 'auto':
                if self.s3_client:
                    provider = 'aws'
                elif self.gcs_client:
                    provider = 'gcs'
                elif self.azure_client:
                    provider = 'azure'
                else:
                    return {'success': False, 'reason': 'no_provider_available'}
            
            # Upload to selected provider
            if provider == 'aws' and self.s3_client:
                return self._upload_to_s3(local_path, remote_path)
            elif provider == 'gcs' and self.gcs_client:
                return self._upload_to_gcs(local_path, remote_path)
            elif provider == 'azure' and self.azure_client:
                return self._upload_to_azure(local_path, remote_path)
            else:
                return {'success': False, 'reason': f'provider_{provider}_not_available'}
        
        except Exception as e:
            self.logger.error(f"Upload failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _upload_to_s3(self, local_path: str, remote_path: str) -> Dict[str, Any]:
        """Upload file to AWS S3"""
        try:
            key = self.s3_prefix + remote_path
            
            self.logger.info(f"Uploading to S3: {local_path} -> s3://{self.s3_bucket}/{key}")
            
            self.s3_client.upload_file(
                local_path,
                self.s3_bucket,
                key,
                ExtraArgs={'ServerSideEncryption': 'AES256'}
            )
            
            # Generate presigned URL (valid for 1 hour)
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.s3_bucket, 'Key': key},
                ExpiresIn=3600
            )
            
            file_size = Path(local_path).stat().st_size
            
            result = {
                'success': True,
                'provider': 'aws_s3',
                'bucket': self.s3_bucket,
                'key': key,
                'url': url,
                'file_size_bytes': file_size,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"S3 upload successful: {file_size} bytes")
            
            return result
        
        except ClientError as e:
            self.logger.error(f"S3 upload failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _upload_to_gcs(self, local_path: str, remote_path: str) -> Dict[str, Any]:
        """Upload file to Google Cloud Storage"""
        try:
            blob_name = self.gcs_prefix + remote_path
            
            self.logger.info(f"Uploading to GCS: {local_path} -> gs://{self.gcs_bucket_name}/{blob_name}")
            
            bucket = self.gcs_client.bucket(self.gcs_bucket_name)
            blob = bucket.blob(blob_name)
            
            blob.upload_from_filename(local_path)
            
            file_size = Path(local_path).stat().st_size
            
            result = {
                'success': True,
                'provider': 'gcs',
                'bucket': self.gcs_bucket_name,
                'blob_name': blob_name,
                'url': blob.public_url,
                'file_size_bytes': file_size,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"GCS upload successful: {file_size} bytes")
            
            return result
        
        except Exception as e:
            self.logger.error(f"GCS upload failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _upload_to_azure(self, local_path: str, remote_path: str) -> Dict[str, Any]:
        """Upload file to Azure Blob Storage"""
        try:
            blob_name = self.azure_prefix + remote_path
            
            self.logger.info(f"Uploading to Azure: {local_path} -> {self.azure_container}/{blob_name}")
            
            blob_client = self.azure_client.get_blob_client(
                container=self.azure_container,
                blob=blob_name
            )
            
            with open(local_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            
            file_size = Path(local_path).stat().st_size
            
            result = {
                'success': True,
                'provider': 'azure_blob',
                'container': self.azure_container,
                'blob_name': blob_name,
                'url': blob_client.url,
                'file_size_bytes': file_size,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Azure upload successful: {file_size} bytes")
            
            return result
        
        except Exception as e:
            self.logger.error(f"Azure upload failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def download_file(self, remote_path: str, local_path: str,
                     provider: str = 'auto') -> Dict[str, Any]:
        """
        Download file from cloud storage
        
        Args:
            remote_path: Remote file path
            local_path: Local destination path
            provider: Cloud provider
        
        Returns:
            Download result dictionary
        """
        try:
            # Determine provider
            if provider == 'auto':
                if self.s3_client:
                    provider = 'aws'
                elif self.gcs_client:
                    provider = 'gcs'
                elif self.azure_client:
                    provider = 'azure'
                else:
                    return {'success': False, 'reason': 'no_provider_available'}
            
            # Download from selected provider
            if provider == 'aws' and self.s3_client:
                return self._download_from_s3(remote_path, local_path)
            elif provider == 'gcs' and self.gcs_client:
                return self._download_from_gcs(remote_path, local_path)
            elif provider == 'azure' and self.azure_client:
                return self._download_from_azure(remote_path, local_path)
            else:
                return {'success': False, 'reason': f'provider_{provider}_not_available'}
        
        except Exception as e:
            self.logger.error(f"Download failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _download_from_s3(self, remote_path: str, local_path: str) -> Dict[str, Any]:
        """Download file from AWS S3"""
        try:
            key = self.s3_prefix + remote_path
            
            self.logger.info(f"Downloading from S3: s3://{self.s3_bucket}/{key} -> {local_path}")
            
            # Create parent directory if needed
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.s3_client.download_file(self.s3_bucket, key, local_path)
            
            file_size = Path(local_path).stat().st_size
            
            return {
                'success': True,
                'provider': 'aws_s3',
                'local_path': local_path,
                'file_size_bytes': file_size
            }
        
        except ClientError as e:
            self.logger.error(f"S3 download failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _download_from_gcs(self, remote_path: str, local_path: str) -> Dict[str, Any]:
        """Download file from Google Cloud Storage"""
        try:
            blob_name = self.gcs_prefix + remote_path
            
            self.logger.info(f"Downloading from GCS: gs://{self.gcs_bucket_name}/{blob_name} -> {local_path}")
            
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            bucket = self.gcs_client.bucket(self.gcs_bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)
            
            file_size = Path(local_path).stat().st_size
            
            return {
                'success': True,
                'provider': 'gcs',
                'local_path': local_path,
                'file_size_bytes': file_size
            }
        
        except Exception as e:
            self.logger.error(f"GCS download failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _download_from_azure(self, remote_path: str, local_path: str) -> Dict[str, Any]:
        """Download file from Azure Blob Storage"""
        try:
            blob_name = self.azure_prefix + remote_path
            
            self.logger.info(f"Downloading from Azure: {self.azure_container}/{blob_name} -> {local_path}")
            
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            blob_client = self.azure_client.get_blob_client(
                container=self.azure_container,
                blob=blob_name
            )
            
            with open(local_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            
            file_size = Path(local_path).stat().st_size
            
            return {
                'success': True,
                'provider': 'azure_blob',
                'local_path': local_path,
                'file_size_bytes': file_size
            }
        
        except Exception as e:
            self.logger.error(f"Azure download failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def list_files(self, prefix: str = '', provider: str = 'auto') -> Dict[str, Any]:
        """
        List files in cloud storage
        
        Args:
            prefix: File prefix filter
            provider: Cloud provider
        
        Returns:
            List of files
        """
        try:
            if provider == 'auto':
                if self.s3_client:
                    provider = 'aws'
                elif self.gcs_client:
                    provider = 'gcs'
                elif self.azure_client:
                    provider = 'azure'
                else:
                    return {'success': False, 'reason': 'no_provider_available'}
            
            if provider == 'aws' and self.s3_client:
                return self._list_s3_files(prefix)
            elif provider == 'gcs' and self.gcs_client:
                return self._list_gcs_files(prefix)
            elif provider == 'azure' and self.azure_client:
                return self._list_azure_files(prefix)
            else:
                return {'success': False, 'reason': f'provider_{provider}_not_available'}
        
        except Exception as e:
            self.logger.error(f"List files failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _list_s3_files(self, prefix: str) -> Dict[str, Any]:
        """List files in S3 bucket"""
        try:
            full_prefix = self.s3_prefix + prefix
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=full_prefix
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    files.append({
                        'key': obj['Key'],
                        'size_bytes': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat()
                    })
            
            return {
                'success': True,
                'provider': 'aws_s3',
                'count': len(files),
                'files': files
            }
        
        except ClientError as e:
            self.logger.error(f"S3 list failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _list_gcs_files(self, prefix: str) -> Dict[str, Any]:
        """List files in GCS bucket"""
        try:
            full_prefix = self.gcs_prefix + prefix
            
            bucket = self.gcs_client.bucket(self.gcs_bucket_name)
            blobs = bucket.list_blobs(prefix=full_prefix)
            
            files = []
            for blob in blobs:
                files.append({
                    'name': blob.name,
                    'size_bytes': blob.size,
                    'last_modified': blob.updated.isoformat() if blob.updated else None
                })
            
            return {
                'success': True,
                'provider': 'gcs',
                'count': len(files),
                'files': files
            }
        
        except Exception as e:
            self.logger.error(f"GCS list failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _list_azure_files(self, prefix: str) -> Dict[str, Any]:
        """List files in Azure container"""
        try:
            full_prefix = self.azure_prefix + prefix
            
            container_client = self.azure_client.get_container_client(self.azure_container)
            blobs = container_client.list_blobs(name_starts_with=full_prefix)
            
            files = []
            for blob in blobs:
                files.append({
                    'name': blob.name,
                    'size_bytes': blob.size,
                    'last_modified': blob.last_modified.isoformat() if blob.last_modified else None
                })
            
            return {
                'success': True,
                'provider': 'azure_blob',
                'count': len(files),
                'files': files
            }
        
        except Exception as e:
            self.logger.error(f"Azure list failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_available_providers(self) -> List[str]:
        """Get list of available cloud providers"""
        providers = []
        if self.s3_client:
            providers.append('aws')
        if self.gcs_client:
            providers.append('gcs')
        if self.azure_client:
            providers.append('azure')
        return providers
    
    def get_status(self) -> Dict[str, Any]:
        """Get cloud storage manager status"""
        return {
            'providers_available': self.get_available_providers(),
            'aws_s3': {
                'enabled': self.s3_client is not None,
                'bucket': self.s3_bucket if self.s3_client else None
            },
            'gcs': {
                'enabled': self.gcs_client is not None,
                'bucket': self.gcs_bucket_name if self.gcs_client else None
            },
            'azure_blob': {
                'enabled': self.azure_client is not None,
                'container': self.azure_container if self.azure_client else None
            }
        }
