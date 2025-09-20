# Storage Abstraction System

This module provides a flexible storage abstraction that allows the B3 model system to save files to either local file system or AWS S3 based on environment configuration.

## Overview

The storage system automatically determines where to save files based on the `enviroment` environment variable:
- `enviroment=AWS`: Files are saved to AWS S3
- `enviroment=SYSTEM` or unset: Files are saved to local file system

## Architecture

### Core Components

1. **DataStorageHandler** (Abstract Base Class)
   - Defines the interface for all storage implementations
   - Provides common functionality like path normalization

2. **DataStorageService** (Factory/Coordinator)
   - Determines which storage backend to use based on environment
   - Provides a unified interface for the rest of the application

3. **S3DataStorageService** (S3 Implementation)
   - Handles file operations with AWS S3
   - Requires AWS credentials and S3 bucket configuration

4. **SystemDataStorageService** (Local Implementation)
   - Handles file operations with local file system
   - Default behavior when not using S3

## Usage

### Basic Usage

```python
from b3.service.data.storage.data_storage_service import DataStorageService

# Initialize storage service (automatically detects environment)
storage_service = DataStorageService()

# Save a file
data = b"Hello, World!"
saved_path = storage_service.save_file("test/file.txt", data, "text/plain")

# Load a file
loaded_data = storage_service.load_file("test/file.txt")

# Check if file exists
exists = storage_service.file_exists("test/file.txt")

# Delete a file
deleted = storage_service.delete_file("test/file.txt")
```

### With Model System

```python
from b3.service.model.model import B3Model

# Model automatically uses the configured storage
model = B3Model()

# Training will save model and plots to the appropriate storage
model.train()
```

## Environment Configuration

### For S3 Storage

Set these environment variables:

```bash
export enviroment=AWS
export S3_BUCKET_NAME=your-bucket-name
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_REGION=us-east-1  # Optional, defaults to us-east-1
```

### For Local Storage

```bash
export enviroment=SYSTEM
# or leave enviroment unset
```

## File Types Supported

The system supports various file types with appropriate MIME type detection:

- **Images**: PNG, JPEG, GIF
- **Models**: Joblib, Pickle files
- **Data**: CSV, JSON, TXT
- **Documents**: PDF
- **Generic**: Binary files

## Integration Points

The storage system is integrated into:

1. **Model Saving Service** (`model_saving_service.py`)
   - Saves trained models using the configured storage

2. **Plotter** (`plotter.py`)
   - Saves generated plots using the configured storage

3. **Model Evaluation Service** (`model_evaluation_service.py`)
   - Saves evaluation visualizations using the configured storage

## Error Handling

The system includes comprehensive error handling:

- **S3 Errors**: Handles AWS credential issues, bucket access, and network problems
- **Local Storage Errors**: Handles file system permissions and disk space issues
- **File Not Found**: Consistent error handling across storage types

This will demonstrate the storage system with your current environment configuration.

## Dependencies

### For S3 Storage
- `boto3`: AWS SDK for Python
- AWS credentials configured (via environment variables, AWS CLI, or IAM roles)

### For Local Storage
- No additional dependencies beyond standard Python libraries

## Migration Notes

The storage system is backward compatible. Existing code will continue to work with local storage by default. To enable S3 storage, simply set the `enviroment=AWS` environment variable and configure AWS credentials.
