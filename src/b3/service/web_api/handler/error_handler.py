"""
Standardized error handling utilities for API handlers.
Provides consistent error response formats across all handlers.
"""
import logging
from typing import Dict, Any, Tuple, Optional

from flask import jsonify


class APIError(Exception):
    """
    Base exception class for API errors.
    Provides consistent error structure.
    """

    def __init__(self, message: str, status_code: int = 400, details: Optional[Dict[str, Any]] = None):
        """
        Initialize API error.
        
        Args:
            message: Error message
            status_code: HTTP status code
            details: Additional error details
        """
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(APIError):
    """Exception raised for validation errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=400, details=details)


class NotFoundError(APIError):
    """Exception raised for resource not found errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=404, details=details)


class InternalServerError(APIError):
    """Exception raised for internal server errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


def create_error_response(message: str, status_code: int = 400,
                          details: Optional[Dict[str, Any]] = None) -> Tuple[Any, int]:
    """
    Create a standardized error response.
    
    Args:
        message: Error message
        status_code: HTTP status code
        details: Additional error details
        
    Returns:
        Tuple of (jsonify response, status_code)
    """
    response = {
        'status': 'error',
        'message': message
    }
    if details:
        response.update(details)

    return jsonify(response), status_code


def create_success_response(message: str, data: Optional[Dict[str, Any]] = None) -> Any:
    """
    Create a standardized success response.
    
    Args:
        message: Success message
        data: Additional response data
        
    Returns:
        jsonify response
    """
    response = {
        'status': 'success',
        'message': message
    }
    if data:
        response.update(data)

    return jsonify(response)


def handle_api_error(error: Exception, default_message: str = "An error occurred") -> Tuple[Any, int]:
    """
    Handle API errors and return standardized response.
    
    Args:
        error: Exception that occurred
        default_message: Default error message if error is not APIError
        
    Returns:
        Tuple of (jsonify response, status_code)
    """
    if isinstance(error, APIError):
        logging.error(f"API Error: {error.message} - {error.details}")
        return create_error_response(error.message, error.status_code, error.details)
    else:
        logging.error(f"Unexpected error: {str(error)}", exc_info=True)
        return create_error_response(default_message, status_code=500,
                                     details={'technical_details': str(error)})
