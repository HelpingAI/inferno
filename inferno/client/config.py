"""
Configuration for the Inferno client.
"""

import os
from typing import Dict, Any, Optional


class InfernoConfig:
    """Configuration for the Inferno client."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the Inferno client configuration.
        
        Args:
            api_key: API key for authentication (not used by Inferno but kept for OpenAI compatibility)
            api_base: Base URL for the Inferno API
            api_version: API version (not used by Inferno but kept for OpenAI compatibility)
            organization: Organization ID (not used by Inferno but kept for OpenAI compatibility)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            default_headers: Default headers to include in all requests
        """
        # API key is not used by Inferno but kept for OpenAI compatibility
        self.api_key = api_key or os.environ.get("INFERNO_API_KEY", "dummy")
        
        # Base URL for the Inferno API
        self.api_base = api_base or os.environ.get("INFERNO_API_BASE", "http://localhost:8000/v1")
        
        # API version is not used by Inferno but kept for OpenAI compatibility
        self.api_version = api_version or os.environ.get("INFERNO_API_VERSION", "v1")
        
        # Organization ID is not used by Inferno but kept for OpenAI compatibility
        self.organization = organization or os.environ.get("INFERNO_ORGANIZATION")
        
        # Request timeout in seconds
        self.timeout = float(os.environ.get("INFERNO_TIMEOUT", timeout))
        
        # Maximum number of retries for failed requests
        self.max_retries = int(os.environ.get("INFERNO_MAX_RETRIES", max_retries))
        
        # Default headers to include in all requests
        self.default_headers = default_headers or {}
        
        # Add organization header if provided
        if self.organization:
            self.default_headers["Inferno-Organization"] = self.organization
    
    def get_headers(self, additional_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Get headers for API requests.
        
        Args:
            additional_headers: Additional headers to include in the request
            
        Returns:
            Dict[str, str]: Headers for the request
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.default_headers
        }
        
        if additional_headers:
            headers.update(additional_headers)
            
        return headers
