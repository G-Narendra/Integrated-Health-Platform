"""
Authentication middleware for API endpoints.
Provides token-based authentication with role-based access control.
"""

from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict

security = HTTPBearer(auto_error=False)


def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> Dict:
    """
    Verify API token and return user context.
    Simplified for development - use proper OAuth2/JWT in production.
    """
    if credentials is None:
        # Allow unauthenticated access in dev mode
        return {
            "user_id": "API-USER",
            "role": "system",
            "facility": "UAE Integrated Healthcare Platform",
        }

    token = credentials.credentials

    # Simple token validation
    if token == "dev-token":
        return {
            "user_id": "DR-001",
            "role": "physician",
            "facility": "Cleveland Clinic Abu Dhabi",
        }

    raise HTTPException(status_code=401, detail="Invalid authentication token")
