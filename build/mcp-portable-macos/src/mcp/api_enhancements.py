#!/usr/bin/env python3
"""
API Enhancements for MCP Server
Provides enhanced API features including:
- Bulk action safety mechanisms
- Rate limiting and authentication
- API versioning
- Comprehensive error handling
- Input validation and sanitization
"""

import time
import hashlib
import os
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from functools import wraps
import re
try:
    import jwt
except ImportError:
    jwt = None

@dataclass
class APIError:
    """Standardized API error structure."""
    code: int
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()

@dataclass
class APIVersion:
    """API version information."""
    version: str
    deprecated: bool = False
    sunset_date: Optional[str] = None
    migration_guide: Optional[str] = None

class APIVersionManager:
    """Manages API versioning and compatibility."""
    
    def __init__(self):
        self.versions = {
            "1.0": APIVersion("1.0", deprecated=False),
            "1.1": APIVersion("1.1", deprecated=False),
            "2.0": APIVersion("2.0", deprecated=False)
        }
        self.default_version = "2.0"
        self.supported_versions = ["1.0", "1.1", "2.0"]
    
    def is_version_supported(self, version: str) -> bool:
        """Check if API version is supported."""
        return version in self.supported_versions
    
    def get_version_info(self, version: str) -> Optional[APIVersion]:
        """Get version information."""
        return self.versions.get(version)
    
    def get_deprecated_versions(self) -> List[str]:
        """Get list of deprecated versions."""
        return [v for v, info in self.versions.items() if info.deprecated]

class BulkActionSafety:
    """Enhanced bulk action safety mechanisms."""
    
    def __init__(self):
        self.max_items_per_bulk = 100
        self.max_critical_items = 5
        self.max_concurrent_bulk_ops = 3
        self.bulk_operation_timeout = 300  # 5 minutes
        self.active_bulk_ops = {}
    
    def validate_bulk_request(self, action_type: str, items: List[Dict], user_id: str) -> Dict[str, Any]:
        """Validate bulk action request for safety."""
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "requires_confirmation": False
        }
        
        # Check item count
        if len(items) > self.max_items_per_bulk:
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Too many items ({len(items)}). Maximum allowed: {self.max_items_per_bulk}"
            )
        
        # Check for critical items
        critical_items = [item for item in items if item.get("accuracy_critical")]
        if len(critical_items) > self.max_critical_items:
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Too many critical items ({len(critical_items)}). Maximum allowed: {self.max_critical_items}"
            )
        
        # Check concurrent operations
        active_ops = len([op for op in self.active_bulk_ops.values() if op["user_id"] == user_id])
        if active_ops >= self.max_concurrent_bulk_ops:
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Too many concurrent bulk operations ({active_ops}). Maximum allowed: {self.max_concurrent_bulk_ops}"
            )
        
        # Check for potentially dangerous operations
        if action_type in ["delete", "update_critical"] and len(items) > 10:
            validation_result["requires_confirmation"] = True
            validation_result["warnings"].append(
                f"Large {action_type} operation detected. Confirmation required."
            )
        
        return validation_result
    
    def start_bulk_operation(self, operation_id: str, user_id: str, action_type: str) -> bool:
        """Start a bulk operation and track it."""
        if len(self.active_bulk_ops) >= self.max_concurrent_bulk_ops:
            return False
        
        self.active_bulk_ops[operation_id] = {
            "user_id": user_id,
            "action_type": action_type,
            "start_time": time.time(),
            "status": "running"
        }
        return True
    
    def complete_bulk_operation(self, operation_id: str) -> None:
        """Mark bulk operation as complete."""
        if operation_id in self.active_bulk_ops:
            del self.active_bulk_ops[operation_id]
    
    def cleanup_expired_operations(self) -> None:
        """Clean up expired bulk operations."""
        current_time = time.time()
        expired_ops = [
            op_id for op_id, op_data in self.active_bulk_ops.items()
            if current_time - op_data["start_time"] > self.bulk_operation_timeout
        ]
        for op_id in expired_ops:
            del self.active_bulk_ops[op_id]

class EnhancedRateLimiter:
    """Enhanced rate limiting with multiple strategies."""
    
    def __init__(self):
        self.rate_limits = {
            "default": {"requests_per_minute": 60, "burst_limit": 10},
            "bulk_operations": {"requests_per_minute": 10, "burst_limit": 2},
            "critical_operations": {"requests_per_minute": 5, "burst_limit": 1},
            "admin": {"requests_per_minute": 200, "burst_limit": 20}
        }
        self.user_limits = {}
        self.ip_limits = {}
        self.operation_tracking = {}
    
    def get_rate_limit(self, user_type: str = "default") -> Dict[str, int]:
        """Get rate limit configuration for user type."""
        return self.rate_limits.get(user_type, self.rate_limits["default"])
    
    def check_rate_limit(self, user_id: str, ip: str, operation_type: str = "default") -> bool:
        """Check if request is within rate limits."""
        current_time = time.time()
        
        # Get appropriate rate limit
        if operation_type in ["bulk_action", "delete", "update_critical"]:
            limit_config = self.get_rate_limit("bulk_operations")
        elif operation_type in ["admin", "system"]:
            limit_config = self.get_rate_limit("admin")
        else:
            limit_config = self.get_rate_limit("default")
        
        # Check user-based rate limit
        if user_id not in self.user_limits:
            self.user_limits[user_id] = {"count": 0, "reset": current_time + 60}
        
        user_limit = self.user_limits[user_id]
        if current_time > user_limit["reset"]:
            user_limit["count"] = 0
            user_limit["reset"] = current_time + 60
        
        if user_limit["count"] >= limit_config["requests_per_minute"]:
            return False
        
        # Check IP-based rate limit
        if ip not in self.ip_limits:
            self.ip_limits[ip] = {"count": 0, "reset": current_time + 60}
        
        ip_limit = self.ip_limits[ip]
        if current_time > ip_limit["reset"]:
            ip_limit["count"] = 0
            ip_limit["reset"] = current_time + 60
        
        if ip_limit["count"] >= limit_config["requests_per_minute"]:
            return False
        
        # Update counters
        user_limit["count"] += 1
        ip_limit["count"] += 1
        
        return True

class InputValidator:
    """Input validation and sanitization."""
    
    def __init__(self):
        self.sanitization_patterns = {
            "sql_injection": re.compile(r"(\b(union|select|insert|update|delete|drop|create|alter)\b)", re.IGNORECASE),
            "xss": re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE),
            "path_traversal": re.compile(r"\.\./|\.\.\\"),
            "command_injection": re.compile(r"[;&|`$()]")
        }
        
        self.validation_schemas = {
            "task": {
                "title": {"type": str, "max_length": 200, "required": True},
                "description": {"type": str, "max_length": 2000, "required": False},
                "priority": {"type": int, "min": 1, "max": 10, "required": False},
                "tags": {"type": list, "max_items": 10, "required": False}
            },
            "memory": {
                "text": {"type": str, "max_length": 10000, "required": True},
                "memory_type": {"type": str, "allowed_values": ["general", "task", "research", "feedback"], "required": False},
                "priority": {"type": float, "min": 0.0, "max": 1.0, "required": False}
            },
            "project": {
                "name": {"type": str, "max_length": 100, "required": True},
                "description": {"type": str, "max_length": 1000, "required": False}
            }
        }
    
    def sanitize_input(self, data: Any) -> Any:
        """Sanitize input data to prevent injection attacks."""
        if isinstance(data, str):
            # Remove potentially dangerous patterns
            for pattern_name, pattern in self.sanitization_patterns.items():
                data = pattern.sub("", data)
            return data
        elif isinstance(data, dict):
            return {k: self.sanitize_input(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_input(item) for item in data]
        else:
            return data
    
    def validate_schema(self, data: Dict[str, Any], schema_name: str) -> Dict[str, Any]:
        """Validate data against a schema."""
        schema = self.validation_schemas.get(schema_name)
        if not schema:
            return {"valid": False, "errors": [f"Unknown schema: {schema_name}"]}
        
        validation_result = {"valid": True, "errors": []}
        
        for field_name, field_config in schema.items():
            if field_config.get("required", False) and field_name not in data:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Required field missing: {field_name}")
                continue
            
            if field_name in data:
                value = data[field_name]
                
                # Type validation
                expected_type = field_config["type"]
                if not isinstance(value, expected_type):
                    validation_result["valid"] = False
                    validation_result["errors"].append(
                        f"Field {field_name} must be {expected_type.__name__}, got {type(value).__name__}"
                    )
                    continue
                
                # String length validation
                if expected_type == str and "max_length" in field_config:
                    if len(value) > field_config["max_length"]:
                        validation_result["valid"] = False
                        validation_result["errors"].append(
                            f"Field {field_name} too long (max {field_config['max_length']})"
                        )
                
                # Numeric range validation
                if expected_type in [int, float]:
                    if "min" in field_config and value < field_config["min"]:
                        validation_result["valid"] = False
                        validation_result["errors"].append(
                            f"Field {field_name} too small (min {field_config['min']})"
                        )
                    if "max" in field_config and value > field_config["max"]:
                        validation_result["valid"] = False
                        validation_result["errors"].append(
                            f"Field {field_name} too large (max {field_config['max']})"
                        )
                
                # List validation
                if expected_type == list:
                    if "max_items" in field_config and len(value) > field_config["max_items"]:
                        validation_result["valid"] = False
                        validation_result["errors"].append(
                            f"Field {field_name} has too many items (max {field_config['max_items']})"
                        )
                
                # Allowed values validation
                if "allowed_values" in field_config and value not in field_config["allowed_values"]:
                    validation_result["valid"] = False
                    validation_result["errors"].append(
                        f"Field {field_name} has invalid value. Allowed: {field_config['allowed_values']}"
                    )
        
        return validation_result

class EnhancedAuthentication:
    """Enhanced authentication with multiple methods."""
    
    def __init__(self):
        self.api_keys = {}
        self.jwt_secret = os.environ.get("MCP_JWT_SECRET", "default_secret_change_in_production")
        self.session_timeout = 3600  # 1 hour
        self.active_sessions = {}
        self._load_api_keys()
    
    def _load_api_keys(self):
        """Load API keys from environment and config files."""
        # Load from environment
        env_keys = os.environ.get("MCP_API_KEYS")
        if env_keys:
            for key in env_keys.split(","):
                self.api_keys[key.strip()] = {"type": "api_key", "permissions": ["read", "write"]}
        
        # Load from config file
        config_path = os.path.join(os.getcwd(), "mcp_api_keys.cfg")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        self.api_keys[line] = {"type": "api_key", "permissions": ["read", "write"]}
    
    def authenticate_api_key(self, api_key: str) -> Dict[str, Any]:
        """Authenticate using API key."""
        if api_key in self.api_keys:
            return {
                "authenticated": True,
                "user_id": f"api_user_{hashlib.md5(api_key.encode()).hexdigest()[:8]}",
                "permissions": self.api_keys[api_key]["permissions"],
                "auth_method": "api_key"
            }
        return {"authenticated": False, "error": "Invalid API key"}
    
    def authenticate_jwt(self, token: str) -> Dict[str, Any]:
        """Authenticate using JWT token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Check if session is still valid
            if payload.get("exp", 0) < time.time():
                return {"authenticated": False, "error": "Token expired"}
            
            session_id = payload.get("session_id")
            if session_id in self.active_sessions:
                return {
                    "authenticated": True,
                    "user_id": payload.get("user_id"),
                    "permissions": payload.get("permissions", ["read"]),
                    "auth_method": "jwt"
                }
            
            return {"authenticated": False, "error": "Invalid session"}
            
        except jwt.InvalidTokenError:
            return {"authenticated": False, "error": "Invalid token"}
    
    def check_permission(self, auth_result: Dict[str, Any], required_permission: str) -> bool:
        """Check if user has required permission."""
        if not auth_result.get("authenticated"):
            return False
        
        permissions = auth_result.get("permissions", [])
        return required_permission in permissions or "admin" in permissions

class APIErrorHandler:
    """Comprehensive error handling for API requests."""
    
    def __init__(self):
        self.logger = logging.getLogger("api_error_handler")
        self.error_codes = {
            "validation_error": 400,
            "authentication_error": 401,
            "authorization_error": 403,
            "rate_limit_error": 429,
            "bulk_action_error": 413,
            "internal_error": 500,
            "service_unavailable": 503
        }
    
    def handle_error(self, error_type: str, message: str, details: Optional[Dict[str, Any]] = None) -> APIError:
        """Create standardized error response."""
        error = APIError(
            code=self.error_codes.get(error_type, 500),
            message=message,
            details=details
        )
        
        self.logger.error(f"API Error: {error_type} - {message}", extra={"error_details": details})
        return error
    
    def handle_exception(self, exception: Exception, context: str = "") -> APIError:
        """Handle unexpected exceptions."""
        error_type = "internal_error"
        message = f"Unexpected error in {context}: {str(exception)}"
        
        # Don't expose internal details in production
        if os.environ.get("MCP_ENVIRONMENT") == "production":
            details = {"context": context}
        else:
            details = {
                "context": context,
                "exception_type": type(exception).__name__,
                "exception_message": str(exception)
            }
        
        return self.handle_error(error_type, message, details)

def api_decorator(version: str = "2.0", require_auth: bool = True, 
                 required_permission: str = "read", validate_input: bool = True,
                 schema_name: Optional[str] = None):
    """Decorator for API endpoints with enhanced features."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, params: Dict[str, Any], *args, **kwargs):
            # Initialize components
            version_manager = APIVersionManager()
            auth = EnhancedAuthentication()
            validator = InputValidator()
            error_handler = APIErrorHandler()
            rate_limiter = EnhancedRateLimiter()
            
            try:
                # API versioning
                api_version = params.get("api_version", "2.0")
                if not version_manager.is_version_supported(api_version):
                    return asdict(error_handler.handle_error(
                        "validation_error", 
                        f"Unsupported API version: {api_version}"
                    ))
                
                # Authentication
                if require_auth:
                    api_key = params.get("api_key")
                    jwt_token = params.get("jwt_token")
                    
                    if api_key:
                        auth_result = auth.authenticate_api_key(api_key)
                    elif jwt_token:
                        auth_result = auth.authenticate_jwt(jwt_token)
                    else:
                        return asdict(error_handler.handle_error(
                            "authentication_error", 
                            "Authentication required"
                        ))
                    
                    if not auth_result["authenticated"]:
                        return asdict(error_handler.handle_error(
                            "authentication_error", 
                            auth_result["error"]
                        ))
                    
                    # Permission check
                    if not auth.check_permission(auth_result, required_permission):
                        return asdict(error_handler.handle_error(
                            "authorization_error", 
                            f"Insufficient permissions. Required: {required_permission}"
                        ))
                    
                    user_id = auth_result["user_id"]
                else:
                    user_id = "anonymous"
                
                # Rate limiting
                ip = params.get("ip", "unknown")
                operation_type = params.get("operation_type", "default")
                
                if not rate_limiter.check_rate_limit(user_id, ip, operation_type):
                    return asdict(error_handler.handle_error(
                        "rate_limit_error", 
                        "Rate limit exceeded"
                    ))
                
                # Input validation
                if validate_input and schema_name:
                    validation_result = validator.validate_schema(params, schema_name)
                    if not validation_result["valid"]:
                        return asdict(error_handler.handle_error(
                            "validation_error", 
                            "Input validation failed",
                            {"validation_errors": validation_result["errors"]}
                        ))
                
                # Input sanitization
                sanitized_params = validator.sanitize_input(params)
                
                # Call the original function
                result = await func(self, sanitized_params, *args, **kwargs)
                return result
                
            except Exception as e:
                return asdict(error_handler.handle_exception(e, func.__name__))
        
        return wrapper
    return decorator 