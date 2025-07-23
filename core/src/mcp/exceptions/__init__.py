"""
Base exception hierarchy for the MCP system.

This module defines the base exceptions used throughout the MCP system.
All custom exceptions should inherit from MCPBaseException.

λexception_hierarchy(base_definition)
"""

class MCPBaseException(Exception):
    """Base exception for all MCP-related exceptions."""
    
    def __init__(self, message="An error occurred in the MCP system", *args, **kwargs):
        self.message = message
        super().__init__(message, *args, **kwargs)


class MCPConfigurationError(MCPBaseException):
    """Exception raised for errors in the configuration."""
    
    def __init__(self, message="Configuration error in the MCP system", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class MCPInterfaceError(MCPBaseException):
    """Exception raised for errors related to interfaces."""
    
    def __init__(self, message="Interface error in the MCP system", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class MCPImplementationError(MCPBaseException):
    """Exception raised when an implementation is incorrect or incomplete."""
    
    def __init__(self, message="Implementation error in the MCP system", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class MCPMemoryError(MCPBaseException):
    """Exception raised for errors in memory management."""
    
    def __init__(self, message="Memory management error in the MCP system", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class MCPWorkflowError(MCPBaseException):
    """Exception raised for errors in workflow processing."""
    
    def __init__(self, message="Workflow error in the MCP system", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class MCPContextError(MCPBaseException):
    """Exception raised for errors in context management."""
    
    def __init__(self, message="Context error in the MCP system", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class MCPDatabaseError(MCPBaseException):
    """Exception raised for errors in database operations."""
    
    def __init__(self, message="Database error in the MCP system", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class MCPLobeError(MCPBaseException):
    """Exception raised for errors in lobe operations."""
    
    def __init__(self, message="Lobe error in the MCP system", *args, **kwargs):
        super().__init__(message, *args, **kwargs)