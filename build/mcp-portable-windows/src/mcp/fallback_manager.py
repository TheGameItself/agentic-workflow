#!/usr/bin/env python3
"""
Fallback Management System for MCP

This module implements a comprehensive fallback management system with error classification
and recovery strategies to ensure the system never breaks and always provides meaningful responses.
"""

import logging
import traceback
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import json
import sqlite3
import asyncio
import inspect


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FallbackStrategy(Enum):
    """Available fallback strategies."""
    RETRY = "retry"
    ALTERNATIVE_METHOD = "alternative_method"
    DEFAULT_VALUE = "default_value"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CACHE_FALLBACK = "cache_fallback"
    SIMPLIFIED_OPERATION = "simplified_operation"
    ERROR_SUPPRESSION = "error_suppression"


@dataclass
class ErrorContext:
    """Context information about an error."""
    error: Exception
    function_name: str
    class_name: Optional[str]
    module_name: str
    args: tuple
    kwargs: dict
    timestamp: datetime
    stack_trace: str
    attempt_count: int = 1
    previous_errors: List[str] = field(default_factory=list)


@dataclass
class FallbackRule:
    """Rule for handling specific types of errors."""
    error_types: List[Type[Exception]]
    strategy: FallbackStrategy
    handler: Callable
    max_retries: int = 3
    retry_delay: float = 1.0
    priority: int = 100
    conditions: Optional[Dict[str, Any]] = None
    description: str = ""


@dataclass
class FallbackResult:
    """Result of a fallback operation."""
    success: bool
    result: Any
    strategy_used: FallbackStrategy
    error_context: ErrorContext
    execution_time: float
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ErrorAnalyzer:
    """Analyzes errors to determine appropriate fallback strategies."""
    
    def __init__(self):
        self.error_patterns = self._initialize_error_patterns()
        self.severity_rules = self._initialize_severity_rules()
    
    def _initialize_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize error pattern recognition."""
        return {
            'database_errors': {
                'patterns': ['sqlite3.', 'database', 'connection', 'cursor'],
                'severity': ErrorSeverity.HIGH,
                'suggested_strategies': [FallbackStrategy.RETRY, FallbackStrategy.CACHE_FALLBACK]
            },
            'network_errors': {
                'patterns': ['connection', 'timeout', 'network', 'http', 'requests'],
                'severity': ErrorSeverity.MEDIUM,
                'suggested_strategies': [FallbackStrategy.RETRY, FallbackStrategy.CACHE_FALLBACK]
            },
            'memory_errors': {
                'patterns': ['memory', 'allocation', 'out of memory'],
                'severity': ErrorSeverity.CRITICAL,
                'suggested_strategies': [FallbackStrategy.GRACEFUL_DEGRADATION, FallbackStrategy.SIMPLIFIED_OPERATION]
            },
            'file_errors': {
                'patterns': ['file', 'directory', 'path', 'permission'],
                'severity': ErrorSeverity.MEDIUM,
                'suggested_strategies': [FallbackStrategy.ALTERNATIVE_METHOD, FallbackStrategy.DEFAULT_VALUE]
            },
            'import_errors': {
                'patterns': ['import', 'module', 'package'],
                'severity': ErrorSeverity.HIGH,
                'suggested_strategies': [FallbackStrategy.ALTERNATIVE_METHOD, FallbackStrategy.GRACEFUL_DEGRADATION]
            },
            'validation_errors': {
                'patterns': ['validation', 'invalid', 'format', 'type'],
                'severity': ErrorSeverity.LOW,
                'suggested_strategies': [FallbackStrategy.DEFAULT_VALUE, FallbackStrategy.ERROR_SUPPRESSION]
            }
        }
    
    def _initialize_severity_rules(self) -> Dict[Type[Exception], ErrorSeverity]:
        """Initialize severity rules for exception types."""
        return {
            SystemExit: ErrorSeverity.CRITICAL,
            KeyboardInterrupt: ErrorSeverity.CRITICAL,
            MemoryError: ErrorSeverity.CRITICAL,
            RecursionError: ErrorSeverity.CRITICAL,
            ImportError: ErrorSeverity.HIGH,
            ModuleNotFoundError: ErrorSeverity.HIGH,
            ConnectionError: ErrorSeverity.MEDIUM,
            TimeoutError: ErrorSeverity.MEDIUM,
            FileNotFoundError: ErrorSeverity.MEDIUM,
            PermissionError: ErrorSeverity.MEDIUM,
            ValueError: ErrorSeverity.LOW,
            TypeError: ErrorSeverity.LOW,
            AttributeError: ErrorSeverity.LOW,
            KeyError: ErrorSeverity.LOW,
            IndexError: ErrorSeverity.LOW,
            NotImplementedError: ErrorSeverity.HIGH,
        }
    
    def analyze_error(self, error_context: ErrorContext) -> Dict[str, Any]:
        """
        Analyze an error and provide recommendations.
        
        Args:
            error_context: Context information about the error
            
        Returns:
            Analysis results with severity and suggested strategies
        """
        error = error_context.error
        error_str = str(error).lower()
        error_type = type(error)
        
        # Determine severity
        severity = self.severity_rules.get(error_type, ErrorSeverity.MEDIUM)
        
        # Check error patterns
        matched_patterns = []
        suggested_strategies = []
        
        for pattern_name, pattern_info in self.error_patterns.items():
            if any(pattern in error_str for pattern in pattern_info['patterns']):
                matched_patterns.append(pattern_name)
                suggested_strategies.extend(pattern_info['suggested_strategies'])
                # Update severity if pattern suggests higher severity
                if pattern_info['severity'].value == 'critical' and severity != ErrorSeverity.CRITICAL:
                    severity = pattern_info['severity']
        
        # Remove duplicates while preserving order
        suggested_strategies = list(dict.fromkeys(suggested_strategies))
        
        return {
            'severity': severity,
            'matched_patterns': matched_patterns,
            'suggested_strategies': suggested_strategies,
            'error_type': error_type.__name__,
            'error_message': str(error),
            'is_recoverable': self._is_recoverable(error, error_context),
            'retry_recommended': self._should_retry(error, error_context)
        }
    
    def _is_recoverable(self, error: Exception, context: ErrorContext) -> bool:
        """Determine if an error is recoverable."""
        # Critical errors are generally not recoverable
        if type(error) in [SystemExit, KeyboardInterrupt, MemoryError]:
            return False
        
        # Too many attempts suggest non-recoverable issue
        if context.attempt_count > 5:
            return False
        
        # Most other errors are potentially recoverable
        return True
    
    def _should_retry(self, error: Exception, context: ErrorContext) -> bool:
        """Determine if an error should be retried."""
        # Don't retry certain error types
        no_retry_types = [ValueError, TypeError, AttributeError, NotImplementedError]
        if type(error) in no_retry_types:
            return False
        
        # Don't retry if we've already tried too many times
        if context.attempt_count >= 3:
            return False
        
        # Retry for transient errors
        retry_types = [ConnectionError, TimeoutError, sqlite3.OperationalError]
        return type(error) in retry_types


class FallbackRegistry:
    """Registry of fallback rules and handlers."""
    
    def __init__(self):
        self.rules: List[FallbackRule] = []
        self.default_handlers = self._initialize_default_handlers()
        self._register_default_rules()
    
    def _initialize_default_handlers(self) -> Dict[FallbackStrategy, Callable]:
        """Initialize default fallback handlers."""
        return {
            FallbackStrategy.RETRY: self._retry_handler,
            FallbackStrategy.ALTERNATIVE_METHOD: self._alternative_method_handler,
            FallbackStrategy.DEFAULT_VALUE: self._default_value_handler,
            FallbackStrategy.GRACEFUL_DEGRADATION: self._graceful_degradation_handler,
            FallbackStrategy.CACHE_FALLBACK: self._cache_fallback_handler,
            FallbackStrategy.SIMPLIFIED_OPERATION: self._simplified_operation_handler,
            FallbackStrategy.ERROR_SUPPRESSION: self._error_suppression_handler
        }
    
    def _register_default_rules(self):
        """Register default fallback rules."""
        # Database errors
        self.register_rule(FallbackRule(
            error_types=[sqlite3.Error, sqlite3.OperationalError],
            strategy=FallbackStrategy.RETRY,
            handler=self.default_handlers[FallbackStrategy.RETRY],
            max_retries=3,
            retry_delay=1.0,
            priority=90,
            description="Retry database operations with exponential backoff"
        ))
        
        # Import errors
        self.register_rule(FallbackRule(
            error_types=[ImportError, ModuleNotFoundError],
            strategy=FallbackStrategy.ALTERNATIVE_METHOD,
            handler=self.default_handlers[FallbackStrategy.ALTERNATIVE_METHOD],
            priority=80,
            description="Use alternative implementation when imports fail"
        ))
        
        # File errors
        self.register_rule(FallbackRule(
            error_types=[FileNotFoundError, PermissionError],
            strategy=FallbackStrategy.DEFAULT_VALUE,
            handler=self.default_handlers[FallbackStrategy.DEFAULT_VALUE],
            priority=70,
            description="Provide default values when file operations fail"
        ))
        
        # Memory errors
        self.register_rule(FallbackRule(
            error_types=[MemoryError],
            strategy=FallbackStrategy.GRACEFUL_DEGRADATION,
            handler=self.default_handlers[FallbackStrategy.GRACEFUL_DEGRADATION],
            priority=95,
            description="Gracefully degrade functionality when memory is low"
        ))
        
        # Network errors
        self.register_rule(FallbackRule(
            error_types=[ConnectionError, TimeoutError],
            strategy=FallbackStrategy.CACHE_FALLBACK,
            handler=self.default_handlers[FallbackStrategy.CACHE_FALLBACK],
            max_retries=2,
            retry_delay=2.0,
            priority=85,
            description="Use cached data when network operations fail"
        ))
        
        # NotImplementedError
        self.register_rule(FallbackRule(
            error_types=[NotImplementedError],
            strategy=FallbackStrategy.DEFAULT_VALUE,
            handler=self.default_handlers[FallbackStrategy.DEFAULT_VALUE],
            priority=100,
            description="Provide meaningful defaults for unimplemented methods"
        ))
    
    def register_rule(self, rule: FallbackRule):
        """Register a new fallback rule."""
        self.rules.append(rule)
        # Sort by priority (higher priority first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def get_applicable_rules(self, error: Exception, context: ErrorContext) -> List[FallbackRule]:
        """Get applicable fallback rules for an error."""
        applicable_rules = []
        
        for rule in self.rules:
            if any(isinstance(error, error_type) for error_type in rule.error_types):
                # Check additional conditions if specified
                if rule.conditions:
                    if not self._check_conditions(rule.conditions, context):
                        continue
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def _check_conditions(self, conditions: Dict[str, Any], context: ErrorContext) -> bool:
        """Check if rule conditions are met."""
        for key, value in conditions.items():
            if key == 'max_attempts' and context.attempt_count > value:
                return False
            elif key == 'function_name' and context.function_name != value:
                return False
            elif key == 'module_name' and context.module_name != value:
                return False
        return True
    
    # Default handler implementations
    async def _retry_handler(self, context: ErrorContext, rule: FallbackRule) -> FallbackResult:
        """Retry the original operation with exponential backoff."""
        if context.attempt_count >= rule.max_retries:
            return FallbackResult(
                success=False,
                result=None,
                strategy_used=FallbackStrategy.RETRY,
                error_context=context,
                execution_time=0.0,
                message=f"Max retries ({rule.max_retries}) exceeded"
            )
        
        # Wait with exponential backoff
        delay = rule.retry_delay * (2 ** (context.attempt_count - 1))
        await asyncio.sleep(delay)
        
        return FallbackResult(
            success=True,
            result="retry_requested",
            strategy_used=FallbackStrategy.RETRY,
            error_context=context,
            execution_time=delay,
            message=f"Retrying after {delay:.1f}s delay (attempt {context.attempt_count + 1})"
        )
    
    async def _alternative_method_handler(self, context: ErrorContext, rule: FallbackRule) -> FallbackResult:
        """Use alternative method implementation."""
        # Try to find alternative implementations
        alternatives = self._find_alternative_methods(context)
        
        if alternatives:
            return FallbackResult(
                success=True,
                result=alternatives[0],  # Use first alternative
                strategy_used=FallbackStrategy.ALTERNATIVE_METHOD,
                error_context=context,
                execution_time=0.0,
                message=f"Using alternative method: {alternatives[0].__name__ if hasattr(alternatives[0], '__name__') else 'unknown'}"
            )
        
        return FallbackResult(
            success=False,
            result=None,
            strategy_used=FallbackStrategy.ALTERNATIVE_METHOD,
            error_context=context,
            execution_time=0.0,
            message="No alternative methods available"
        )
    
    async def _default_value_handler(self, context: ErrorContext, rule: FallbackRule) -> FallbackResult:
        """Provide appropriate default value."""
        default_value = self._determine_default_value(context)
        
        return FallbackResult(
            success=True,
            result=default_value,
            strategy_used=FallbackStrategy.DEFAULT_VALUE,
            error_context=context,
            execution_time=0.0,
            message=f"Using default value: {type(default_value).__name__}"
        )
    
    async def _graceful_degradation_handler(self, context: ErrorContext, rule: FallbackRule) -> FallbackResult:
        """Gracefully degrade functionality."""
        # Implement reduced functionality
        degraded_result = self._create_degraded_response(context)
        
        return FallbackResult(
            success=True,
            result=degraded_result,
            strategy_used=FallbackStrategy.GRACEFUL_DEGRADATION,
            error_context=context,
            execution_time=0.0,
            message="Operating in degraded mode with reduced functionality"
        )
    
    async def _cache_fallback_handler(self, context: ErrorContext, rule: FallbackRule) -> FallbackResult:
        """Use cached data as fallback."""
        cached_result = self._get_cached_result(context)
        
        if cached_result is not None:
            return FallbackResult(
                success=True,
                result=cached_result,
                strategy_used=FallbackStrategy.CACHE_FALLBACK,
                error_context=context,
                execution_time=0.0,
                message="Using cached result"
            )
        
        return FallbackResult(
            success=False,
            result=None,
            strategy_used=FallbackStrategy.CACHE_FALLBACK,
            error_context=context,
            execution_time=0.0,
            message="No cached result available"
        )
    
    async def _simplified_operation_handler(self, context: ErrorContext, rule: FallbackRule) -> FallbackResult:
        """Perform simplified version of the operation."""
        simplified_result = self._create_simplified_result(context)
        
        return FallbackResult(
            success=True,
            result=simplified_result,
            strategy_used=FallbackStrategy.SIMPLIFIED_OPERATION,
            error_context=context,
            execution_time=0.0,
            message="Using simplified operation"
        )
    
    async def _error_suppression_handler(self, context: ErrorContext, rule: FallbackRule) -> FallbackResult:
        """Suppress error and continue with minimal impact."""
        return FallbackResult(
            success=True,
            result=None,
            strategy_used=FallbackStrategy.ERROR_SUPPRESSION,
            error_context=context,
            execution_time=0.0,
            message="Error suppressed, continuing with null result"
        )
    
    def _find_alternative_methods(self, context: ErrorContext) -> List[Callable]:
        """Find alternative method implementations."""
        alternatives = []
        
        # Look for common alternative patterns
        function_name = context.function_name
        
        # Try _fallback_ prefixed methods
        fallback_name = f"_fallback_{function_name}"
        if hasattr(context, 'instance'):
            if hasattr(context.instance, fallback_name):
                alternatives.append(getattr(context.instance, fallback_name))
        
        # Try _simple_ prefixed methods
        simple_name = f"_simple_{function_name}"
        if hasattr(context, 'instance'):
            if hasattr(context.instance, simple_name):
                alternatives.append(getattr(context.instance, simple_name))
        
        return alternatives
    
    def _determine_default_value(self, context: ErrorContext) -> Any:
        """Determine appropriate default value based on context."""
        function_name = context.function_name.lower()
        
        # Return type hints from function signature
        if hasattr(context, 'function'):
            try:
                sig = inspect.signature(context.function)
                if sig.return_annotation != inspect.Signature.empty:
                    if sig.return_annotation == dict:
                        return {}
                    elif sig.return_annotation == list:
                        return []
                    elif sig.return_annotation == str:
                        return ""
                    elif sig.return_annotation == bool:
                        return False
                    elif sig.return_annotation == int:
                        return 0
                    elif sig.return_annotation == float:
                        return 0.0
            except Exception:
                pass
        
        # Heuristic-based defaults
        if 'get_' in function_name or 'find_' in function_name:
            return {}
        elif 'list_' in function_name or 'search_' in function_name:
            return []
        elif 'is_' in function_name or 'has_' in function_name or 'can_' in function_name:
            return False
        elif 'count_' in function_name:
            return 0
        elif 'calculate_' in function_name or 'compute_' in function_name:
            return 0.0
        else:
            return None
    
    def _create_degraded_response(self, context: ErrorContext) -> Dict[str, Any]:
        """Create a degraded response with limited functionality."""
        return {
            "status": "degraded",
            "message": f"Operating in degraded mode due to error in {context.function_name}",
            "error": str(context.error),
            "timestamp": context.timestamp.isoformat(),
            "limited_functionality": True
        }
    
    def _get_cached_result(self, context: ErrorContext) -> Any:
        """Get cached result if available."""
        # This would integrate with a caching system
        # For now, return None to indicate no cache available
        return None
    
    def _create_simplified_result(self, context: ErrorContext) -> Any:
        """Create a simplified result."""
        return {
            "simplified": True,
            "function": context.function_name,
            "message": "Simplified operation completed",
            "timestamp": datetime.now().isoformat()
        }


class FallbackManager:
    """
    Comprehensive fallback management system.
    
    This manager coordinates error handling, recovery strategies, and ensures
    the system never breaks by providing appropriate fallbacks for all error conditions.
    """
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize the fallback manager."""
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent.parent
        self.logger = self._setup_logging()
        self.error_analyzer = ErrorAnalyzer()
        self.fallback_registry = FallbackRegistry()
        self.error_history: deque = deque(maxlen=1000)  # Keep last 1000 errors
        self.recovery_stats = defaultdict(int)
        self.cache = {}
        self.circuit_breakers = {}  # For preventing cascading failures
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the fallback manager."""
        logger = logging.getLogger("fallback_manager")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> Any:
        """
        Handle an error with appropriate fallback strategy.
        
        Args:
            error: The exception that occurred
            context: Context information about where the error occurred
            
        Returns:
            Result from fallback strategy or raises if no fallback available
        """
        # Create error context
        error_context = self._create_error_context(error, context)
        
        # Log the error
        self.logger.warning(f"Handling error in {error_context.function_name}: {error}")
        
        # Add to error history
        self.error_history.append(error_context)
        
        # Check circuit breaker
        if self._is_circuit_breaker_open(error_context):
            return await self._handle_circuit_breaker_open(error_context)
        
        # Analyze the error
        analysis = self.error_analyzer.analyze_error(error_context)
        
        # Get applicable fallback rules
        applicable_rules = self.fallback_registry.get_applicable_rules(error, error_context)
        
        if not applicable_rules:
            self.logger.error(f"No fallback rules found for error: {error}")
            return await self._handle_no_fallback_available(error_context)
        
        # Try fallback strategies in order of priority
        for rule in applicable_rules:
            try:
                result = await self._execute_fallback(rule, error_context)
                if result.success:
                    self.recovery_stats[rule.strategy.value] += 1
                    self.logger.info(f"Successfully recovered using {rule.strategy.value}: {result.message}")
                    
                    # Cache successful result if appropriate
                    if rule.strategy in [FallbackStrategy.CACHE_FALLBACK, FallbackStrategy.DEFAULT_VALUE]:
                        self._cache_result(error_context, result.result)
                    
                    return result.result
                else:
                    self.logger.warning(f"Fallback strategy {rule.strategy.value} failed: {result.message}")
                    
            except Exception as fallback_error:
                self.logger.error(f"Fallback strategy {rule.strategy.value} raised error: {fallback_error}")
                continue
        
        # If all fallback strategies failed
        self.logger.error(f"All fallback strategies failed for error: {error}")
        return await self._handle_all_fallbacks_failed(error_context)
    
    def _create_error_context(self, error: Exception, context: Dict[str, Any]) -> ErrorContext:
        """Create error context from exception and context information."""
        # Extract information from the current stack frame
        frame = sys._getframe(2)  # Go back 2 frames to get the original caller
        
        function_name = context.get('function_name', frame.f_code.co_name)
        class_name = context.get('class_name')
        module_name = context.get('module_name', frame.f_globals.get('__name__', 'unknown'))
        
        # Get previous errors for this context
        previous_errors = [
            str(ec.error) for ec in self.error_history 
            if ec.function_name == function_name and ec.module_name == module_name
        ][-5:]  # Last 5 errors
        
        return ErrorContext(
            error=error,
            function_name=function_name,
            class_name=class_name,
            module_name=module_name,
            args=context.get('args', ()),
            kwargs=context.get('kwargs', {}),
            timestamp=datetime.now(),
            stack_trace=traceback.format_exc(),
            attempt_count=context.get('attempt_count', 1),
            previous_errors=previous_errors
        )
    
    def _is_circuit_breaker_open(self, error_context: ErrorContext) -> bool:
        """Check if circuit breaker is open for this function."""
        key = f"{error_context.module_name}.{error_context.function_name}"
        
        if key not in self.circuit_breakers:
            self.circuit_breakers[key] = {
                'failure_count': 0,
                'last_failure': None,
                'state': 'closed'  # closed, open, half-open
            }
        
        breaker = self.circuit_breakers[key]
        
        # Check if we should open the circuit breaker
        if breaker['failure_count'] >= 5:  # 5 failures threshold
            if breaker['state'] == 'closed':
                breaker['state'] = 'open'
                breaker['last_failure'] = datetime.now()
                self.logger.warning(f"Circuit breaker opened for {key}")
            
            # Check if we should try half-open
            elif breaker['state'] == 'open':
                if datetime.now() - breaker['last_failure'] > timedelta(minutes=5):
                    breaker['state'] = 'half-open'
                    self.logger.info(f"Circuit breaker half-open for {key}")
        
        return breaker['state'] == 'open'
    
    async def _handle_circuit_breaker_open(self, error_context: ErrorContext) -> Any:
        """Handle case when circuit breaker is open."""
        self.logger.warning(f"Circuit breaker open for {error_context.function_name}, using cached result")
        
        # Try to get cached result
        cached_result = self._get_cached_result(error_context)
        if cached_result is not None:
            return cached_result
        
        # Return appropriate default
        return self.fallback_registry._determine_default_value(error_context)
    
    async def _execute_fallback(self, rule: FallbackRule, error_context: ErrorContext) -> FallbackResult:
        """Execute a specific fallback rule."""
        start_time = time.time()
        
        try:
            # Special handling for retry strategy
            if rule.strategy == FallbackStrategy.RETRY:
                if error_context.attempt_count >= rule.max_retries:
                    return FallbackResult(
                        success=False,
                        result=None,
                        strategy_used=rule.strategy,
                        error_context=error_context,
                        execution_time=0.0,
                        message=f"Max retries ({rule.max_retries}) exceeded"
                    )
                
                # For retry, we need to re-execute the original function
                return await self._handle_retry(rule, error_context)
            
            # Execute the fallback handler
            result = await rule.handler(error_context, rule)
            result.execution_time = time.time() - start_time
            return result
            
        except Exception as e:
            return FallbackResult(
                success=False,
                result=None,
                strategy_used=rule.strategy,
                error_context=error_context,
                execution_time=time.time() - start_time,
                message=f"Fallback handler failed: {e}"
            )
    
    async def _handle_retry(self, rule: FallbackRule, error_context: ErrorContext) -> FallbackResult:
        """Handle retry strategy."""
        # Calculate delay with exponential backoff
        delay = rule.retry_delay * (2 ** (error_context.attempt_count - 1))
        
        self.logger.info(f"Retrying {error_context.function_name} after {delay:.1f}s (attempt {error_context.attempt_count + 1})")
        
        await asyncio.sleep(delay)
        
        return FallbackResult(
            success=True,
            result="retry_requested",
            strategy_used=FallbackStrategy.RETRY,
            error_context=error_context,
            execution_time=delay,
            message=f"Retry scheduled after {delay:.1f}s delay"
        )
    
    async def _handle_no_fallback_available(self, error_context: ErrorContext) -> Any:
        """Handle case when no fallback rules are available."""
        self.logger.error(f"No fallback available for {error_context.function_name}")
        
        # Try to provide a sensible default based on function name
        default_value = self.fallback_registry._determine_default_value(error_context)
        
        if default_value is not None:
            self.logger.info(f"Using heuristic default value for {error_context.function_name}")
            return default_value
        
        # Last resort: return error information
        return {
            "error": True,
            "message": f"Function {error_context.function_name} failed with no fallback available",
            "error_type": type(error_context.error).__name__,
            "timestamp": error_context.timestamp.isoformat()
        }
    
    async def _handle_all_fallbacks_failed(self, error_context: ErrorContext) -> Any:
        """Handle case when all fallback strategies have failed."""
        self.logger.critical(f"All fallbacks failed for {error_context.function_name}")
        
        # Update circuit breaker
        key = f"{error_context.module_name}.{error_context.function_name}"
        if key in self.circuit_breakers:
            self.circuit_breakers[key]['failure_count'] += 1
        
        # Return error response that won't break the system
        return {
            "error": True,
            "message": f"All fallback strategies failed for {error_context.function_name}",
            "error_type": type(error_context.error).__name__,
            "fallback_attempts": len(self.fallback_registry.get_applicable_rules(error_context.error, error_context)),
            "timestamp": error_context.timestamp.isoformat(),
            "degraded_mode": True
        }
    
    def _cache_result(self, error_context: ErrorContext, result: Any):
        """Cache a successful result for future fallback use."""
        key = f"{error_context.module_name}.{error_context.function_name}"
        self.cache[key] = {
            'result': result,
            'timestamp': datetime.now(),
            'args_hash': hash(str(error_context.args) + str(error_context.kwargs))
        }
        
        # Limit cache size
        if len(self.cache) > 100:
            # Remove oldest entries
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
    
    def _get_cached_result(self, error_context: ErrorContext) -> Any:
        """Get cached result if available."""
        key = f"{error_context.module_name}.{error_context.function_name}"
        
        if key in self.cache:
            cached = self.cache[key]
            # Check if cache is still valid (within 1 hour)
            if datetime.now() - cached['timestamp'] < timedelta(hours=1):
                return cached['result']
            else:
                # Remove expired cache
                del self.cache[key]
        
        return None
    
    def register_custom_fallback(self, error_types: List[Type[Exception]], 
                                strategy: FallbackStrategy, 
                                handler: Callable,
                                **kwargs):
        """Register a custom fallback rule."""
        rule = FallbackRule(
            error_types=error_types,
            strategy=strategy,
            handler=handler,
            **kwargs
        )
        self.fallback_registry.register_rule(rule)
        self.logger.info(f"Registered custom fallback rule for {[t.__name__ for t in error_types]}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error and recovery statistics."""
        if not self.error_history:
            return {"message": "No error history available"}
        
        # Count errors by type
        error_types = defaultdict(int)
        error_functions = defaultdict(int)
        error_modules = defaultdict(int)
        
        for error_context in self.error_history:
            error_types[type(error_context.error).__name__] += 1
            error_functions[error_context.function_name] += 1
            error_modules[error_context.module_name] += 1
        
        return {
            "total_errors": len(self.error_history),
            "error_types": dict(error_types),
            "error_functions": dict(error_functions),
            "error_modules": dict(error_modules),
            "recovery_stats": dict(self.recovery_stats),
            "circuit_breakers": {
                k: v for k, v in self.circuit_breakers.items() 
                if v['failure_count'] > 0
            },
            "cache_size": len(self.cache)
        }
    
    def reset_circuit_breaker(self, function_key: str):
        """Reset a circuit breaker manually."""
        if function_key in self.circuit_breakers:
            self.circuit_breakers[function_key] = {
                'failure_count': 0,
                'last_failure': None,
                'state': 'closed'
            }
            self.logger.info(f"Reset circuit breaker for {function_key}")
    
    def clear_error_history(self):
        """Clear error history."""
        self.error_history.clear()
        self.logger.info("Cleared error history")
    
    def export_error_report(self, format: str = "json") -> Union[Dict[str, Any], str]:
        """Export comprehensive error report."""
        stats = self.get_error_statistics()
        
        report_data = {
            "report_timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "recent_errors": [
                {
                    "timestamp": ec.timestamp.isoformat(),
                    "function": ec.function_name,
                    "module": ec.module_name,
                    "error_type": type(ec.error).__name__,
                    "error_message": str(ec.error),
                    "attempt_count": ec.attempt_count
                }
                for ec in list(self.error_history)[-20:]  # Last 20 errors
            ],
            "fallback_rules": len(self.fallback_registry.rules),
            "system_health": self._assess_system_health()
        }
        
        if format == "json":
            return report_data
        elif format == "text":
            return self._format_text_report(report_data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health based on error patterns."""
        if not self.error_history:
            return {"status": "healthy", "score": 100}
        
        recent_errors = [ec for ec in self.error_history if datetime.now() - ec.timestamp < timedelta(hours=1)]
        
        if len(recent_errors) == 0:
            health_score = 100
            status = "healthy"
        elif len(recent_errors) < 5:
            health_score = 90
            status = "good"
        elif len(recent_errors) < 20:
            health_score = 70
            status = "warning"
        else:
            health_score = 30
            status = "critical"
        
        # Adjust score based on circuit breakers
        open_breakers = sum(1 for cb in self.circuit_breakers.values() if cb['state'] == 'open')
        health_score -= open_breakers * 10
        
        return {
            "status": status,
            "score": max(0, health_score),
            "recent_errors": len(recent_errors),
            "open_circuit_breakers": open_breakers,
            "recovery_rate": sum(self.recovery_stats.values()) / max(len(self.error_history), 1) * 100
        }
    
    def _format_text_report(self, report_data: Dict[str, Any]) -> str:
        """Format error report as text."""
        stats = report_data["statistics"]
        health = report_data["system_health"]
        
        report = f"""
FALLBACK MANAGER ERROR REPORT
============================

Report generated: {report_data['report_timestamp']}

SYSTEM HEALTH: {health['status'].upper()} (Score: {health['score']}/100)
- Recent errors (1h): {health['recent_errors']}
- Open circuit breakers: {health['open_circuit_breakers']}
- Recovery rate: {health['recovery_rate']:.1f}%

ERROR STATISTICS:
- Total errors: {stats['total_errors']}
- Cache size: {stats['cache_size']}

TOP ERROR TYPES:
"""
        
        for error_type, count in sorted(stats['error_types'].items(), key=lambda x: x[1], reverse=True)[:5]:
            report += f"- {error_type}: {count}\n"
        
        report += "\nTOP ERROR FUNCTIONS:\n"
        for func, count in sorted(stats['error_functions'].items(), key=lambda x: x[1], reverse=True)[:5]:
            report += f"- {func}: {count}\n"
        
        report += "\nRECOVERY STATISTICS:\n"
        for strategy, count in stats['recovery_stats'].items():
            report += f"- {strategy}: {count}\n"
        
        return report


# Decorator for automatic fallback handling
def with_fallback(fallback_manager: Optional[FallbackManager] = None, 
                  context: Optional[Dict[str, Any]] = None):
    """Decorator to automatically handle errors with fallback strategies."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            nonlocal fallback_manager
            if fallback_manager is None:
                fallback_manager = FallbackManager()
            
            attempt_count = 1
            max_attempts = 3
            
            while attempt_count <= max_attempts:
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                        
                except Exception as e:
                    error_context = context or {}
                    error_context.update({
                        'function_name': func.__name__,
                        'module_name': func.__module__,
                        'args': args,
                        'kwargs': kwargs,
                        'attempt_count': attempt_count
                    })
                    
                    result = await fallback_manager.handle_error(e, error_context)
                    
                    # If result indicates retry, increment attempt count and continue
                    if result == "retry_requested":
                        attempt_count += 1
                        continue
                    else:
                        return result
            
            # If we've exhausted all attempts
            return await fallback_manager._handle_all_fallbacks_failed(
                fallback_manager._create_error_context(e, error_context)
            )
        
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, run the async wrapper in an event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Convenience functions
def create_fallback_manager(project_root: Optional[str] = None) -> FallbackManager:
    """Create a fallback manager instance."""
    return FallbackManager(project_root)


async def handle_error_with_fallback(error: Exception, context: Dict[str, Any], 
                                   fallback_manager: Optional[FallbackManager] = None) -> Any:
    """Convenience function to handle an error with fallback."""
    if fallback_manager is None:
        fallback_manager = FallbackManager()
    
    return await fallback_manager.handle_error(error, context)