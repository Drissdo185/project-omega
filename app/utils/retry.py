# app/utils/retry.py
"""
Retry logic for API calls with exponential backoff
"""
import asyncio
import functools
from typing import Callable, Any, Optional, Type, Tuple
from loguru import logger


class RetryError(Exception):
    """Exception raised when all retry attempts fail"""
    pass


def async_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator for retrying async functions with exponential backoff
    
    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exception types to catch and retry
        
    Example:
        @async_retry(max_attempts=3, initial_delay=1.0)
        async def fetch_data():
            # Your async code here
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            attempt = 1
            delay = initial_delay
            
            while attempt <= max_attempts:
                try:
                    result = await func(*args, **kwargs)
                    
                    if attempt > 1:
                        logger.info(
                            f"✅ {func.__name__} succeeded on attempt {attempt}"
                        )
                    
                    return result
                    
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(
                            f"❌ {func.__name__} failed after {max_attempts} attempts. "
                            f"Last error: {type(e).__name__}: {str(e)}"
                        )
                        raise RetryError(
                            f"Function {func.__name__} failed after {max_attempts} attempts"
                        ) from e
                    
                    logger.warning(
                        f"⚠️ {func.__name__} attempt {attempt}/{max_attempts} failed: "
                        f"{type(e).__name__}: {str(e)}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    
                    await asyncio.sleep(delay)
                    
                    # Exponential backoff
                    delay = min(delay * exponential_base, max_delay)
                    attempt += 1
        
        return wrapper
    return decorator


def sync_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator for retrying synchronous functions with exponential backoff
    
    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exception types to catch and retry
        
    Example:
        @sync_retry(max_attempts=3, initial_delay=1.0)
        def fetch_data():
            # Your sync code here
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import time
            
            attempt = 1
            delay = initial_delay
            
            while attempt <= max_attempts:
                try:
                    result = func(*args, **kwargs)
                    
                    if attempt > 1:
                        logger.info(
                            f"✅ {func.__name__} succeeded on attempt {attempt}"
                        )
                    
                    return result
                    
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(
                            f"❌ {func.__name__} failed after {max_attempts} attempts. "
                            f"Last error: {type(e).__name__}: {str(e)}"
                        )
                        raise RetryError(
                            f"Function {func.__name__} failed after {max_attempts} attempts"
                        ) from e
                    
                    logger.warning(
                        f"⚠️ {func.__name__} attempt {attempt}/{max_attempts} failed: "
                        f"{type(e).__name__}: {str(e)}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    
                    time.sleep(delay)
                    
                    # Exponential backoff
                    delay = min(delay * exponential_base, max_delay)
                    attempt += 1
        
        return wrapper
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failures detected, requests fail immediately
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0
    ):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            success_threshold: Number of successes needed to close circuit
            timeout: Time in seconds before attempting to close circuit
        """
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        
        self.failures = 0
        self.successes = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def is_open(self) -> bool:
        """Check if circuit is open"""
        if self.state == "OPEN":
            # Check if timeout has passed
            import time
            current_time = time.time()
            
            if self.last_failure_time and \
               (current_time - self.last_failure_time) >= self.timeout:
                logger.info("⚡ Circuit breaker: Transitioning to HALF_OPEN state")
                self.state = "HALF_OPEN"
                self.successes = 0
                return False
            
            return True
        
        return False
    
    def record_success(self) -> None:
        """Record a successful call"""
        if self.state == "HALF_OPEN":
            self.successes += 1
            
            if self.successes >= self.success_threshold:
                logger.info("✅ Circuit breaker: Transitioning to CLOSED state")
                self.state = "CLOSED"
                self.failures = 0
                self.successes = 0
        
        elif self.state == "CLOSED":
            # Reset failure count on success
            self.failures = 0
    
    def record_failure(self) -> None:
        """Record a failed call"""
        import time
        self.last_failure_time = time.time()
        
        if self.state == "HALF_OPEN":
            logger.warning("⚠️ Circuit breaker: Failure in HALF_OPEN, returning to OPEN")
            self.state = "OPEN"
            self.failures = 0
            self.successes = 0
        
        elif self.state == "CLOSED":
            self.failures += 1
            
            if self.failures >= self.failure_threshold:
                logger.error(
                    f"❌ Circuit breaker: Opening circuit after "
                    f"{self.failures} failures"
                )
                self.state = "OPEN"
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Async function to call
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.is_open():
            raise Exception(
                f"Circuit breaker is OPEN. Service unavailable. "
                f"Last failure: {self.last_failure_time}"
            )
        
        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        
        except Exception as e:
            self.record_failure()
            raise e


# Global circuit breaker instance for OpenAI API calls
openai_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    success_threshold=2,
    timeout=60.0
)
