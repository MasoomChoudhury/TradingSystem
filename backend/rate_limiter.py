"""
Rate Limiter for Gemini API

Implements:
1. Minimum spacing between requests (ceil(60/RPM) seconds)
2. Concurrency = 1 (no parallel requests)
3. Rolling 60-second token window tracking
4. Exponential backoff retry on 429 errors
"""
import os
import time
import asyncio
import logging
from typing import Callable, Any, Optional
from dataclasses import dataclass, field
from collections import deque
from functools import wraps
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration for Gemini API."""
    rpm: int = 15  # Requests per minute (default for free tier)
    tpm: int = 1_000_000  # Tokens per minute
    max_retries: int = 5
    base_delay: float = 2.0  # Base delay for exponential backoff
    max_delay: float = 60.0  # Maximum delay


class RateLimiter:
    """
    Client-side rate limiter for LLM API calls.
    
    Features:
    - Minimum spacing between requests: ceil(60/RPM) seconds
    - Concurrency = 1: Sequential requests only
    - Token tracking: Rolling 60-second window
    - Exponential backoff on 429 errors
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.config = RateLimitConfig(
            rpm=int(os.environ.get("GEMINI_RPM", "15")),
            tpm=int(os.environ.get("GEMINI_TPM", "1000000"))
        )
        
        # Minimum spacing between requests
        self.min_spacing = 60.0 / self.config.rpm
        
        # Last request timestamp
        self._last_request_time = 0.0
        
        # Semaphore for concurrency = 1
        self._semaphore = asyncio.Semaphore(1)
        self._sync_lock = threading.Lock()
        
        # Token tracking: (timestamp, token_count) tuples
        self._token_window: deque = deque()
        
        # Stats
        self._total_requests = 0
        self._throttled_requests = 0
        self._retry_count = 0
        
        self._initialized = True
        logger.info(f"RateLimiter initialized: RPM={self.config.rpm}, spacing={self.min_spacing:.2f}s")
    
    def _wait_for_spacing(self) -> float:
        """Wait for minimum spacing between requests. Returns wait time."""
        now = time.time()
        elapsed = now - self._last_request_time
        
        if elapsed < self.min_spacing:
            wait_time = self.min_spacing - elapsed
            time.sleep(wait_time)
            self._throttled_requests += 1
            return wait_time
        return 0.0
    
    async def _async_wait_for_spacing(self) -> float:
        """Async version of wait_for_spacing."""
        now = time.time()
        elapsed = now - self._last_request_time
        
        if elapsed < self.min_spacing:
            wait_time = self.min_spacing - elapsed
            await asyncio.sleep(wait_time)
            self._throttled_requests += 1
            return wait_time
        return 0.0
    
    def _check_token_limit(self, estimated_tokens: int) -> bool:
        """Check if we're within the TPM limit."""
        now = time.time()
        cutoff = now - 60  # 60-second window
        
        # Remove old entries
        while self._token_window and self._token_window[0][0] < cutoff:
            self._token_window.popleft()
        
        # Sum tokens in window
        window_tokens = sum(t[1] for t in self._token_window)
        
        if window_tokens + estimated_tokens > self.config.tpm:
            logger.warning(f"Token limit approaching: {window_tokens}/{self.config.tpm} TPM")
            return False
        return True
    
    def _record_tokens(self, tokens: int):
        """Record tokens used."""
        self._token_window.append((time.time(), tokens))
    
    def _get_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = self.config.base_delay * (2 ** attempt)
        return min(delay, self.config.max_delay)
    
    def call_with_rate_limit(
        self,
        func: Callable,
        *args,
        estimated_tokens: int = 1000,
        **kwargs
    ) -> Any:
        """
        Call a function with rate limiting.
        Synchronous version for LangChain/LangGraph compatibility.
        """
        with self._sync_lock:
            # Wait for minimum spacing
            wait_time = self._wait_for_spacing()
            if wait_time > 0:
                logger.debug(f"Throttled request by {wait_time:.2f}s")
            
            # Check token limit
            if not self._check_token_limit(estimated_tokens):
                # Wait a bit for tokens to free up
                time.sleep(self.min_spacing)
            
            self._last_request_time = time.time()
            self._total_requests += 1
        
        # Execute with retry
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                result = func(*args, **kwargs)
                self._record_tokens(estimated_tokens)
                return result
            
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a rate limit error
                if "429" in str(e) or "resource_exhausted" in error_str or "rate" in error_str:
                    delay = self._get_backoff_delay(attempt)
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}/{self.config.max_retries}), backing off {delay:.1f}s")
                    self._retry_count += 1
                    time.sleep(delay)
                    last_error = e
                else:
                    # Non-rate-limit error, don't retry
                    raise
        
        # All retries exhausted
        raise last_error or Exception("Max retries exceeded")
    
    async def async_call_with_rate_limit(
        self,
        coro_func: Callable,
        *args,
        estimated_tokens: int = 1000,
        **kwargs
    ) -> Any:
        """
        Call an async function with rate limiting.
        Uses semaphore for concurrency = 1.
        """
        async with self._semaphore:
            # Wait for minimum spacing
            await self._async_wait_for_spacing()
            
            # Check token limit
            if not self._check_token_limit(estimated_tokens):
                await asyncio.sleep(self.min_spacing)
            
            self._last_request_time = time.time()
            self._total_requests += 1
            
            # Execute with retry
            last_error = None
            for attempt in range(self.config.max_retries):
                try:
                    result = await coro_func(*args, **kwargs)
                    self._record_tokens(estimated_tokens)
                    return result
                
                except Exception as e:
                    error_str = str(e).lower()
                    
                    if "429" in str(e) or "resource_exhausted" in error_str or "rate" in error_str:
                        delay = self._get_backoff_delay(attempt)
                        logger.warning(f"Rate limit hit (attempt {attempt + 1}/{self.config.max_retries}), backing off {delay:.1f}s")
                        self._retry_count += 1
                        await asyncio.sleep(delay)
                        last_error = e
                    else:
                        raise
            
            raise last_error or Exception("Max retries exceeded")
    
    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        return {
            "total_requests": self._total_requests,
            "throttled_requests": self._throttled_requests,
            "retry_count": self._retry_count,
            "tokens_in_window": sum(t[1] for t in self._token_window),
            "config": {
                "rpm": self.config.rpm,
                "tpm": self.config.tpm,
                "min_spacing_seconds": self.min_spacing
            }
        }


def get_rate_limiter() -> RateLimiter:
    """Get the singleton rate limiter instance."""
    return RateLimiter()


def rate_limited(estimated_tokens: int = 1000):
    """
    Decorator to add rate limiting to a function.
    
    Usage:
        @rate_limited(estimated_tokens=2000)
        def call_llm(prompt):
            return llm.invoke(prompt)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter = get_rate_limiter()
            return limiter.call_with_rate_limit(func, *args, estimated_tokens=estimated_tokens, **kwargs)
        return wrapper
    return decorator


def async_rate_limited(estimated_tokens: int = 1000):
    """
    Async decorator to add rate limiting.
    
    Usage:
        @async_rate_limited(estimated_tokens=2000)
        async def call_llm(prompt):
            return await llm.ainvoke(prompt)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            limiter = get_rate_limiter()
            return await limiter.async_call_with_rate_limit(func, *args, estimated_tokens=estimated_tokens, **kwargs)
        return wrapper
    return decorator


# Wrapper for ChatGoogleGenerativeAI that adds rate limiting
class RateLimitedLLM:
    """
    Wrapper around any LLM that adds rate limiting.
    Drop-in replacement for ChatGoogleGenerativeAI.
    """
    
    def __init__(self, llm, estimated_tokens_per_call: int = 2000):
        self._llm = llm
        self._estimated_tokens = estimated_tokens_per_call
        self._limiter = get_rate_limiter()
    
    def invoke(self, *args, **kwargs):
        """Rate-limited invoke."""
        return self._limiter.call_with_rate_limit(
            self._llm.invoke, *args, 
            estimated_tokens=self._estimated_tokens, 
            **kwargs
        )
    
    async def ainvoke(self, *args, **kwargs):
        """Rate-limited async invoke."""
        return await self._limiter.async_call_with_rate_limit(
            self._llm.ainvoke, *args,
            estimated_tokens=self._estimated_tokens,
            **kwargs
        )
    
    def bind_tools(self, tools, **kwargs):
        """Bind tools and return a new rate-limited wrapper."""
        bound_llm = self._llm.bind_tools(tools, **kwargs)
        return RateLimitedLLM(bound_llm, self._estimated_tokens)
    
    def __getattr__(self, name):
        """Proxy other attributes to the underlying LLM."""
        return getattr(self._llm, name)


__all__ = [
    "RateLimitConfig",
    "RateLimiter",
    "get_rate_limiter",
    "rate_limited",
    "async_rate_limited",
    "RateLimitedLLM",
]
