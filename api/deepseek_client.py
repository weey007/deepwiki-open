"""DeepSeek ModelClient integration."""

import os
import logging
from typing import Dict, Sequence, Optional, Any, List

from adalflow.core.model_client import ModelClient
from adalflow.core.types import (
    CompletionUsage,
    ModelType,
    GeneratorOutput,
)

log = logging.getLogger(__name__)

class DeepSeekClient(ModelClient):
    __doc__ = r"""A component wrapper for the DeepSeek API client.

    DeepSeek provides AI models through their API endpoint.
    The API is compatible with OpenAI's API format.

    Visit https://platform.deepseek.com/ for more details.

    Example:
        ```python
        from api.deepseek_client import DeepSeekClient

        client = DeepSeekClient()
        generator = adal.Generator(
            model_client=client,
            model_kwargs={"model": "deepseek-chat"}
        )
        ```
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the DeepSeek client."""
        super().__init__(*args, **kwargs)
        self.sync_client = self.init_sync_client()
        self.async_client = None  # Initialize async client only when needed

    def init_sync_client(self):
        """Initialize the synchronous DeepSeek client."""
        from api.config import DEEPSEEK_API_KEY
        api_key = DEEPSEEK_API_KEY
        if not api_key:
            log.warning("DEEPSEEK_API_KEY not configured")
            return None

        # Optional import
        from adalflow.utils.lazy_import import safe_import, OptionalPackages
        openai = safe_import(OptionalPackages.OPENAI.value[0], OptionalPackages.OPENAI.value[1])
        
        if openai is None:
            log.error("OpenAI package not installed. Please install with: pip install openai")
            return None

        # DeepSeek uses OpenAI-compatible API
        from openai import OpenAI
        
        # Get base URL from environment or use default
        base_url = os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
        
        return OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def init_async_client(self):
        """Initialize the asynchronous DeepSeek client."""
        from api.config import DEEPSEEK_API_KEY
        api_key = DEEPSEEK_API_KEY
        if not api_key:
            log.warning("DEEPSEEK_API_KEY not configured")
            return None

        # Optional import
        from adalflow.utils.lazy_import import safe_import, OptionalPackages
        openai = safe_import(OptionalPackages.OPENAI.value[0], OptionalPackages.OPENAI.value[1])
        
        if openai is None:
            log.error("OpenAI package not installed. Please install with: pip install openai")
            return None

        from openai import AsyncOpenAI
        
        # Get base URL from environment or use default
        base_url = os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
        
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def get_async_client(self):
        """Get or initialize the async client."""
        if self.async_client is None:
            self.async_client = self.init_async_client()
        return self.async_client

    def generate(
        self,
        messages: Sequence[Dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> GeneratorOutput:
        """Generate a completion using DeepSeek API."""
        if self.sync_client is None:
            raise ValueError("DeepSeek client not properly initialized")

        try:
            response = self.sync_client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )

            # Extract the content from the response
            content = response.choices[0].message.content
            
            # Create usage information
            usage = CompletionUsage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )

            return GeneratorOutput(
                content=content,
                usage=usage,
                model=model,
                finish_reason=response.choices[0].finish_reason,
            )

        except Exception as e:
            log.error(f"DeepSeek API error: {str(e)}")
            raise

    async def agenerate(
        self,
        messages: Sequence[Dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> GeneratorOutput:
        """Asynchronously generate a completion using DeepSeek API."""
        client = self.get_async_client()
        if client is None:
            raise ValueError("DeepSeek async client not properly initialized")

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )

            # Extract the content from the response
            content = response.choices[0].message.content
            
            # Create usage information
            usage = CompletionUsage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )

            return GeneratorOutput(
                content=content,
                usage=usage,
                model=model,
                finish_reason=response.choices[0].finish_reason,
            )

        except Exception as e:
            log.error(f"DeepSeek async API error: {str(e)}")
            raise

    def list_models(self) -> List[ModelType]:
        """List available DeepSeek models."""
        # Return a list of supported DeepSeek models
        return [
            ModelType(
                id="deepseek-chat",
                name="DeepSeek Chat",
                description="DeepSeek's primary chat model",
                context_length=32768,
            ),
            ModelType(
                id="deepseek-coder",
                name="DeepSeek Coder",
                description="DeepSeek's code generation model",
                context_length=32768,
            ),
        ]