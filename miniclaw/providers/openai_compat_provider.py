"""OpenAI compatible provider.

Based on miniclaw's design.
"""

import openai
import json
import time
import asyncio
import uuid
from loguru import logger
from miniclaw.providers.base import LLMProvider, ToolCallRequest, LLMResponse


class OpenAICompatProvider(LLMProvider):
    """OpenAI compatible provider."""

    def __init__(self, api_key, base_url=None, default_model='gpt-4o', request_interval=1.0, max_retries=3):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.default_model = default_model
        self.request_interval = request_interval
        self.last_request_time = 0
        self.max_retries = max_retries

    async def chat(self, messages: list, tools: list = None) -> LLMResponse:
        """Chat with the LLM."""
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                current_time = time.time()
                if current_time - self.last_request_time < self.request_interval:
                    wait_time = self.request_interval - (current_time - self.last_request_time)
                    logger.debug(f"Rate limiting: waiting {wait_time:.2f} seconds")
                    await asyncio.sleep(wait_time)
                
                # Make the request
                response = self.client.chat.completions.create(
                    model=self.default_model,
                    messages=messages,
                    tools=tools
                )
                self.last_request_time = time.time()
                
                # Parse response
                if not response.choices or not response.choices[0]:
                    continue
                
                message = response.choices[0].message
                tool_calls = []
                
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        try:
                            arguments = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                        
                        tool_calls.append(ToolCallRequest(
                            id=tool_call.id,
                            name=tool_call.function.name,
                            arguments=arguments
                        ))
                
                return LLMResponse(
                    content=message.content,
                    tool_calls=tool_calls,
                    finish_reason=response.choices[0].finish_reason,
                    usage=response.usage.model_dump() if response.usage else None
                )
                
            except Exception as e:
                logger.debug(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2)
                else:
                    raise

    def generate(self, messages: list, tools: list = None) -> Any:
        """Generate a response from the LLM."""
        try:
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=messages,
                tools=tools
            )
            
            if not response.choices or not response.choices[0]:
                return "Sorry, no response from model"
            
            message = response.choices[0].message
            
            if message.tool_calls:
                tool_calls = []
                for tool_call in message.tool_calls:
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                    
                    tool_calls.append({
                        'id': tool_call.id,
                        'name': tool_call.function.name,
                        'params': arguments
                    })
                return {
                    'content': message.content,
                    'tool_calls': tool_calls
                }
            else:
                return message.content
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
