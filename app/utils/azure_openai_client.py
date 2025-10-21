import os
from typing import Dict, Any
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# IMPORTANT: user requested this import style
from openai import AzureOpenAI

class AzureOpenAIClient:
    def __init__(self):
        self.api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        self.base = os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.deployment = os.environ.get("AZURE_DEPLOYMENT_NAME")
        self.api_version = os.environ.get("AZURE_VERSION", None)

        if not (self.api_key and self.base and self.deployment):
            logger.warning("Azure OpenAI environment variables not fully set. Client will not call API.")
            self.client = None
            return

        self.client = AzureOpenAI(
            azure_endpoint=self.base,
            api_key=self.api_key,
            api_version=self.api_version
        )

    
    def call_completion(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 300,
        system_message: str = "You are a helpful assistant.",
        functions: list = None,
        function_call: str | dict = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        if not self.client:
            return {"text": "", "raw": None}

        try:
            args = {
                "model": self.deployment,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if functions:
                args["functions"] = functions
                args["function_call"] = function_call

            response = self.client.chat.completions.create(**args)
            choice = response.choices[0].message

            if hasattr(choice, "function_call") and choice.function_call:
                return {
                    "function_call": {
                        "name": choice.function_call.name,
                        "arguments": choice.function_call.arguments,
                    },
                    "raw": response,
                }

            return {"text": choice.content.strip(), "raw": response}

        except Exception as e:
            logger.error(f"AzureOpenAI error: {e}")
            return {"text": "", "raw": None}