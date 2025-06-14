from typing import Iterator, Any
import ollama
from google import genai
from abc import ABC, abstractmethod
from llm_data_quality_assistant.enums import Models
import os


class AbstractLLMModel(ABC):
    def __init__(self, model_name):
        """
        Initialize the LLM model with a specific model name.
        """
        self.model_name = model_name

    @abstractmethod
    def chat(self, messages, format=None) -> str:
        """
        Run the model with the given prompt and return the response.
        """
        pass

    @abstractmethod
    def chat_stream(self, messages, format=None) -> Iterator[str]:
        """
        Run the model with the given prompt and return the response.
        """
        pass

    @abstractmethod
    def generate(self, prompt, format=None) -> str:
        """
        Generate a response based on the given prompt.
        """
        pass

    @abstractmethod
    def generate_stream(self, prompt, format=None) -> Iterator[str]:
        """
        Generate a response based on the given prompt.
        """
        pass


class OllamaModel(AbstractLLMModel):
    def __init__(self, model_name: Models.OllamaModels):
        super().__init__(model_name)

    def chat(self, messages, format=None) -> str:
        response = ollama.chat(
            model=self.model_name.value,
            messages=messages,
            format=format.model_json_schema() if format else None,
            stream=False,
        )
        return response.get("response", "")

    def chat_stream(self, messages, format=None) -> Iterator[str]:
        response = ollama.chat(
            model=self.model_name.value,
            messages=messages,
            format=format.model_json_schema() if format else None,
            stream=True,
        )
        return (chunk.get("response", "") for chunk in response)

    def generate(self, prompt, format=None) -> str:
        response = ollama.generate(
            model=self.model_name.value,
            prompt=prompt,
            stream=False,
            format=format.model_json_schema() if format else None,
        )
        return response.get("response", "")

    def generate_stream(self, prompt, format=None) -> Iterator[str]:
        response = ollama.generate(
            model=self.model_name.value,
            prompt=prompt,
            stream=True,
            format=format.model_json_schema() if format else None,
        )
        return (chunk.get("response", "") for chunk in response)


class OpenAIModel(AbstractLLMModel):
    def __init__(self, model_name):
        super().__init__(model_name)

    def chat(self, messages, format=None) -> str:
        raise NotImplementedError("OpenAI API integration is not implemented yet.")

    def chat_stream(self, messages, format=None) -> Iterator[str]:
        raise NotImplementedError("OpenAI API integration is not implemented yet.")

    def generate(self, prompt, format=None) -> str:
        raise NotImplementedError("OpenAI API integration is not implemented yet.")

    def generate_stream(self, prompt, format=None) -> Iterator[str]:
        raise NotImplementedError("OpenAI API integration is not implemented yet.")


class GeminiModel(AbstractLLMModel):
    def __init__(self, model_name: Models.GeminiModels):
        super().__init__(model_name)
        API_KEY = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=API_KEY)

    def chat(self, messages, format=None) -> str:
        prompt = "\n".join([msg.get("content", str(msg)) for msg in messages])
        config = {}

        if format is not None:
            config.update(
                {
                    "response_mime_type": "application/json",
                    "response_schema": format,
                }
            )

        response = self.client.models.generate_content(
            model=self.model_name.value,
            contents=prompt,
            config=config,  # type: ignore
        )
        return response.text if response.text is not None else ""

    def chat_stream(self, messages, format=None) -> Iterator[str]:
        prompt = "\n".join([msg.get("content", str(msg)) for msg in messages])
        config = {}
        if format is not None:
            config.update(
                {
                    "response_mime_type": "application/json",
                    "response_schema": format,
                }
            )

        response = self.client.models.generate_content_stream(
            model=self.model_name.value,
            contents=prompt,
            config=config,  # type: ignore
        )
        return (chunk.text for chunk in response if chunk.text is not None)

    def generate(self, prompt, format=None) -> str:
        config = {}
        if format is not None:
            config.update(
                {
                    "response_mime_type": "application/json",
                    "response_schema": format,
                }
            )

        response = self.client.models.generate_content(
            model=self.model_name.value,
            contents=prompt,
            config=config,  # type: ignore
        )
        return response.text if response.text is not None else ""

    def generate_stream(self, prompt, format=None) -> Iterator[str]:
        config = {}
        if format is not None:
            config.update(
                {
                    "response_mime_type": "application/json",
                    "response_schema": format,
                }
            )
        response = self.client.models.generate_content_stream(
            model=self.model_name.value,
            contents=prompt,
            config=config,  # type: ignore
        )
        return (chunk.text for chunk in response if chunk.text is not None)


def get_model(model) -> AbstractLLMModel:
    """
    Factory function to create an LLM model instance based on the model enum.
    Accepts either an OllamaModels, OpenAIModels, or GeminiModels enum value.
    """
    if isinstance(model, Models.OllamaModels):
        return OllamaModel(model)
    elif isinstance(model, Models.OpenAIModels):
        return OpenAIModel(model)
    elif isinstance(model, Models.GeminiModels):
        return GeminiModel(model)
    else:
        raise ValueError(f"Unknown model type: {model}")
