from typing import Iterator
import ollama
import google.generativeai as genai
from abc import ABC, abstractmethod
from enums import Models


class ChatResponse:
    pass


class AbstractLLMModel(ABC):
    def __init__(self, model_name):
        """
        Initialize the LLM model with a specific model name.
        """
        self.model_name = model_name

    @abstractmethod
    def chat(self, messages, stream=False) -> str | Iterator[str]:
        """
        Run the model with the given prompt and return the response.
        """
        pass

    @abstractmethod
    def generate(self, prompt, stream=False) -> str | Iterator[str]:
        """
        Generate a response based on the given prompt.
        """
        pass


class OllamaModel(AbstractLLMModel):
    def __init__(self, model_name):
        """
        Initialize the Ollama model with a specific model name.
        """
        super().__init__(model_name)


    def chat(self, messages, stream=False) -> str | Iterator[str]:
        """
        Run the Ollama model with the given messages and return the response.
        """
        if stream:
            response = ollama.chat(
                model=self.model_name, messages=messages, stream=True
            )
            return (chunk.get("response", "") for chunk in response)
        else:
            response = ollama.chat(
                model=self.model_name, messages=messages, stream=stream
            )
            return response.get("response", "")

    def generate(self, prompt, stream=False) -> str | Iterator[str]:
        """
        Generate a response using the Ollama model.
        """
        if stream:
            response = ollama.generate(
                model=self.model_name, prompt=prompt, stream=True
            )
            return (chunk.get("response", "") for chunk in response)
        else:
            response = ollama.generate(model=self.model_name, prompt=prompt)
            return response.get("response", "")


class OpenAIModel(AbstractLLMModel):
    def __init__(self, model_name):
        """
        Initialize the OpenAI model with a specific model name.
        """
        super().__init__(model_name)

    def chat(self, messages, stream=False) -> str | Iterator[str]:
        """
        Run the OpenAI model with the given messages and return the response.
        (Implementation depends on the OpenAI API)
        """
        raise NotImplementedError("OpenAI API integration is not implemented yet.")

    def generate(self, prompt, stream=False) -> str | Iterator[str]:
        """
        Generate a response using the OpenAI model.
        (Implementation depends on the OpenAI API)
        """
        raise NotImplementedError("OpenAI API integration is not implemented yet.")


class GeminiModel(AbstractLLMModel):
    def __init__(self, model_name):
        """
        Initialize the Gemini model with a specific model name.
        """
        super().__init__(model_name)
        self.model = genai.GenerativeModel(model_name)

    def chat(self, messages, stream=False) -> str | Iterator[str]:
        """
        Run the Gemini model with the given messages and return the response.
        """
        # Gemini expects a single prompt string, so concatenate messages
        prompt = "\n".join([msg.get("content", str(msg)) for msg in messages])
        if stream:
            response = self.model.generate_content(prompt, stream=True)
            return (chunk.text for chunk in response)
        else:
            response = self.model.generate_content(prompt)
            return response.text

    def generate(self, prompt, stream=False) -> str | Iterator[str]:
        """
        Generate a response using the Gemini model.
        """
        if stream:
            response = self.model.generate_content(prompt, stream=True)
            return (chunk.text for chunk in response)
        else:
            response = self.model.generate_content(prompt)
            return response.text


def get_model(model) -> AbstractLLMModel:
    """
    Factory function to create an LLM model instance based on the model enum.
    Accepts either an OllamaModels, OpenAIModels, or GeminiModels enum value.
    """
    if isinstance(model, Models.OllamaModels):
        return OllamaModel(model.value)
    elif hasattr(Models, "OpenAIModels") and isinstance(model, Models.OpenAIModels):
        return OpenAIModel(model.value)
    elif hasattr(Models, "GeminiModels") and isinstance(model, Models.GeminiModels):
        return GeminiModel(model.value)
    else:
        raise ValueError(f"Unknown model type: {model}")

"""
if __name__ == "__main__":
    gemma_model = OllamaModel("gemma3:1b")
    response = gemma_model.generate("tell me a story abount a dragon", stream=True)
    for chunk in response:
        print(chunk, end="", flush=True)
"""
    

if __name__ == "__main__":
    # List available Gemini models
    gemini_model = GeminiModel("models/gemini-1.5-pro")  # Use your available Gemini model name
    prompt = "Tell me a fun fact about space."
    response = gemini_model.generate(prompt)
    print(response)