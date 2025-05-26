from enum import Enum


class OllamaModels(Enum):
    LLAMA3 = "llama3"
    MISTRAL = "mistral"
    GEMMA = "gemma"
    PHI = "phi"
    QWEN = "qwen"
    DEEPSEEK = "deepseek-coder"


class OpenAIModels(Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4O = "gpt-4o"

class GeminiModels(Enum):
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
