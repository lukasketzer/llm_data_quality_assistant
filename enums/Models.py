from enum import Enum


class OllamaModels(Enum):
    GEMMA3_4B = "gemma3:4b"
    GEMMA3_12B = "gemma3:12b"
    GEMMA3_1B = "gemma3:1b"
    DEEPSEEK_R1_7B = "deepseek-r1:7b"
    DEEPSEEK_R1_1_5B = "deepseek-r1:1.5b"
    DEEPSEEK_R1_LATEST = "deepseek-r1:latest"
    QWEN3_14B = "qwen3:14b"
    QWEN3_4B = "qwen3:4b"
    QWEN3_1_7B = "qwen3:1.7b"
    QWEN3_LATEST = "qwen3:latest"
    LLAMA3_LATEST = "llama3:latest"


class OpenAIModels(Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4O = "gpt-4o"


class GeminiModels(Enum):
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_PRO = "gemini-pro"
    GEMINI_ULTRA = "gemini-ultra"
