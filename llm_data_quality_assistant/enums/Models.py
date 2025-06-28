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
    # GPT_3_5_TURBO = "gpt-3.5-turbo"
    # GPT_4 = "gpt-4"
    # GPT_4O = "gpt-4o"
    GPT_4_1_NANO = "gpt-4.1-nano-2025-04-14"  # cheapest
    GPT_4_1_MINI = "gpt-4.1-mini-2025-04-14"


class GeminiModels(Enum):
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite"
    GEMINI_2_5_FLASH_LITE_PREVIEW_06_17 = "gemini-2.5-flash-lite-preview-06-17"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_PRO = "gemini-pro"
    GEMINI_ULTRA = "gemini-ultra"
    GEMMA_3_1B = "gemma-3-1b-it"
    GEMMA_3_12B = "gemma-3-12b-it"
