"""LLM integration for Oh My Repos.

This package provides LLM integration for the Oh My Repos application.
"""

from src.llm.chat_adapter import ChatAdapter
from src.llm.prompt_builder import PromptBuilder
from src.llm.reply_extractor import extract_json_dict, extract_json_dict_sync
from src.llm.generator import LLMGenerator

__all__ = [
    "ChatAdapter",
    "PromptBuilder",
    "extract_json_dict",
    "extract_json_dict_sync",
    "LLMGenerator",
]
