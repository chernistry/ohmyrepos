"""Utility for extracting structured JSON from LLM responses.

This module provides utilities for extracting JSON blocks from LLM responses,
with various fallback strategies to handle malformed JSON.
"""

import json
import re
import logging
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# Try importing json5, but don't make it a hard requirement
try:
    import json5

    JSON5_AVAILABLE = True
except ImportError:
    json5 = None
    JSON5_AVAILABLE = False


class JsonBlock(BaseModel):
    """Model for validating and representing JSON blocks."""

    text: str = Field(..., description="Text content containing potential JSON")

    @model_validator(mode="before")
    @classmethod
    def check_text_is_str(cls, values):
        if not isinstance(values.get("text", ""), str):
            raise ValueError("Text must be a string")
        return values


class JsonExtractResult(BaseModel):
    """Model for JSON extraction results."""

    data: Optional[Dict[str, Any]] = Field(None, description="Extracted JSON data")
    success: bool = Field(False, description="Whether extraction was successful")
    error: Optional[str] = Field(None, description="Error message if extraction failed")


async def _find_json_block(text: str) -> Optional[str]:
    """
    Asynchronously finds a JSON block in the text, preferring blocks in markdown.

    Args:
        text: The input text to search for a JSON block

    Returns:
        The found JSON block or None if no block is found
    """
    # Pattern for markdown code blocks (json, javascript, or none)
    code_block_match = re.search(
        r"```(?:json|javascript)?\s*(\{[\s\S]*?\})\s*```", text, re.DOTALL
    )
    if code_block_match:
        return code_block_match.group(1).strip()

    # If no code block, find the largest top-level JSON object
    brace_level = 0
    max_len = 0
    best_match = None
    start_index = -1

    for i, char in enumerate(text):
        if char == "{":
            if brace_level == 0:
                start_index = i
            brace_level += 1
        elif char == "}":
            if brace_level > 0:
                brace_level -= 1
                if brace_level == 0 and start_index != -1:
                    length = i - start_index + 1
                    if length > max_len:
                        max_len = length
                        best_match = text[start_index : i + 1]

    return best_match


async def _repair_json_string(s: str) -> str:
    """
    Asynchronously fixes common errors in JSON strings from LLM.

    Args:
        s: The JSON string to fix

    Returns:
        The repaired JSON string
    """
    # Remove trailing commas
    s = re.sub(r",\s*([\}\]])", r"\1", s)
    # Fix unquoted keys - simplified pattern
    s = re.sub(r"([{,]\s*)([a-zA-Z_]\w*)(\s*:)", r'\1"\2"\3', s)
    # Replace single quotes with double quotes (basic)
    s = re.sub(r"':\s*'([^']*)'", r'": "\1"', s)  # For values
    s = re.sub(r"'([\w_]+)':", r'"\1":', s)  # For keys

    # Handle python constants
    s = s.replace("True", "true").replace("False", "false").replace("None", "null")

    return s


async def extract_json_dict(raw_text: Optional[str]) -> JsonExtractResult:
    """
    Robustly extracts a JSON dictionary from a raw LLM response string.
    This function implements a chain of strategies for finding and parsing JSON.

    Args:
        raw_text: The raw LLM response text

    Returns:
        A JsonExtractResult model with extracted data and operation status
    """
    result = JsonExtractResult()

    if not raw_text or not isinstance(raw_text, str):
        result.error = "Invalid input: text is empty or not a string"
        return result

    try:
        # Validate input through Pydantic
        validated_input = JsonBlock(text=raw_text)

        # Find the JSON block
        json_block = await _find_json_block(validated_input.text)
        if not json_block:
            result.error = "No potential JSON block found in the text"
            logger.debug(result.error)
            return result

        # Strategy 1: Try to parse directly
        try:
            result.data = json.loads(json_block)
            result.success = True
            return result
        except json.JSONDecodeError as e:
            logger.debug(f"Initial json.loads failed: {e}. Trying repairs.")

        # Strategy 2: Repair the string and try again
        repaired_block = await _repair_json_string(json_block)
        try:
            result.data = json.loads(repaired_block)
            result.success = True
            return result
        except json.JSONDecodeError as e:
            logger.debug(f"json.loads on repaired string failed: {e}. Trying json5.")

        # Strategy 3: Use json5 for more lenient parsing
        if JSON5_AVAILABLE:
            try:
                result.data = json5.loads(repaired_block)
                result.success = True
                return result
            except Exception as e:
                error_msg = (
                    f"json5 parsing failed: {e}. All parsing strategies exhausted."
                )
                logger.debug(error_msg)
                result.error = error_msg

        # If all else fails, log the failure
        result.error = "All JSON parsing attempts failed for the text block"
        logger.error(result.error)
        return result

    except Exception as e:
        result.error = f"Unexpected error during JSON extraction: {str(e)}"
        logger.exception("Error in extract_json_dict")
        return result


# Synchronous version that works properly in async contexts
def extract_json_dict_sync(raw_text: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Synchronous wrapper for JSON extraction that works in both sync and async contexts.

    Args:
        raw_text: The raw LLM response text

    Returns:
        The extracted JSON dictionary or None in case of an error
    """
    if not raw_text:
        return None

    # Directly try to find and parse JSON without async
    try:
        # Try to find JSON block using regex
        json_block = None

        # Look for code blocks first
        code_block_match = re.search(
            r"```(?:json|javascript)?\s*(\{[\s\S]*?\})\s*```", raw_text, re.DOTALL
        )
        if code_block_match:
            json_block = code_block_match.group(1).strip()
        else:
            # Find largest JSON object
            brace_level = 0
            max_len = 0
            best_match = None
            start_index = -1

            for i, char in enumerate(raw_text):
                if char == "{":
                    if brace_level == 0:
                        start_index = i
                    brace_level += 1
                elif char == "}":
                    if brace_level > 0:
                        brace_level -= 1
                        if brace_level == 0 and start_index != -1:
                            length = i - start_index + 1
                            if length > max_len:
                                max_len = length
                                best_match = raw_text[start_index : i + 1]

            json_block = best_match

        if not json_block:
            logger.debug("No JSON block found in text")
            return None

        # Try direct parsing
        try:
            return json.loads(json_block)
        except json.JSONDecodeError:
            # Try basic repairs
            repaired = json_block
            # Remove trailing commas
            repaired = re.sub(r",\s*([\}\]])", r"\1", repaired)
            # Fix unquoted keys
            repaired = re.sub(r"([{,]\s*)([a-zA-Z_]\w*)(\s*:)", r'\1"\2"\3', repaired)
            # Replace single quotes
            repaired = re.sub(r"':\s*'([^']*)'", r'": "\1"', repaired)
            repaired = re.sub(r"'([\w_]+)':", r'"\1":', repaired)
            # Handle Python constants
            repaired = (
                repaired.replace("True", "true")
                .replace("False", "false")
                .replace("None", "null")
            )

            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                # Try json5 if available
                if JSON5_AVAILABLE:
                    try:
                        return json5.loads(repaired)
                    except Exception:
                        logger.debug("JSON5 parsing failed")
                return None

    except Exception as e:
        logger.exception(f"Error in extract_json_dict_sync: {e}")
        return None
