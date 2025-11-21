"""Prompt builder for LLM interactions.

This module provides functionality to build prompts for LLM interactions.
"""

import logging
from pathlib import Path
from typing import Optional

from src.config import settings

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Prompt builder for LLM interactions.

    This class handles building prompts for LLM interactions.
    """

    def __init__(
        self,
        prompts_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the prompt builder.

        Args:
            prompts_dir: Directory containing prompt templates
        """
        # If prompts_dir is not provided, use the default prompts directory
        if prompts_dir is None:
            # Try to find the prompts directory
            base_dir = settings.base_dir
            self.prompts_dir = base_dir / "prompts"

            # If the prompts directory doesn't exist at the base directory,
            # try to find it relative to the current file
            if not self.prompts_dir.exists():
                current_dir = Path(__file__).resolve().parent
                self.prompts_dir = current_dir.parent.parent / "prompts"
        else:
            self.prompts_dir = prompts_dir

        # Ensure the prompts directory exists
        if not self.prompts_dir.exists():
            raise ValueError(f"Prompts directory not found: {self.prompts_dir}")

        logger.debug(f"Initialized PromptBuilder with prompts_dir: {self.prompts_dir}")

    def build_system_message(self) -> str:
        """Build a system message for the LLM.

        Returns:
            System message text
        """
        return (
            "You are a helpful AI assistant that provides accurate and concise responses. "
            "When asked to analyze or summarize content, focus on the most important information "
            "and provide structured output in the requested format."
        )

    def build_summarize_repo_prompt(
        self,
        description: str,
        readme: str,
    ) -> str:
        """Build a prompt for summarizing a repository.

        Args:
            description: Repository description
            readme: Repository README content

        Returns:
            Prompt text
        """
        # Load the template
        template_path = self.prompts_dir / "summarize_repo.md"
        if not template_path.exists():
            raise ValueError(f"Summarize repo template not found: {template_path}")

        template = template_path.read_text(encoding="utf-8")

        # Build the prompt
        prompt = (
            f"{template}\n\n"
            f"## Repository Description\n{description}\n\n"
            f"## README Content\n{readme}"
        )

        return prompt

    def build_rag_system_message(self) -> str:
        """Build a system message for RAG-based chat.

        Returns:
            System message text
        """
        return (
            "You are a helpful AI assistant for the 'Oh My Repos' application. "
            "You have access to the user's software repositories and can answer questions about them. "
            "Use the provided context from the repositories to answer the user's questions accurately. "
            "If the answer cannot be found in the context, admit it and try to be helpful based on general knowledge, "
            "but clearly distinguish between what is in the context and what is general knowledge. "
            "Provide code snippets when relevant. "
            "Keep responses concise and to the point."
        )

    def build_rag_user_prompt(self, question: str, context: str) -> str:
        """Build a user prompt with context for RAG.

        Args:
            question: User's question
            context: Retrieved context from repositories

        Returns:
            Prompt text with context
        """
        return (
            f"Context from repositories:\n{context}\n\n"
            f"User Question: {question}\n\n"
            "Answer the question based on the context provided above."
        )
