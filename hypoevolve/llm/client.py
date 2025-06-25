"""
LLM client for HypoEvolve
"""

from typing import List
from openai import OpenAI

from hypoevolve.core.config import Config
from hypoevolve.llm.prompts import PromptManager
from hypoevolve.utils import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Simple and efficient LLM client"""

    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.api_base)
        self.prompt_manager = PromptManager()

    def generate_mutation(
        self,
        current_code: str,
        inspirations: List[str] = None,
        context: str = "",
        iteration: int = 0,
        recent_programs: List[dict] = None,
        top_programs: List[dict] = None,
    ) -> str:
        """Generate a code mutation using diff format"""

        system_prompt, user_prompt = self.prompt_manager.get_mutation_prompts(
            current_code=current_code,
            context=context,
            inspirations=inspirations,
            iteration=iteration,
            recent_programs=recent_programs,
            top_programs=top_programs,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return ""

    def generate_full_rewrite(
        self,
        current_code: str,
        inspirations: List[str] = None,
        context: str = "",
        iteration: int = 0,
        recent_programs: List[dict] = None,
        top_programs: List[dict] = None,
    ) -> str:
        """Generate a complete code rewrite"""

        system_prompt, user_prompt = self.prompt_manager.get_rewrite_prompts(
            current_code=current_code,
            context=context,
            inspirations=inspirations,
            iteration=iteration,
            recent_programs=recent_programs,
            top_programs=top_programs,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LLM rewrite failed: {e}")
            return current_code
