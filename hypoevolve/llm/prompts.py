"""
Prompt templates for HypoEvolve
"""

# System prompt for code mutation
MUTATION_SYSTEM_PROMPT = """You are an expert programmer tasked with evolving code to improve its performance.

Your task is to analyze the current code and suggest improvements using the SEARCH/REPLACE format.

Format your response as:
<<<<<<< SEARCH
[exact code to be replaced]
=======
[improved replacement code]
>>>>>>> REPLACE

Focus on:
1. Algorithmic improvements
2. Performance optimizations
3. Code simplification
4. Bug fixes

Make meaningful changes that could improve the program's performance."""

# System prompt for full code rewrite
REWRITE_SYSTEM_PROMPT = """You are an expert programmer tasked with rewriting code to improve its performance.

Analyze the current code and create a completely new implementation that achieves the same goal but with better performance.

Return only the new code without any explanations or markdown formatting."""

# User prompt template for mutation
MUTATION_USER_TEMPLATE = """Current code:
{current_code}

{context}

{evolution_history}

Please suggest an improvement using the SEARCH/REPLACE format."""

# User prompt template for full rewrite
REWRITE_USER_TEMPLATE = """Current code:
{current_code}

{context}

{evolution_history}

Please provide a complete rewrite of this code with improved performance."""

# Template for adding inspiration examples
INSPIRATION_TEMPLATE = """

Here are some high-performing code examples for inspiration:
{inspiration_text}"""

# Template for evolution history context
EVOLUTION_HISTORY_TEMPLATE = """
## Evolution Context (Iteration {iteration})

### Recent Performance History:
{recent_programs}

### Current Best Programs:
{top_programs}
"""

# Template for recent program entry
RECENT_PROGRAM_TEMPLATE = """- Generation {generation}: Score {score:.4f} {trend}"""

# Template for top program entry
TOP_PROGRAM_TEMPLATE = """- Rank #{rank}: Score {score:.4f} (Gen {generation})
  Code snippet: {snippet}"""


class PromptManager:
    """Manages prompts for HypoEvolve LLM interactions"""

    @staticmethod
    def get_mutation_prompts(
        current_code: str,
        context: str = "",
        inspirations: list = None,
        iteration: int = 0,
        recent_programs: list = None,
        top_programs: list = None,
    ) -> tuple[str, str]:
        """Get system and user prompts for code mutation"""
        system_prompt = MUTATION_SYSTEM_PROMPT

        # Format evolution history
        evolution_history = PromptManager._format_evolution_history(
            iteration, recent_programs or [], top_programs or []
        )

        user_prompt = MUTATION_USER_TEMPLATE.format(
            current_code=current_code,
            context=context,
            evolution_history=evolution_history,
        )

        if inspirations:
            inspiration_text = "\n\n".join(
                [f"Inspiration {i + 1}:\n{code}" for i, code in enumerate(inspirations)]
            )
            user_prompt += INSPIRATION_TEMPLATE.format(
                inspiration_text=inspiration_text
            )

        return system_prompt, user_prompt

    @staticmethod
    def get_rewrite_prompts(
        current_code: str,
        context: str = "",
        inspirations: list = None,
        iteration: int = 0,
        recent_programs: list = None,
        top_programs: list = None,
    ) -> tuple[str, str]:
        """Get system and user prompts for full code rewrite"""
        system_prompt = REWRITE_SYSTEM_PROMPT

        # Format evolution history
        evolution_history = PromptManager._format_evolution_history(
            iteration, recent_programs or [], top_programs or []
        )

        user_prompt = REWRITE_USER_TEMPLATE.format(
            current_code=current_code,
            context=context,
            evolution_history=evolution_history,
        )

        if inspirations:
            inspiration_text = "\n\n".join(
                [f"Inspiration {i + 1}:\n{code}" for i, code in enumerate(inspirations)]
            )
            user_prompt += INSPIRATION_TEMPLATE.format(
                inspiration_text=inspiration_text
            )

        return system_prompt, user_prompt

    @staticmethod
    def _format_evolution_history(
        iteration: int, recent_programs: list, top_programs: list
    ) -> str:
        """Format evolution history for prompt context"""
        if iteration == 0 or (not recent_programs and not top_programs):
            return ""

        # Format recent programs with trend analysis
        recent_text = ""
        if recent_programs:
            for i, program in enumerate(recent_programs[-3:]):  # Last 3 programs
                trend = ""
                if i > 0:
                    prev_score = recent_programs[-3:][i - 1].get("score", 0)
                    curr_score = program.get("score", 0)
                    if curr_score > prev_score:
                        trend = "↗️"
                    elif curr_score < prev_score:
                        trend = "↘️"
                    else:
                        trend = "→"

                recent_text += (
                    RECENT_PROGRAM_TEMPLATE.format(
                        generation=program.get("generation", 0),
                        score=program.get("score", 0),
                        trend=trend,
                    )
                    + "\n"
                )

        # Format top programs with code snippets
        top_text = ""
        if top_programs:
            for i, program in enumerate(top_programs[:3]):  # Top 3 programs
                code = program.get("code", "")
                snippet = code[:100] + "..." if len(code) > 100 else code
                snippet = snippet.replace("\n", " ").strip()

                top_text += (
                    TOP_PROGRAM_TEMPLATE.format(
                        rank=i + 1,
                        score=program.get("score", 0),
                        generation=program.get("generation", 0),
                        snippet=snippet,
                    )
                    + "\n"
                )

        if not recent_text and not top_text:
            return ""

        return EVOLUTION_HISTORY_TEMPLATE.format(
            iteration=iteration,
            recent_programs=recent_text.strip(),
            top_programs=top_text.strip(),
        )
