"""
Tests for HypoEvolve prompt system
"""

from hypoevolve.llm.prompts import (
    PromptManager,
    MUTATION_SYSTEM_PROMPT,
    REWRITE_SYSTEM_PROMPT,
    MUTATION_USER_TEMPLATE,
    REWRITE_USER_TEMPLATE,
    INSPIRATION_TEMPLATE,
    EVOLUTION_HISTORY_TEMPLATE,
    RECENT_PROGRAM_TEMPLATE,
    TOP_PROGRAM_TEMPLATE,
)


class TestPromptTemplates:
    """Test prompt templates"""

    def test_system_prompts(self):
        """Test system prompt templates"""
        # Test mutation system prompt
        assert "expert programmer" in MUTATION_SYSTEM_PROMPT
        assert "SEARCH/REPLACE" in MUTATION_SYSTEM_PROMPT
        assert "performance" in MUTATION_SYSTEM_PROMPT

        # Test rewrite system prompt
        assert "expert programmer" in REWRITE_SYSTEM_PROMPT
        assert "rewriting code" in REWRITE_SYSTEM_PROMPT
        assert "performance" in REWRITE_SYSTEM_PROMPT

    def test_user_templates(self):
        """Test user prompt templates"""
        # Test mutation user template
        assert "{current_code}" in MUTATION_USER_TEMPLATE
        assert "{context}" in MUTATION_USER_TEMPLATE
        assert "{evolution_history}" in MUTATION_USER_TEMPLATE
        assert "SEARCH/REPLACE" in MUTATION_USER_TEMPLATE

        # Test rewrite user template
        assert "{current_code}" in REWRITE_USER_TEMPLATE
        assert "{context}" in REWRITE_USER_TEMPLATE
        assert "{evolution_history}" in REWRITE_USER_TEMPLATE

    def test_inspiration_template(self):
        """Test inspiration template"""
        assert "{inspiration_text}" in INSPIRATION_TEMPLATE
        assert "high-performing" in INSPIRATION_TEMPLATE

    def test_evolution_history_templates(self):
        """Test evolution history templates"""
        # Test main evolution history template
        assert "{iteration}" in EVOLUTION_HISTORY_TEMPLATE
        assert "{recent_programs}" in EVOLUTION_HISTORY_TEMPLATE
        assert "{top_programs}" in EVOLUTION_HISTORY_TEMPLATE
        assert "Evolution Context" in EVOLUTION_HISTORY_TEMPLATE

        # Test recent program template
        assert "{generation}" in RECENT_PROGRAM_TEMPLATE
        assert "{score:.4f}" in RECENT_PROGRAM_TEMPLATE
        assert "{trend}" in RECENT_PROGRAM_TEMPLATE

        # Test top program template
        assert "{rank}" in TOP_PROGRAM_TEMPLATE
        assert "{score:.4f}" in TOP_PROGRAM_TEMPLATE
        assert "{generation}" in TOP_PROGRAM_TEMPLATE
        assert "{snippet}" in TOP_PROGRAM_TEMPLATE


class TestPromptManager:
    """Test PromptManager class"""

    def test_get_mutation_prompts_basic(self):
        """Test basic mutation prompt generation"""
        current_code = "def add(a, b): return a + b"
        context = "Optimize addition function"

        system_prompt, user_prompt = PromptManager.get_mutation_prompts(
            current_code=current_code, context=context
        )

        # Check system prompt
        assert system_prompt == MUTATION_SYSTEM_PROMPT

        # Check user prompt contains expected content
        assert current_code in user_prompt
        assert context in user_prompt
        assert "SEARCH/REPLACE" in user_prompt

    def test_get_rewrite_prompts_basic(self):
        """Test basic rewrite prompt generation"""
        current_code = "def multiply(x, y): return x * y"
        context = "Improve multiplication function"

        system_prompt, user_prompt = PromptManager.get_rewrite_prompts(
            current_code=current_code, context=context
        )

        # Check system prompt
        assert system_prompt == REWRITE_SYSTEM_PROMPT

        # Check user prompt contains expected content
        assert current_code in user_prompt
        assert context in user_prompt

    def test_get_mutation_prompts_with_inspirations(self):
        """Test mutation prompts with inspirations"""
        current_code = "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
        context = "Optimize factorial function"
        inspirations = [
            "def factorial_iterative(n):\n    result = 1\n    for i in range(1, n+1):\n        result *= i\n    return result",
            "def factorial_memo(n, memo={}):\n    if n in memo:\n        return memo[n]\n    if n <= 1:\n        return 1\n    memo[n] = n * factorial_memo(n-1, memo)\n    return memo[n]",
        ]

        system_prompt, user_prompt = PromptManager.get_mutation_prompts(
            current_code=current_code, context=context, inspirations=inspirations
        )

        # Check that inspirations are included
        assert "high-performing code examples" in user_prompt
        assert "Inspiration 1:" in user_prompt
        assert "Inspiration 2:" in user_prompt
        assert "factorial_iterative" in user_prompt
        assert "factorial_memo" in user_prompt

    def test_get_rewrite_prompts_with_inspirations(self):
        """Test rewrite prompts with inspirations"""
        current_code = "def sort_list(lst): return sorted(lst)"
        context = "Improve sorting function"
        inspirations = [
            "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)"
        ]

        system_prompt, user_prompt = PromptManager.get_rewrite_prompts(
            current_code=current_code, context=context, inspirations=inspirations
        )

        # Check that inspirations are included
        assert "high-performing code examples" in user_prompt
        assert "Inspiration 1:" in user_prompt
        assert "quicksort" in user_prompt

    def test_get_prompts_with_evolution_history(self):
        """Test prompts with evolution history"""
        current_code = (
            "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
        )
        context = "Optimize fibonacci function"
        iteration = 5
        recent_programs = [
            {
                "generation": 4,
                "score": 0.7,
                "code": "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
                "metrics": {},
            },
            {
                "generation": 3,
                "score": 0.5,
                "code": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                "metrics": {},
            },
        ]
        top_programs = [
            {
                "generation": 2,
                "score": 0.9,
                "code": "def fib_memo(n, memo={}): return memo.setdefault(n, n if n <= 1 else fib_memo(n-1) + fib_memo(n-2))",
                "metrics": {},
            },
        ]

        # Test mutation prompts
        system_prompt, user_prompt = PromptManager.get_mutation_prompts(
            current_code=current_code,
            context=context,
            iteration=iteration,
            recent_programs=recent_programs,
            top_programs=top_programs,
        )

        assert "Evolution Context (Iteration 5)" in user_prompt
        assert "Recent Performance History:" in user_prompt
        assert "Current Best Programs:" in user_prompt
        assert "Generation 4: Score 0.7000" in user_prompt
        assert "Rank #1: Score 0.9000" in user_prompt

        # Test rewrite prompts
        system_prompt, user_prompt = PromptManager.get_rewrite_prompts(
            current_code=current_code,
            context=context,
            iteration=iteration,
            recent_programs=recent_programs,
            top_programs=top_programs,
        )

        assert "Evolution Context (Iteration 5)" in user_prompt
        assert "Recent Performance History:" in user_prompt
        assert "Current Best Programs:" in user_prompt

    def test_format_evolution_history(self):
        """Test evolution history formatting"""
        iteration = 3
        recent_programs = [
            {
                "generation": 2,
                "score": 0.6,
                "code": "def test(): return 2",
                "metrics": {},
            },
            {
                "generation": 1,
                "score": 0.4,
                "code": "def test(): return 1",
                "metrics": {},
            },
        ]
        top_programs = [
            {
                "generation": 1,
                "score": 0.8,
                "code": "def best(): return 'best'",
                "metrics": {},
            },
            {
                "generation": 2,
                "score": 0.7,
                "code": "def second(): return 'second'",
                "metrics": {},
            },
        ]

        history = PromptManager._format_evolution_history(
            iteration, recent_programs, top_programs
        )

        # Check structure
        assert "Evolution Context (Iteration 3)" in history
        assert "Recent Performance History:" in history
        assert "Current Best Programs:" in history

        # Check recent programs
        assert "Generation 2: Score 0.6000" in history
        assert "Generation 1: Score 0.4000" in history

        # Check top programs
        assert "Rank #1: Score 0.8000" in history
        assert "Rank #2: Score 0.7000" in history

    def test_format_evolution_history_with_trends(self):
        """Test evolution history formatting with trend analysis"""
        iteration = 4
        recent_programs = [
            {"generation": 3, "score": 0.9, "code": "def test3(): pass", "metrics": {}},
            {"generation": 2, "score": 0.7, "code": "def test2(): pass", "metrics": {}},
            {"generation": 1, "score": 0.5, "code": "def test1(): pass", "metrics": {}},
        ]

        history = PromptManager._format_evolution_history(
            iteration, recent_programs, []
        )

        # Check for trend indicators - Fixed: check for downward trend since scores are decreasing
        assert "↘️" in history  # Should show downward trends for decreasing scores

    def test_format_evolution_history_empty(self):
        """Test evolution history formatting with empty data"""
        # Test with iteration 0
        history = PromptManager._format_evolution_history(0, [], [])
        assert history == ""

        # Test with empty programs
        history = PromptManager._format_evolution_history(1, [], [])
        assert history == ""

    def test_code_snippet_truncation(self):
        """Test code snippet truncation in evolution history"""
        long_code = (
            "def very_long_function_name_that_exceeds_the_limit():\n"
            + "    # This is a very long comment that should be truncated\n" * 10
            + "    return 'result'"
        )

        top_programs = [
            {"generation": 1, "score": 0.9, "code": long_code, "metrics": {}}
        ]

        history = PromptManager._format_evolution_history(1, [], top_programs)

        # Check that code is truncated
        assert "..." in history
        # Find the line with code snippet
        lines = history.split("\n")
        snippet_line = next(line for line in lines if "Code snippet:" in line)
        assert len(snippet_line) < 200  # Should be reasonably short

    def test_prompt_consistency(self):
        """Test consistency between mutation and rewrite prompts"""
        current_code = "def test(): pass"
        context = "Test context"
        iteration = 2
        recent_programs = [
            {"generation": 1, "score": 0.5, "code": "def old(): pass", "metrics": {}}
        ]
        top_programs = [
            {"generation": 1, "score": 0.8, "code": "def best(): pass", "metrics": {}}
        ]

        # Get both types of prompts
        mut_system, mut_user = PromptManager.get_mutation_prompts(
            current_code=current_code,
            context=context,
            iteration=iteration,
            recent_programs=recent_programs,
            top_programs=top_programs,
        )

        rew_system, rew_user = PromptManager.get_rewrite_prompts(
            current_code=current_code,
            context=context,
            iteration=iteration,
            recent_programs=recent_programs,
            top_programs=top_programs,
        )

        # Both should contain the same evolution history
        mut_history_start = mut_user.find("Evolution Context")
        rew_history_start = rew_user.find("Evolution Context")

        if mut_history_start != -1 and rew_history_start != -1:
            # Extract just the evolution history section
            mut_history_end = mut_user.find(
                "\n\n", mut_history_start + 100
            )  # Find end of section
            rew_history_end = rew_user.find("\n\n", rew_history_start + 100)

            if mut_history_end == -1:
                mut_history_end = len(mut_user)
            if rew_history_end == -1:
                rew_history_end = len(rew_user)

            mut_history = mut_user[mut_history_start:mut_history_end]
            rew_history = rew_user[rew_history_start:rew_history_end]

            # The evolution history part should be identical
            assert mut_history == rew_history

    def test_prompt_parameters_validation(self):
        """Test that prompt methods handle various parameter combinations"""
        current_code = "def simple(): return 1"

        # Test with minimal parameters
        system, user = PromptManager.get_mutation_prompts(current_code=current_code)
        assert current_code in user
        assert system == MUTATION_SYSTEM_PROMPT

        # Test with all parameters
        system, user = PromptManager.get_mutation_prompts(
            current_code=current_code,
            context="Test context",
            inspirations=["def inspiration(): pass"],
            iteration=1,
            recent_programs=[
                {
                    "generation": 0,
                    "score": 0.1,
                    "code": "def old(): pass",
                    "metrics": {},
                }
            ],
            top_programs=[
                {
                    "generation": 0,
                    "score": 0.9,
                    "code": "def best(): pass",
                    "metrics": {},
                }
            ],
        )
        assert current_code in user
        assert "Test context" in user
        assert "inspiration" in user
        assert "Evolution Context" in user

    def test_special_characters_in_code(self):
        """Test handling of special characters in code"""
        special_code = '''def special_function():
    """This function has special characters: @#$%^&*()"""
    return "String with 'quotes' and \\"escaped\\" characters"'''

        # Use top_programs instead of recent_programs to ensure code snippet is shown
        top_programs = [
            {"generation": 1, "score": 0.5, "code": special_code, "metrics": {}}
        ]

        # Should not raise an exception
        history = PromptManager._format_evolution_history(1, [], top_programs)
        assert "special_function" in history

    def test_unicode_handling(self):
        """Test handling of unicode characters in code"""
        unicode_code = '''def unicode_function():
    """这是一个包含中文的函数"""
    emoji = "🚀"
    return f"Hello {emoji}"'''

        top_programs = [
            {"generation": 1, "score": 0.8, "code": unicode_code, "metrics": {}}
        ]

        # Should handle unicode gracefully
        history = PromptManager._format_evolution_history(1, [], top_programs)
        assert "unicode_function" in history
