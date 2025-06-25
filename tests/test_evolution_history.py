"""
Tests for HypoEvolve evolution history functionality
"""

from unittest.mock import Mock, patch
from hypoevolve.core.program import Program, ProgramDatabase
from hypoevolve.llm.prompts import PromptManager
from hypoevolve.llm.client import LLMClient
from hypoevolve.core.config import Config
from hypoevolve import HypoEvolve


class TestEvolutionHistory:
    """Test evolution history functionality"""

    def test_program_database_evolution_history(self):
        """Test ProgramDatabase evolution history methods"""
        db = ProgramDatabase(population_size=10)

        # Add programs with different generations and scores
        programs_data = [
            ("def func1(): return 1", 0.5, 1),
            ("def func2(): return 2", 0.7, 2),
            ("def func3(): return 3", 0.9, 3),
            ("def func4(): return 4", 0.6, 4),
            ("def func5(): return 5", 0.8, 5),
        ]

        for code, score, generation in programs_data:
            program = Program(code=code, score=score, generation=generation)
            db.add(program)

        # Test get_recent_programs
        recent_programs = db.get_recent_programs(3)
        assert len(recent_programs) == 3
        # Should be sorted by generation (descending)
        assert recent_programs[0].generation >= recent_programs[1].generation
        assert recent_programs[1].generation >= recent_programs[2].generation

        # Test get_top_programs
        top_programs = db.get_top_programs(3)
        assert len(top_programs) == 3
        # Should be sorted by score (descending)
        assert top_programs[0].score >= top_programs[1].score
        assert top_programs[1].score >= top_programs[2].score

        # Test get_evolution_history
        recent_history, top_history = db.get_evolution_history(n_recent=2, n_top=2)

        assert len(recent_history) == 2
        assert len(top_history) == 2

        # Check structure of history data
        for entry in recent_history:
            assert "generation" in entry
            assert "score" in entry
            assert "code" in entry
            assert "metrics" in entry

        for entry in top_history:
            assert "generation" in entry
            assert "score" in entry
            assert "code" in entry
            assert "metrics" in entry

    def test_prompt_manager_evolution_history_formatting(self):
        """Test PromptManager evolution history formatting"""
        # Create mock history data
        recent_programs = [
            {
                "generation": 3,
                "score": 0.7,
                "code": "def func(): return 3",
                "metrics": {},
            },
            {
                "generation": 2,
                "score": 0.5,
                "code": "def func(): return 2",
                "metrics": {},
            },
            {
                "generation": 1,
                "score": 0.8,
                "code": "def func(): return 1",
                "metrics": {},
            },
        ]

        top_programs = [
            {
                "generation": 1,
                "score": 0.9,
                "code": "def best(): return 'best'",
                "metrics": {},
            },
            {
                "generation": 2,
                "score": 0.8,
                "code": "def second(): return 'second'",
                "metrics": {},
            },
        ]

        # Test evolution history formatting
        history = PromptManager._format_evolution_history(
            5, recent_programs, top_programs
        )

        assert "Evolution Context (Iteration 5)" in history
        assert "Recent Performance History:" in history
        assert "Current Best Programs:" in history
        assert "Generation 3" in history
        assert "Score 0.9000" in history

        # Test with trend analysis
        assert "↗️" in history or "↘️" in history or "→" in history

    def test_prompt_manager_with_evolution_history(self):
        """Test PromptManager integration with evolution history"""
        recent_programs = [
            {"generation": 2, "score": 0.6, "code": "def test(): pass", "metrics": {}}
        ]

        top_programs = [
            {
                "generation": 1,
                "score": 0.9,
                "code": "def best(): return 1",
                "metrics": {},
            }
        ]

        # Test mutation prompts with evolution history
        system_prompt, user_prompt = PromptManager.get_mutation_prompts(
            current_code="def original(): return 0",
            context="Test context",
            iteration=3,
            recent_programs=recent_programs,
            top_programs=top_programs,
        )

        assert "You are an expert programmer" in system_prompt
        assert "Evolution Context (Iteration 3)" in user_prompt
        assert "Recent Performance History:" in user_prompt
        assert "Current Best Programs:" in user_prompt

        # Test rewrite prompts with evolution history
        system_prompt, user_prompt = PromptManager.get_rewrite_prompts(
            current_code="def original(): return 0",
            context="Test context",
            iteration=3,
            recent_programs=recent_programs,
            top_programs=top_programs,
        )

        assert "rewriting code to improve" in system_prompt
        assert "Evolution Context (Iteration 3)" in user_prompt

    def test_llm_client_with_evolution_history(self):
        """Test LLMClient with evolution history parameters"""
        config = Config(api_key="test-key", model="gpt-3.5-turbo")

        with patch("hypoevolve.llm.client.OpenAI") as mock_openai:
            # Mock OpenAI response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "test response"

            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            llm_client = LLMClient(config)

            recent_programs = [
                {
                    "generation": 2,
                    "score": 0.6,
                    "code": "def test(): pass",
                    "metrics": {},
                }
            ]

            top_programs = [
                {
                    "generation": 1,
                    "score": 0.9,
                    "code": "def best(): return 1",
                    "metrics": {},
                }
            ]

            # Test generate_mutation with evolution history
            result = llm_client.generate_mutation(
                current_code="def original(): return 0",
                context="Test context",
                iteration=3,
                recent_programs=recent_programs,
                top_programs=top_programs,
            )

            assert result == "test response"
            mock_client.chat.completions.create.assert_called_once()

            # Verify the call was made with proper parameters
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            assert len(messages) == 2
            assert "Evolution Context (Iteration 3)" in messages[1]["content"]

    def test_empty_evolution_history(self):
        """Test behavior with empty evolution history"""
        # Test with no history data
        history = PromptManager._format_evolution_history(0, [], [])
        assert history == ""

        # Test with iteration 0
        history = PromptManager._format_evolution_history(
            0, [{"generation": 1, "score": 0.5}], []
        )
        assert history == ""

        # Test prompts with empty history
        system_prompt, user_prompt = PromptManager.get_mutation_prompts(
            current_code="def test(): pass",
            iteration=0,
            recent_programs=[],
            top_programs=[],
        )

        assert "Evolution Context" not in user_prompt

    def test_trend_analysis(self):
        """Test trend analysis in evolution history"""
        recent_programs = [
            {"generation": 1, "score": 0.5, "code": "def func1(): pass", "metrics": {}},
            {"generation": 2, "score": 0.7, "code": "def func2(): pass", "metrics": {}},
            {"generation": 3, "score": 0.6, "code": "def func3(): pass", "metrics": {}},
        ]

        history = PromptManager._format_evolution_history(3, recent_programs, [])

        # Should contain trend indicators
        assert "↗️" in history  # Improvement from gen 1 to 2
        assert "↘️" in history  # Decline from gen 2 to 3

    def test_code_snippet_truncation(self):
        """Test code snippet truncation in top programs"""
        long_code = (
            "def very_long_function():\n"
            + "    # comment\n" * 20
            + "    return 'result'"
        )

        top_programs = [
            {"generation": 1, "score": 0.9, "code": long_code, "metrics": {}}
        ]

        history = PromptManager._format_evolution_history(1, [], top_programs)

        # Code should be truncated to 100 characters + "..."
        assert "..." in history
        assert (
            len([line for line in history.split("\n") if "Code snippet:" in line][0])
            < 200
        )


class TestIntegrationWithEvolutionHistory:
    """Integration tests for evolution history in HypoEvolve"""

    def test_hypoevolve_controller_with_evolution_history(self):
        """Test HypoEvolve controller integration with evolution history"""
        config = Config(
            model="gpt-3.5-turbo",
            max_iterations=2,
            population_size=5,
            api_key="test-key",
        )
        hypo = HypoEvolve(config)

        # Add some programs to database for history
        for i in range(3):
            program = Program(
                code=f"def func{i}(): return {i}", score=i * 0.3, generation=i
            )
            hypo.database.add(program)

        # Set current_iteration to simulate being in evolution loop
        hypo.current_iteration = 1

        # Mock LLM client methods
        with (
            patch.object(hypo.llm_client, "generate_mutation") as mock_mutation,
            patch.object(hypo.llm_client, "generate_full_rewrite") as mock_rewrite,
        ):
            mock_mutation.return_value = """<<<<<<< SEARCH
def func(): pass
=======
def func(): return 1
>>>>>>> REPLACE"""

            mock_rewrite.return_value = "def func(): return 'rewritten'"

            # Test mutation with evolution history
            parent = hypo.database.get_best()
            result = hypo._mutate_program(parent)

            # Verify that evolution history was passed to LLM
            mock_mutation.assert_called_once()
            call_args = mock_mutation.call_args[1]
            assert "iteration" in call_args
            assert "recent_programs" in call_args
            assert "top_programs" in call_args
            assert call_args["iteration"] == 1  # current_iteration was set to 1

            # Test rewrite with evolution history
            result = hypo._rewrite_program(parent)

            # Verify that evolution history was passed to LLM
            mock_rewrite.assert_called_once()
            call_args = mock_rewrite.call_args[1]
            assert "iteration" in call_args
            assert "recent_programs" in call_args
            assert "top_programs" in call_args

    def test_full_evolution_with_history(self):
        """Test full evolution process with history tracking"""
        config = Config(
            model="gpt-3.5-turbo",
            max_iterations=3,
            population_size=5,
            api_key="test-key",
        )
        hypo = HypoEvolve(config)

        # Set up a simple evaluator
        def simple_evaluator(program, context):
            # Simple scoring based on code length (longer = better)
            score = min(len(program.code) / 100.0, 1.0)
            return {"score": score, "length": len(program.code)}

        hypo.set_custom_evaluator(simple_evaluator)

        # Mock LLM responses
        with (
            patch.object(hypo.llm_client, "generate_mutation") as mock_mutation,
            patch.object(hypo.llm_client, "generate_full_rewrite") as mock_rewrite,
        ):
            # Mock progressive improvements
            mock_mutation.side_effect = [
                """<<<<<<< SEARCH
def add(a, b):
    return a + b
=======
def add(a, b):
    \"\"\"Addition function with documentation\"\"\"
    return a + b
>>>>>>> REPLACE""",
                """<<<<<<< SEARCH
def add(a, b):
    \"\"\"Addition function with documentation\"\"\"
    return a + b
=======
def add(a, b):
    \"\"\"Addition function with documentation and validation\"\"\"
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Arguments must be numbers")
    return a + b
>>>>>>> REPLACE""",
            ]

            mock_rewrite.return_value = """def add(a, b):
    \"\"\"Completely rewritten addition function with comprehensive error handling\"\"\"
    if not isinstance(a, (int, float)):
        raise TypeError(f"First argument must be a number, got {type(a)}")
    if not isinstance(b, (int, float)):
        raise TypeError(f"Second argument must be a number, got {type(b)}")
    return a + b"""

            # Run evolution
            result = hypo.evolve(
                initial_code="def add(a, b): return a + b",
                problem_description="Improve addition function",
                max_iterations=3,
            )

            # Verify evolution history was used
            assert mock_mutation.call_count > 0

            # Check that later calls received evolution history
            if mock_mutation.call_count > 1:
                later_call = mock_mutation.call_args_list[-1]
                call_kwargs = later_call[1]
                assert "recent_programs" in call_kwargs
                assert "top_programs" in call_kwargs
                assert call_kwargs["iteration"] > 1

            # Verify final result
            assert result is not None
            assert result.score > 0
            assert len(result.code) > len("def add(a, b): return a + b")

    def test_database_stats_with_evolution_history(self):
        """Test database statistics with evolution history"""
        db = ProgramDatabase(population_size=10)

        # Add programs across multiple generations
        for gen in range(5):
            for i in range(2):
                program = Program(
                    code=f"def func_gen{gen}_prog{i}(): return {gen * 10 + i}",
                    score=(gen * 0.2) + (i * 0.1),
                    generation=gen,
                )
                db.add(program)

        # Test stats
        stats = db.stats()
        assert stats["size"] <= 10  # Should respect population limit
        assert stats["generation"] >= 0  # Should have some generation data
        assert stats["best_score"] > 0  # Should have positive scores
        assert stats["avg_score"] >= 0  # Average should be non-negative

        # Test evolution history retrieval
        recent_programs, top_programs = db.get_evolution_history()
        assert isinstance(recent_programs, list)
        assert isinstance(top_programs, list)
        assert len(recent_programs) <= 5  # Should limit recent programs
        assert len(top_programs) <= 3  # Should limit top programs
