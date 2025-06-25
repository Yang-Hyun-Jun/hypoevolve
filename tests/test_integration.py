"""
Integration tests for HypoEvolve
"""

import tempfile
import os
from unittest.mock import Mock, patch

from hypoevolve import HypoEvolve
from hypoevolve.core.config import Config
from hypoevolve.evaluation.evaluator import FunctionEvaluator


class TestHypoEvolveIntegration:
    """Integration tests for HypoEvolve"""

    def test_hypoevolve_initialization(self):
        """Test HypoEvolve initialization"""
        hypo = HypoEvolve()

        assert hypo.config is not None
        assert hypo.database is not None
        assert hypo.llm_client is not None
        assert hypo.evaluator is None
        assert hypo.current_iteration == 0
        assert hypo.best_program is None

    def test_set_custom_evaluator(self):
        """Test setting custom evaluator"""
        hypo = HypoEvolve()

        def custom_eval(program, context):
            return {"score": 0.8}

        hypo.set_custom_evaluator(custom_eval)
        assert hypo.evaluator is not None

    def test_set_evaluation_function(self):
        """Test setting evaluation function"""
        hypo = HypoEvolve()

        # Create temporary evaluation file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
def evaluate(program_path, context):
    return {"score": 0.9, "accuracy": 0.95}
""")
            eval_file = f.name

        try:
            hypo.set_evaluation_function(eval_file, {"test_mode": True})
            assert isinstance(hypo.evaluator, FunctionEvaluator)
        finally:
            os.unlink(eval_file)

    def test_simple_evolution(self):
        """Test simple evolution process"""
        config = Config(model="gpt-3.5-turbo", max_iterations=2, population_size=5)
        hypo = HypoEvolve(config)

        # Mock LLM client
        with patch.object(
            hypo.llm_client, "generate_mutation", new_callable=Mock
        ) as mock_mutation:
            mock_mutation.return_value = """<<<<<<< SEARCH
def add(a, b):
    return a + b
=======
def add(a, b):
    \"\"\"Addition function\"\"\"
    return a + b
>>>>>>> REPLACE"""

            # Set custom evaluator
            def custom_eval(program, context):
                return {"score": 0.8}

            hypo.set_custom_evaluator(custom_eval)

            # Run evolution
            result = hypo.evolve(
                initial_code="def add(a, b): return a + b",
                problem_description="Add two numbers",
                max_iterations=2,
            )

            assert result is not None
            assert result.code is not None
            assert result.score >= 0.0

    def test_evolution_with_function_evaluator(self):
        """Test evolution with function evaluator"""
        config = Config(model="gpt-3.5-turbo", max_iterations=2, population_size=5)
        hypo = HypoEvolve(config)

        # Create temporary evaluation file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
def evaluate(program_path, context):
    return {"score": 0.9, "accuracy": 0.95}
""")
            eval_file = f.name

        try:
            hypo.set_evaluation_function(eval_file, {"test_mode": True})

            # Mock LLM client
            with patch.object(
                hypo.llm_client, "generate_mutation", new_callable=Mock
            ) as mock_mutation:
                mock_mutation.return_value = """<<<<<<< SEARCH
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
=======
def factorial(n):
    \"\"\"Calculate factorial\"\"\"
    if n <= 1:
        return 1
    return n * factorial(n - 1)
>>>>>>> REPLACE"""

                # Run evolution
                result = hypo.evolve(
                    initial_code="def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)",
                    problem_description="Factorial function",
                    max_iterations=2,
                )

                assert result is not None
                assert result.code is not None
                assert result.score >= 0.0
        finally:
            os.unlink(eval_file)


class TestEndToEndFlow:
    """End-to-end integration tests"""

    def test_complete_evolution_flow(self):
        """Test complete evolution flow from start to finish"""
        config = Config(
            model="gpt-3.5-turbo",
            max_iterations=3,
            population_size=5,
            save_interval=10,  # Don't save during test
        )
        hypo = HypoEvolve(config)

        # Create evaluation function
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
def evaluate(program_path, context):
    # Simple evaluation that returns increasing scores
    import random
    return {
        "score": min(0.9 + random.random() * 0.1, 1.0),
        "accuracy": 0.95,
        "performance": 0.88
    }
""")
            eval_file = f.name

        try:
            # Set evaluation function
            hypo.set_evaluation_function(eval_file, {"test_mode": True})

            # Mock LLM responses
            with (
                patch.object(
                    hypo.llm_client, "generate_mutation", new_callable=Mock
                ) as mock_mutation,
                patch.object(
                    hypo.llm_client, "generate_full_rewrite", new_callable=Mock
                ) as mock_rewrite,
            ):
                mock_mutation.return_value = """<<<<<<< SEARCH
def add(a, b):
    return a + b
=======
def add(a, b):
    \"\"\"Optimized addition\"\"\"
    return a + b
>>>>>>> REPLACE"""

                mock_rewrite.return_value = """def add(a, b):
    \"\"\"Completely rewritten addition function\"\"\"
    return a + b"""

                # Run evolution
                initial_code = "def add(a, b): return a + b"
                result = hypo.evolve(
                    initial_code=initial_code,
                    problem_description="Optimize addition function",
                    max_iterations=3,
                )

                # Verify results
                assert result is not None
                assert result.code is not None
                assert result.score >= 0.0
                assert result.language == "python"

        finally:
            os.unlink(eval_file)
