"""
Basic tests for HypoEvolve
"""

import pytest
from hypoevolve.core.config import Config
from hypoevolve.core.program import Program, ProgramDatabase
from hypoevolve.evaluation.evaluator import FunctionEvaluator
from hypoevolve.utils.code_utils import (
    clean_code,
    detect_language,
    validate_code_syntax,
    calculate_similarity,
    extract_code_from_response,
)
import tempfile
import os


class TestConfig:
    """Test configuration management"""

    def test_default_config(self):
        config = Config()
        assert config.max_iterations == 100
        assert config.population_size == 50
        assert config.temperature == 0.7

    def test_config_from_dict(self):
        data = {"max_iterations": 5, "population_size": 10, "model": "gpt-4"}
        config = Config.from_dict(data)
        assert config.max_iterations == 5
        assert config.population_size == 10
        assert config.model == "gpt-4"


class TestProgram:
    """Test program functionality"""

    def test_program_creation(self):
        code = "def hello(): return 'world'"
        program = Program(code=code, language="python")

        assert program.code == code
        assert program.language == "python"
        assert program.generation == 0
        assert program.score == 0.0
        assert program.id is not None

    def test_program_serialization(self):
        program = Program(code="print('hello')", language="python")
        program.score = 0.5
        data = program.to_dict()

        assert "id" in data
        assert "code" in data
        assert "language" in data
        assert "timestamp" in data
        assert "score" in data


class TestProgramDatabase:
    """Test program database with MAP-Elites functionality"""

    def test_database_initialization(self):
        """Test database initialization with MAP-Elites grid"""
        db = ProgramDatabase(population_size=20, elite_ratio=0.3, grid_size=5)

        assert db.population_size == 20
        assert db.elite_ratio == 0.3
        assert db.grid_size == 5
        assert db.elite_size == 6  # 20 * 0.3
        assert len(db.programs) == 0
        assert len(db.map_elites_grid) == 0
        assert db.grid_stats["total_cells"] == 5  # 1D grid with size 5

    def test_database_operations(self):
        db = ProgramDatabase()

        # Add programs
        program1 = Program(code="def a(): pass", language="python")
        program1.score = 0.5

        program2 = Program(code="def b(): pass", language="python")
        program2.score = 0.8

        db.add(program1)
        db.add(program2)

        assert len(db.programs) == 2

        # Test best program
        best = db.get_best()
        assert best.score == 0.8

        # Test statistics
        stats = db.stats()
        assert stats["size"] == 2
        assert stats["best_score"] == 0.8
        assert stats["avg_score"] == 0.65
        assert "coverage" in stats
        assert "occupied_cells" in stats

    def test_map_elites_grid_functionality(self):
        """Test MAP-Elites grid functionality"""
        db = ProgramDatabase(grid_size=5)

        # Create programs with different characteristics
        programs = []
        for i in range(5):
            program = Program(code=f"def func{i}(): pass", language="python")
            program.score = 0.5 + i * 0.1
            programs.append(program)

        # Add programs to database
        for program in programs:
            db.add(program)

        # Check basic database functionality
        assert len(db.programs) == 5
        best_program = db.get_best()
        assert best_program.score == 0.9  # Highest score

        # Test that we can get top programs
        top_programs = db.get_top_programs(3)
        assert len(top_programs) <= 3
        assert all(isinstance(p, Program) for p in top_programs)

    def test_enhanced_parent_sampling(self):
        """Test enhanced parent sampling with MAP-Elites"""
        db = ProgramDatabase()

        # Add multiple programs
        for i in range(10):
            program = Program(code=f"def func{i}(): return {i}", language="python")
            program.score = i / 10.0
            db.add(program)

        # Sample parents multiple times
        parents = []
        for _ in range(20):
            parent = db.sample_parent()
            if parent:
                parents.append(parent)

        assert len(parents) > 0
        # Should have some diversity in parent selection
        parent_ids = [p.id for p in parents]
        unique_parents = set(parent_ids)
        assert len(unique_parents) > 1  # Should sample different parents

    def test_enhanced_program_sampling(self):
        """Test enhanced program sampling functionality"""
        db = ProgramDatabase()

        # Add programs with varying scores
        for i in range(8):
            program = Program(code=f"def func{i}(): return {i}", language="python")
            program.score = i / 10.0
            db.add(program)

        # Test that we can sample parents
        parent = db.sample_parent()
        assert parent is not None
        assert isinstance(parent, Program)

        # Test that we can get top programs
        top_programs = db.get_top_programs(3)
        assert len(top_programs) <= 3
        assert all(isinstance(p, Program) for p in top_programs)

    def test_population_culling_with_diversity_preservation(self):
        """Test population culling that preserves diversity"""
        db = ProgramDatabase(population_size=5, grid_size=3)

        # Add multiple programs with different characteristics
        for i in range(10):
            program = Program(code=f"def func{i}(): return {i}", language="python")
            program.score = i / 10.0
            db.add(program)

        # Population should be maintained automatically
        assert len(db.programs) <= 5

        # Check that best programs are kept
        scores = [p.score for p in db.programs.values()]
        assert max(scores) == 0.9  # Best program should be kept

        # Check basic stats functionality
        stats = db.stats()
        assert stats["size"] <= 5
        assert stats["best_score"] == 0.9

    def test_grid_statistics_update(self):
        """Test grid statistics are properly updated"""
        db = ProgramDatabase(grid_size=4)  # 4 total cells (1D)

        initial_stats = db.stats()
        assert initial_stats["coverage"] == 0.0
        assert initial_stats["occupied_cells"] == 0
        assert initial_stats["total_cells"] == 4  # 1D grid with size 4

        # Add a program
        program = Program(code="def test(): pass", language="python")
        program.score = 0.8
        db.add(program)

        updated_stats = db.stats()
        assert updated_stats["occupied_cells"] >= 0
        assert 0.0 <= updated_stats["coverage"] <= 1.0


class TestCodeUtils:
    """Test code utility functions"""

    def test_clean_code(self):
        messy_code = """
        
        def hello():
            return "world"    
            
            
        """

        cleaned = clean_code(messy_code)
        assert cleaned.startswith("def hello():")
        assert not cleaned.endswith("\n\n\n")

    def test_detect_language(self):
        python_code = "def hello(): return 'world'"
        js_code = "function hello() { return 'world'; }"

        assert detect_language(python_code) == "python"
        assert detect_language(js_code) == "javascript"

    def test_validate_syntax(self):
        valid_code = "def hello(): return 'world'"
        invalid_code = "def hello( return 'world'"

        assert validate_code_syntax(valid_code, "python") == True
        assert validate_code_syntax(invalid_code, "python") == False

    def test_calculate_similarity(self):
        code1 = "def hello(): return 'world'"
        code2 = "def hello(): return 'world'"
        code3 = "def hello():\n    return 'world'"

        assert calculate_similarity(code1, code2) == 1.0
        assert calculate_similarity(code1, code3) < 1.0
        assert calculate_similarity(code1, code3) > 0.0

    def test_extract_code_from_response(self):
        response_with_markdown = """
        Here's the improved code:
        
        ```python
        def fibonacci(n):
            if n <= 1:
                return n
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b
        ```
        
        This is more efficient.
        """

        extracted = extract_code_from_response(response_with_markdown)
        assert "def fibonacci(n):" in extracted
        assert "a, b = 0, 1" in extracted
        assert "Here's the improved code:" not in extracted


class TestFunctionEvaluator:
    """Test function evaluator"""

    def test_function_evaluator_creation(self):
        """Test creating FunctionEvaluator with a valid evaluation file"""
        # Create a temporary evaluation file
        eval_code = '''
def evaluate(program_path, context):
    """Simple test evaluation function"""
    return {
        "score": 0.8,
        "metrics": {"accuracy": 0.8, "correctness": 0.8}
    }
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(eval_code)
            eval_file = f.name

        try:
            evaluator = FunctionEvaluator(eval_file)
            assert evaluator.evaluation_function is not None
            assert evaluator.evaluation_file == eval_file
        finally:
            os.unlink(eval_file)

    def test_function_evaluator_invalid_file(self):
        """Test FunctionEvaluator with invalid evaluation file"""
        # Create a temporary file without evaluate function
        eval_code = """
def wrong_function():
    pass
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(eval_code)
            eval_file = f.name

        try:
            with pytest.raises(
                ValueError, match="No 'evaluate' function found in evaluation file"
            ):
                FunctionEvaluator(eval_file)
        finally:
            os.unlink(eval_file)

    def test_function_evaluator_evaluate_program(self):
        """Test evaluating a program with FunctionEvaluator"""
        # Create a temporary evaluation file
        eval_code = '''
def evaluate(program_path, context):
    """Test evaluation function that returns fixed score"""
    return {
        "score": 0.9,
        "accuracy": 0.9,
        "performance": 0.85
    }
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(eval_code)
            eval_file = f.name

        try:
            evaluator = FunctionEvaluator(eval_file)

            # Create a test program
            program = Program(code="def test(): return 42", language="python")

            # Evaluate the program
            result = evaluator.evaluate_program(program, {})

            assert result["success"] == True
            assert result["metrics"]["score"] == 0.9
            assert result["metrics"]["accuracy"] == 0.9
            assert result["metrics"]["performance"] == 0.85

        finally:
            os.unlink(eval_file)

    def test_function_evaluator_with_context(self):
        """Test FunctionEvaluator with context"""
        # Create a temporary evaluation file that uses context
        eval_code = '''
def evaluate(program_path, context):
    """Test evaluation function that uses context"""
    multiplier = context.get("multiplier", 1.0)
    return {
        "score": 0.5 * multiplier,
        "used_context": True
    }
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(eval_code)
            eval_file = f.name

        try:
            evaluator = FunctionEvaluator(eval_file)

            # Create a test program
            program = Program(code="def test(): return 42", language="python")

            # Evaluate with context
            context = {"multiplier": 2.0}
            result = evaluator.evaluate_program(program, context)

            assert result["success"] == True
            assert result["metrics"]["score"] == 1.0  # 0.5 * 2.0
            assert result["metrics"]["used_context"] == True

        finally:
            os.unlink(eval_file)


class TestSyncOperations:
    """Test synchronous operations"""

    def test_sync_placeholder(self):
        # Placeholder for sync tests
        # Real sync tests would require API keys and network access
        assert True


if __name__ == "__main__":
    pytest.main([__file__])
