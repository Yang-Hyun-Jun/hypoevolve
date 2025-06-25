"""
Tests for Program and ProgramDatabase classes
"""

import pytest
from hypoevolve.core.program import Program, ProgramDatabase


def test_program_creation():
    """Test basic program creation"""
    program = Program(code="print('hello')", language="python")

    assert program.code == "print('hello')"
    assert program.language == "python"
    assert program.score == 0.0
    assert isinstance(program.metadata, dict)


def test_program_serialization():
    """Test program to_dict"""
    program = Program(
        code="def test(): pass",
        language="python",
        score=0.8,
        metadata={"custom_field": "value"},
    )

    # Test to_dict
    data = program.to_dict()
    assert data["code"] == "def test(): pass"
    assert data["score"] == 0.8
    assert data["custom_field"] == "value"


def test_program_database_default():
    """Test ProgramDatabase with default descriptor"""
    db = ProgramDatabase(population_size=10)

    # Add some programs
    for i in range(5):
        program = Program(code=f"def func{i}():\n    return {i}", score=i * 0.2)
        db.add(program)

    assert len(db.programs) == 5
    assert db.get_best().score == 0.8

    # Test basic functionality
    stats = db.stats()
    assert stats["size"] == 5
    assert stats["best_score"] == 0.8
    assert "coverage" in stats
    assert "occupied_cells" in stats


def test_custom_descriptor_function():
    """Test ProgramDatabase with custom descriptor function"""

    def custom_descriptor(program: Program) -> tuple:
        """Custom descriptor based on code length and score"""
        length_bin = min(len(program.code) // 20, 9)  # 0-9
        score_bin = min(int(program.score * 10), 9)  # 0-9
        return (length_bin, score_bin)

    db = ProgramDatabase(
        population_size=10, grid_size=10, descriptor_fn=custom_descriptor
    )

    # Add programs with different lengths and scores
    programs_data = [
        ("x = 1", 0.1),
        ("def short(): pass", 0.5),
        ("def longer_function():\n    return 42\n    # comment", 0.8),
        (
            "class MyClass:\n    def __init__(self):\n        self.value = 0\n    def method(self):\n        return self.value",
            0.9,
        ),
    ]

    for code, score in programs_data:
        program = Program(code=code, score=score)
        db.add(program)

    assert len(db.programs) == 4

    # Test that programs are accessible
    best_program = db.get_best()
    assert best_program is not None
    assert best_program.score == 0.9

    # Test sampling
    parent = db.sample_parent()
    assert parent is not None
    assert parent in db.programs.values()


def test_three_dimensional_descriptor():
    """Test ProgramDatabase with 3D descriptor function"""

    def three_d_descriptor(program: Program) -> tuple:
        """3D descriptor: length, score, and complexity"""
        length_bin = min(len(program.code) // 10, 4)  # 0-4
        score_bin = min(int(program.score * 5), 4)  # 0-4
        complexity_bin = min(program.code.count("\n"), 4)  # 0-4 based on line count
        return (length_bin, score_bin, complexity_bin)

    db = ProgramDatabase(
        population_size=20,
        grid_size=5,  # 5^3 = 125 total cells
        descriptor_fn=three_d_descriptor,
    )

    # Add diverse programs
    test_codes = [
        "x = 1",
        "def f():\n    return 1",
        "def complex():\n    x = 1\n    y = 2\n    return x + y",
        "class A:\n    def __init__(self):\n        pass\n    def method(self):\n        return 42",
    ]

    for i, code in enumerate(test_codes):
        program = Program(code=code, score=i * 0.3)
        db.add(program)

    stats = db.stats()
    assert stats["size"] == 4
    assert "coverage" in stats
    assert "occupied_cells" in stats
    assert "total_cells" in stats


def test_population_management():
    """Test population size management and culling"""

    def simple_descriptor(program: Program) -> tuple:
        return (min(len(program.code) // 5, 9), min(int(program.score * 10), 9))

    db = ProgramDatabase(
        population_size=5,
        elite_ratio=0.4,  # Keep top 2 programs
        descriptor_fn=simple_descriptor,
    )

    # Add more programs than population size
    for i in range(10):
        program = Program(
            code=f"def func{i}(): return {i}" * (i + 1),  # Varying lengths
            score=i * 0.1,
        )
        db.add(program)

    # Should maintain population size
    assert len(db.programs) <= 5

    # Best programs should be kept
    best = db.get_best()
    assert best.score >= 0.5  # Should be one of the higher scoring programs


def test_sampling_methods():
    """Test various sampling methods"""

    def simple_descriptor(program: Program) -> tuple:
        return (min(len(program.code) // 10, 4), min(int(program.score * 5), 4))

    db = ProgramDatabase(
        population_size=10, grid_size=5, descriptor_fn=simple_descriptor
    )

    # Add programs
    for i in range(8):
        program = Program(code=f"def func{i}(): return {i}", score=i * 0.125)
        db.add(program)

    # Test sample_parent
    parent = db.sample_parent()
    assert parent is not None
    assert parent in db.programs.values()

    # Test get_top_programs
    top_programs = db.get_top_programs(3)
    assert len(top_programs) <= 3
    assert all(prog in db.programs.values() for prog in top_programs)


def test_stats():
    """Test database statistics"""

    db = ProgramDatabase(population_size=10)

    # Empty database stats
    stats = db.stats()
    assert stats["size"] == 0
    assert stats["best_score"] == 0.0

    # Add some programs
    scores = [0.1, 0.5, 0.8, 0.3, 0.9]
    for i, score in enumerate(scores):
        program = Program(code=f"def func{i}(): return {i}", score=score)
        db.add(program)

    stats = db.stats()
    assert stats["size"] == 5
    assert stats["best_score"] == 0.9
    assert stats["avg_score"] == sum(scores) / len(scores)
    assert stats["min_score"] == 0.1
    assert "coverage" in stats
    assert "occupied_cells" in stats


if __name__ == "__main__":
    pytest.main([__file__])
