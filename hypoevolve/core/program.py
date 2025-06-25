"""
Program representation and database for HypoEvolve
"""

import os
import json
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable


@dataclass
class Program:
    """Represents a program in the evolution process"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    code: str = ""
    language: str = "python"

    # Evolution metadata
    parent_id: Optional[str] = None
    generation: int = 0
    timestamp: float = field(default_factory=time.time)

    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "code": self.code,
            "language": self.language,
            "parent_id": self.parent_id,
            "generation": self.generation,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "score": self.score,
            **self.metadata,
        }


class ProgramDatabase:
    """MAP-Elites program database with user-defined behavior descriptors"""

    def __init__(
        self,
        population_size: int = 50,
        elite_ratio: float = 0.2,
        grid_size: int = 10,
        descriptor_fn: Optional[Callable[[Program], Tuple[int, ...]]] = None,
    ):
        # 데이터베이스에서 관리할 프로그램 개수
        self.population_size = population_size
        # 제거하지 않고 보존할 프로그램 개수
        self.elite_ratio = elite_ratio
        # 차원 당 구간 개수
        self.grid_size = grid_size
        # 프로그램 목록
        self.programs: Dict[str, Program] = {}
        # 세대 카운트
        self.generation = 0

        # Elite archive (traditional approach)
        self.elite_size = max(1, int(population_size * elite_ratio))
        # MAP-Elites grid: user-defined descriptor -> program_id
        self.map_elites_grid: Dict[Tuple[int, ...], str] = {}
        # Behavior descriptor: use user function or default
        self.descriptor_fn = descriptor_fn or self._default_descriptor

        dimensions = len(self.descriptor_fn(Program()))

        # Grid statistics
        self.grid_stats = {
            "dimensions": dimensions,
            "total_cells": grid_size**dimensions,
            "occupied_cells": 0,
            "coverage": 0.0,
        }

    def _default_descriptor(self, program: Program) -> Tuple[int]:
        """Default 1D descriptor based on score only"""
        # Bin by normalized score (assuming score is between 0-1)
        score_bin = min(int(abs(program.score) * self.grid_size), self.grid_size - 1)

        return (score_bin,)

    def add(self, program: Program) -> None:
        """Add program to database with MAP-Elites grid management"""
        program.generation = self.generation

        # Add to main database
        self.programs[program.id] = program

        # Compute behavior descriptor
        behavior = self.descriptor_fn(program)

        # Update MAP-Elites grid
        existing_id = self.map_elites_grid.get(behavior)
        if existing_id is None or program.score > self.programs[existing_id].score:
            self.map_elites_grid[behavior] = program.id

        # Maintain population size
        if len(self.programs) > self.population_size:
            self._cull_population()

        # Update grid statistics
        self._update_grid_stats()

    def _update_grid_stats(self) -> None:
        """Update grid coverage statistics"""
        # 그리드에 존재하는 프로그램 개수
        self.grid_stats["occupied_cells"] = len(self.map_elites_grid)
        # 그리드 커버리지
        self.grid_stats["coverage"] = (
            self.grid_stats["occupied_cells"] / self.grid_stats["total_cells"]
        )

    def _cull_population(self) -> None:
        """Remove worst programs while preserving grid diversity"""
        # 프로그램 개수가 데이터베이스 크기보다 작으면 리턴
        if len(self.programs) <= self.population_size:
            return

        # Keep programs that are in MAP-Elites grid (they represent unique niches)
        grid_program_ids = set(self.map_elites_grid.values())

        # Sort all programs by score
        sorted_programs = sorted(
            self.programs.values(), key=lambda p: p.score, reverse=True
        )

        # Keep elite programs
        elite_programs = sorted_programs[: self.elite_size]

        # Keep grid programs (diversity preservation)
        grid_programs = [p for p in self.programs.values() if p.id in grid_program_ids]

        # Combine elite and grid programs, avoiding duplicates
        keep_programs = []
        seen_ids = set()

        # First, add elite programs (엘리트 프로그램은 다 넣고)
        for program in elite_programs:
            if program.id not in seen_ids and len(keep_programs) < self.population_size:
                keep_programs.append(program)
                seen_ids.add(program.id)

        # Then, add grid programs (for diversity) if there's space (그리드 프로그램은 다 넣고)
        for program in grid_programs:
            if program.id not in seen_ids and len(keep_programs) < self.population_size:
                keep_programs.append(program)
                seen_ids.add(program.id)

        # Fill remaining slots with random selection from the rest (남는 슬롯에 랜덤 프로그램 넣기)
        remaining_programs = [p for p in sorted_programs if p.id not in seen_ids]
        remaining_slots = self.population_size - len(keep_programs)

        if remaining_slots > 0 and remaining_programs:
            additional_programs = random.sample(
                remaining_programs, min(remaining_slots, len(remaining_programs))
            )
            keep_programs.extend(additional_programs)

        # Ensure we don't exceed population_size
        keep_programs = keep_programs[: self.population_size]

        # Update database
        keep_ids = {p.id for p in keep_programs}
        self.programs = {pid: p for pid, p in self.programs.items() if pid in keep_ids}

        # Clean up MAP-Elites grid
        self.map_elites_grid = {
            behavior: pid
            for behavior, pid in self.map_elites_grid.items()
            if pid in keep_ids
        }

    def sample_parent(self) -> Optional[Program]:
        """Sample a parent program for mutation from MAP-Elites grid"""
        if not self.map_elites_grid:
            # Fallback to random selection from all programs
            if not self.programs:
                return None
            return random.choice(list(self.programs.values()))

        # Uniform random selection from grid elites
        program_id = random.choice(list(self.map_elites_grid.values()))
        return self.programs[program_id]

    def get_best(self) -> Optional[Program]:
        """Get the best program by score"""
        if not self.programs:
            return None
        return max(self.programs.values(), key=lambda p: p.score)

    def get_top_programs(self, n: int = 5) -> List[Program]:
        """Get top N programs by score"""
        sorted_programs = sorted(
            self.programs.values(), key=lambda p: p.score, reverse=True
        )
        return sorted_programs[:n]

    def get_recent_programs(self, n: int = 5) -> List[Program]:
        """Get recent N programs sorted by generation"""
        if not self.programs:
            return []

        sorted_programs = sorted(
            self.programs.values(), key=lambda p: p.generation, reverse=True
        )
        return sorted_programs[:n]

    def get_evolution_history(
        self, n_recent: int = 3, n_top: int = 3
    ) -> tuple[List[dict], List[dict]]:
        """Get evolution history data for prompt context"""
        # Get recent programs as dict
        recent_programs = []
        for program in self.get_recent_programs(n_recent):
            recent_programs.append(
                {
                    "generation": program.generation,
                    "score": program.score,
                    "code": program.code,
                    "metrics": program.metrics,
                }
            )

        # Get top programs as dict
        top_programs = []
        for program in self.get_top_programs(n_top):
            top_programs.append(
                {
                    "generation": program.generation,
                    "score": program.score,
                    "code": program.code,
                    "metrics": program.metrics,
                }
            )

        return recent_programs, top_programs

    def stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if not self.programs:
            return {
                "size": 0,
                "generation": self.generation,
                "best_score": 0.0,
                **self.grid_stats,
            }

        scores = [p.score for p in self.programs.values()]

        return {
            "size": len(self.programs),
            "generation": self.generation,
            "best_score": max(scores),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            **self.grid_stats,
        }

    def save(self, path: str) -> None:
        """Save database to JSON file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Prepare data for serialization
            data = {
                "programs": {
                    pid: program.to_dict() for pid, program in self.programs.items()
                },
                "generation": self.generation,
                "population_size": self.population_size,
                "elite_ratio": self.elite_ratio,
                "grid_size": self.grid_size,
                "map_elites_grid": {str(k): v for k, v in self.map_elites_grid.items()},
                "grid_stats": self.grid_stats,
                "timestamp": time.time(),
            }

            # Save to JSON file
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            raise Exception(f"Failed to save database: {e}")
