"""
HypoEvolve - Simplified code evolution framework
"""

import os
import json
from typing import Dict, Any, List, Optional, Callable

from hypoevolve.core.config import Config
from hypoevolve.core.program import Program, ProgramDatabase
from hypoevolve.llm.client import LLMClient
from hypoevolve.evaluation.evaluator import FunctionEvaluator, SimpleEvaluator
from hypoevolve.utils import get_logger
from hypoevolve.utils.code_utils import (
    apply_diff,
    extract_code_from_response,
    clean_code,
    detect_language,
    validate_code_syntax,
    calculate_similarity,
)

logger = get_logger(__name__)


class HypoEvolve:
    """Main HypoEvolve controller"""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize HypoEvolve

        Args:
            config: Configuration object (optional, defaults to Config())
        """
        self.config = config or Config()
        self.database = ProgramDatabase(population_size=self.config.population_size)
        self.llm_client = LLMClient(self.config)
        self.evaluator = None
        self.current_iteration = 0
        self.best_program = None
        self.evaluation_context = {}

        logger.info("HypoEvolve initialized successfully")

    def set_evaluation_function(
        self, evaluation_file: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Set evaluation function from file

        Args:
            evaluation_file: Path to evaluation function file
            context: Evaluation context dictionary
        """
        self.evaluator = FunctionEvaluator(
            evaluation_file,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )
        self.evaluation_context = context or {}
        logger.info(f"Evaluation function set from: {evaluation_file}")

    def set_custom_evaluator(self, custom_evaluator: Callable) -> None:
        """
        Set custom evaluation function

        Args:
            custom_evaluator: Custom evaluation function
        """
        self.evaluator = SimpleEvaluator(custom_evaluator)
        logger.info("Custom evaluator set successfully")

    def evolve(
        self,
        initial_code: str,
        problem_description: str = "",
        max_iterations: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Program]:
        """
        Run code evolution process

        순서:
            - 초기화: 초기 프로그램 평가 후 데이터베이스 저장
            -

        Args:
            initial_code: Initial code to evolve
            problem_description: Description of the problem to solve
            max_iterations: Maximum number of iterations (optional)
            context: Additional context for evaluation

        Returns:
            Best program found during evolution
        """
        if not self.evaluator:
            raise ValueError("Evaluator must be set before running evolution")

        # Set context
        if context:
            self.evaluation_context.update(context)

        # Initialize population
        self._initialize_population(initial_code, problem_description)

        # Evolution loop
        max_iter = max_iterations or self.config.max_iterations

        for iteration in range(max_iter):
            self.current_iteration = iteration + 1
            logger.info(f"Starting iteration {self.current_iteration}/{max_iter}")

            # Generate candidates
            candidates = self._generate_candidates()

            if not candidates:
                logger.warning("No candidates generated, stopping evolution")
                break

            # Evaluate candidates (평가 후 프로그램 스코어 속성 기록)
            self._evaluate_candidates(candidates)

            # Update population (데이터베이스에 추가)
            self._update_population(candidates)

            # Save progress
            if (
                self.config.save_interval > 0
                and self.current_iteration % self.config.save_interval == 0
            ):
                self._save_progress()

            # Check for improvement
            current_best = self.database.get_best()
            if current_best and (
                not self.best_program or current_best.score > self.best_program.score
            ):
                self.best_program = current_best
                logger.info(f"New best program found: score={current_best.score:.4f}")

        logger.info(f"Evolution completed after {self.current_iteration} iterations")
        return self.best_program

    def _initialize_population(self, initial_code: str, problem_description: str):
        """
        Initialize population with initial program

        - 초기 프로그램 실행 및 평가
        - 초기 프로그램 속성 업데이트
        - 초기 프로그램 데이터베이스 저장
        """

        logger.info("Initializing population")

        # Clean and detect language
        clean_initial_code = clean_code(initial_code)
        language = detect_language(clean_initial_code)

        # Create initial program
        initial_program = Program(
            code=clean_initial_code,
            language=language,
            generation=0,
        )

        # Evaluate initial program
        result = self._evaluate_single_program(initial_program)
        if result and result["success"]:
            initial_program.metrics = result["metrics"]
            initial_program.score = result["metrics"].get("score", 0.0)

        # Add to database
        self.database.add(initial_program)
        self.best_program = self.database.get_best()

        logger.info(f"Initial program added with score: {initial_program.score:.4f}")

    def _evaluate_single_program(self, program: Program) -> Optional[Dict[str, Any]]:
        """Evaluate a single program"""
        try:
            result = self.evaluator.evaluate_program(program, self.evaluation_context)
            return result
        except Exception as e:
            logger.error(f"Program evaluation failed: {e}")
            return None

    def _generate_candidates(self) -> List[Program]:
        """
        Generate candidate programs for current iteration

        program 수정 또는 변화를 주어서 후보군 프로그램을 생성

        - 데이터베이스 속 프로그램 중, 일부 코드라인 mutation (80%)
        - 데이터베이스 속 프로그램 중, 전체 코드리인 rewrite (20%)
        - 수정은 데이터베이스 크기의 절반만 수행
        """
        candidates = []

        # Calculate number of mutations and rewrites
        total_candidates = max(1, self.config.population_size // 2)
        num_mutations = int(total_candidates * 0.8)
        num_rewrites = total_candidates - num_mutations

        # Generate mutations
        for _ in range(num_mutations):
            parent = self.database.sample_parent()
            if parent:
                mutated = self._mutate_program(parent)
                if mutated:
                    candidates.append(mutated)

        # Generate rewrites
        for _ in range(num_rewrites):
            parent = self.database.sample_parent()
            if parent:
                rewritten = self._rewrite_program(parent)
                if rewritten:
                    candidates.append(rewritten)

        logger.info(f"Generated {len(candidates)} candidates")
        return candidates

    def _mutate_program(self, parent: Program) -> Optional[Program]:
        """Generate mutation of a program"""
        try:
            # Get inspirations
            inspirations = [p.code for p in self.database.get_top_programs(3)]

            # Get evolution history
            recent_programs, top_programs = self.database.get_evolution_history()

            # Generate mutation
            diff_response = self.llm_client.generate_mutation(
                current_code=parent.code,
                inspirations=inspirations,
                context=f"Iteration {self.current_iteration}",
                iteration=self.current_iteration,
                recent_programs=recent_programs,
                top_programs=top_programs,
            )

            if not diff_response:
                return None

            # Apply diff
            try:
                mutated_code = apply_diff(parent.code, diff_response)
            except Exception as e:
                logger.warning(f"Failed to apply diff: {e}")
                return None

            # Clean and validate
            mutated_code = clean_code(mutated_code)

            if not validate_code_syntax(mutated_code, parent.language):
                logger.warning("Mutated code has invalid syntax")
                return None

            # Check similarity
            similarity = calculate_similarity(parent.code, mutated_code)

            if similarity > 0.95:
                logger.debug("Mutation too similar to parent")
                return None

            # Create new program
            mutated_program = Program(
                code=mutated_code,
                language=parent.language,
                parent_id=parent.id,
                generation=parent.generation + 1,
            )

            return mutated_program

        except Exception as e:
            logger.error(f"Mutation generation failed: {e}")
            return None

    def _rewrite_program(self, parent: Program) -> Optional[Program]:
        """Generate complete rewrite of a program"""
        try:
            # Get inspirations
            inspirations = [p.code for p in self.database.get_top_programs(3)]

            # Get evolution history
            recent_programs, top_programs = self.database.get_evolution_history()

            # Generate rewrite
            rewritten_code = self.llm_client.generate_full_rewrite(
                current_code=parent.code,
                inspirations=inspirations,
                context=f"Iteration {self.current_iteration}",
                iteration=self.current_iteration,
                recent_programs=recent_programs,
                top_programs=top_programs,
            )

            if not rewritten_code:
                return None

            # Extract and clean code
            rewritten_code = extract_code_from_response(rewritten_code)
            rewritten_code = clean_code(rewritten_code)

            # Validate syntax
            if not validate_code_syntax(rewritten_code, parent.language):
                logger.warning("Rewritten code has invalid syntax")
                return None

            # Create new program
            rewritten_program = Program(
                code=rewritten_code,
                language=parent.language,
                parent_id=parent.id,
                generation=parent.generation + 1,
            )

            return rewritten_program

        except Exception as e:
            logger.error(f"Rewrite generation failed: {e}")
            return None

    def _evaluate_candidates(self, candidates: List[Program]):
        """Evaluate candidate programs"""
        logger.info(f"Evaluating {len(candidates)} candidates")

        for program in candidates:
            result = self._evaluate_single_program(program)
            if result and result["success"]:
                program.metrics = result["metrics"]
                program.score = result["metrics"].get("score", 0.0)
            else:
                program.score = 0.0
                program.metrics = {"score": 0.0}

    def _update_population(self, candidates: List[Program]):
        """Update population with new candidates"""
        # Add successful candidates to database
        added_count = 0
        for program in candidates:
            if program.score > 0:
                self.database.add(program)
                added_count += 1

        logger.info(f"Added {added_count} programs to database")

        # Update statistics
        stats = self.database.stats()
        logger.info(
            f"Population: {stats['size']}, Best: {stats.get('best_score', 0):.4f}, "
            f"Avg: {stats.get('avg_score', 0):.4f}"
        )

    def _save_progress(self):
        """Save current progress"""
        if not self.config.output_dir:
            return

        try:
            os.makedirs(self.config.output_dir, exist_ok=True)

            # Save database
            db_path = os.path.join(self.config.output_dir, "database.json")
            self.database.save(db_path)

            # Save best program
            if self.best_program:
                best_path = os.path.join(self.config.output_dir, "best_program.json")
                with open(best_path, "w") as f:
                    json.dump(self.best_program.to_dict(), f, indent=2)

            logger.info(f"Progress saved to {self.config.output_dir}")

        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
