"""
Program evaluation for HypoEvolve
"""

import time
import importlib.util
from typing import Dict, Any, Optional, Callable
import tempfile
import os

from hypoevolve.core.program import Program
from hypoevolve.utils import get_logger


logger = get_logger(__name__)


class FunctionEvaluator:
    """OpenEvolve-style function evaluator"""

    def __init__(
        self,
        evaluation_file: str,
        timeout: int = 30,
        max_retries: int = 3,
        temp_dir: Optional[str] = None,
    ):
        self.evaluation_file = evaluation_file
        self.timeout = timeout
        self.max_retries = max_retries
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.evaluation_function = None
        self._load_evaluation_function()

    def _load_evaluation_function(self):
        """Load evaluation function from file"""
        try:
            spec = importlib.util.spec_from_file_location(
                "evaluation_module", self.evaluation_file
            )
            evaluation_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(evaluation_module)

            if hasattr(evaluation_module, "evaluate"):
                self.evaluation_function = evaluation_module.evaluate
                logger.info(f"Evaluation function loaded from: {self.evaluation_file}")
            else:
                raise ValueError("No 'evaluate' function found in evaluation file")

        except Exception as e:
            logger.error(f"Failed to load evaluation function: {e}")
            raise

    def evaluate_program(
        self, program: Program, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate a program using the loaded evaluation function"""

        start_time = time.time()

        try:
            # Create temporary file for the program
            temp_file = self._create_temp_file(program)

            # Run evaluation function
            result = self.evaluation_function(temp_file, context or {})

            # Clean up
            os.unlink(temp_file)

            # Ensure result has required format
            if isinstance(result, (int, float)):
                # Simple score
                metrics = {"score": float(result)}
            elif isinstance(result, dict):
                # Dictionary result
                metrics = result
                if "score" not in metrics:
                    metrics["score"] = 0.0
            else:
                # Fallback
                metrics = {"score": 0.0}

            return {
                "success": True,
                "metrics": metrics,
                "execution_time": time.time() - start_time,
            }

        except Exception as e:
            logger.error(f"Program evaluation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "metrics": {"score": 0.0},
                "execution_time": time.time() - start_time,
            }

    def _create_temp_file(self, program: Program) -> str:
        """Create temporary file for program execution"""

        suffix = {"python": ".py", "javascript": ".js", "java": ".java"}.get(
            program.language, ".py"
        )

        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, dir=self.temp_dir, delete=False
        )

        temp_file.write(program.code)
        temp_file.flush()
        temp_file.close()

        return temp_file.name


class SimpleEvaluator:
    """Simple evaluator using custom evaluation function"""

    def __init__(self, custom_evaluator: Callable):
        """
        Initialize SimpleEvaluator

        Args:
            custom_evaluator: Custom evaluation function (program, context) -> dict
        """
        self.custom_evaluator = custom_evaluator
        logger.info("SimpleEvaluator initialized successfully")

    def evaluate_program(
        self, program: Program, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate program using custom evaluation function"""

        start_time = time.time()

        try:
            # Execute custom evaluation function
            result = self.custom_evaluator(program, context or {})

            # Normalize result format
            if isinstance(result, dict):
                metrics = result
            elif isinstance(result, (int, float)):
                metrics = {"score": float(result)}
            else:
                metrics = {"score": 0.0}

            # Add score key if missing
            if "score" not in metrics:
                metrics["score"] = 0.0

            return {
                "success": True,
                "metrics": metrics,
                "execution_time": time.time() - start_time,
            }

        except Exception as e:
            logger.error(f"Custom evaluation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "metrics": {"score": 0.0},
                "execution_time": time.time() - start_time,
            }
