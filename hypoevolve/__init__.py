"""Public package exports for the HypoEvolve application layer."""

from .memory.archive import ArchiveEntry, MAPElitesArchive
from .memory.coulomb_archive import CoulombArchive, CoulombDescriptor
from .elg.kernel import (
    LAMBDA_NEG,
    LAMBDA_WRAP,
    atomic_sim,
    tree_distance,
    tree_kernel,
)
from .core.config import (
    ArchiveConfig,
    ConfigError,
    CoulombArchiveConfig,
    EvaluatorConfig,
    HypoEvolveConfig,
    LLMConfig,
    LoggingConfig,
    OutputConfig,
    ParserConfig,
    SearchConfig,
    WorkerConfig,
    load_config,
)
from .core.orchestrator import HypoEvolveController, RunResult
from .data.dataset import (
    ColumnSpec,
    DataFile,
    DatasetAccessor,
    DatasetSchema,
    DatasetSchemaError,
    dataset_schema_from_dict,
    load_dataset_schema,
)
from .skills.evaluation import Evaluator, LLMEvaluator
from .runtime.sandbox import CodeExecutor, ExecutionResult, LocalSubprocessExecutor
from .skills.seed_generation import (
    HypothesisGenerationError,
    TreePairHypothesis,
    generate_random_tree_pair_hypothesis,
)
from .runtime.llm_client import LLMClient, LLMError
from .observability.logger import configure_logger, logger
from .skills.mutation import MutationDecision, steer_mutation
from .skills.elg_compile import (
    ParseError,
    llm_hypothesis_to_natural_language,
    llm_make_hypothesis_measurable,
    parse_hypothesis_text,
)
from .runtime.worker import WorkerResult, WorkerTask
from .runtime.worker import run_worker_task

# Protocol and policy exports
from .core.events import HookBus
from .policies.protocols import SelectionPolicy, StoppingPolicy
from .context.protocols import ContextProvider
from .skills.protocols import (
    SeedGenerationSkill,
    CompileSkill,
    MutationSkill,
    EvaluationSkill,
    ReportingSkill,
)
from .policies.selection import CoulombSelectionPolicy, UCBSelectionPolicy
from .policies.stopping import IterationStoppingPolicy

__all__ = [
    "ArchiveEntry",
    "ColumnSpec",
    "ConfigError",
    "CoulombArchive",
    "CoulombArchiveConfig",
    "CoulombDescriptor",
    "CoulombSelectionPolicy",
    "DataFile",
    "DatasetAccessor",
    "DatasetSchema",
    "DatasetSchemaError",
    "Evaluator",
    "ExecutionResult",
    "LLMEvaluator",
    "HypoEvolveConfig",
    "HypoEvolveController",
    "LAMBDA_NEG",
    "LAMBDA_WRAP",
    "LLMClient",
    "LLMConfig",
    "MAPElitesArchive",
    "HypothesisGenerationError",
    "ParseError",
    "atomic_sim",
    "steer_mutation",
    "tree_distance",
    "tree_kernel",
    "generate_random_tree_pair_hypothesis",
    "llm_hypothesis_to_natural_language",
    "llm_make_hypothesis_measurable",
    "parse_hypothesis_text",
    "run_worker_task",
    # Protocol and policy exports
    "HookBus",
    "SelectionPolicy",
    "StoppingPolicy",
    "ContextProvider",
    "SeedGenerationSkill",
    "CompileSkill",
    "MutationSkill",
    "EvaluationSkill",
    "ReportingSkill",
    "UCBSelectionPolicy",
    "IterationStoppingPolicy",
]
