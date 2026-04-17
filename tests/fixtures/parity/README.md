# Parity Fixtures

This directory stores normalized parity fixtures for behavior-preserving reimplementation work.

Scenario directories:
- single_worker_provided_hypothesis
- single_worker_generated_seed
- multi_worker_completed
- duplicate_skip
- steering_failure_skip
- evaluator_repair

Each scenario should eventually contain:
- manifest.json
- input_config.json
- stubbed_llm_responses.json
- normalized_expected/

Normalization targets:
- run ids
- timestamps
- temp paths
- executor work dirs
- non-contract logging noise
