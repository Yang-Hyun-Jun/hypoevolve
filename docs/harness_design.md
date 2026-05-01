# Hypoevolve Research Harness 설계 문서

## 0. 문서 목적

이 문서는 Hypoevolve를 단순한 “LLM 기반 feature mining 알고리즘”이 아니라, **리서치 및 최적화 문제를 위한 closed-loop research harness**로 확장하기 위한 설계 초안이다.

핵심 관점은 다음이다.

> 좋은 가설을 만들고, 엄격하게 의심하고, 재현 가능한 실험으로 검증하고, 실패까지 기억하면서, 점점 더 나은 탐색 분포를 학습하는 시스템.

즉, 목표는 “LLM이 좋은 아이디어를 생성한다”가 아니라, **가설 생성 → 구조화 → 변형 → 실험 → 평가 → 비판 → 기억 → 재탐색**을 재현 가능하고 확장 가능한 소프트웨어 루프로 만드는 것이다.

---

## 1. 코딩 하네스와 리서치 하네스의 차이

요즘 코딩 하네스는 보통 다음 목적을 가진다.

- 테스트 통과
- 빌드 성공
- 타입 체크 성공
- PR 생성
- 코드 리뷰 통과

반면 Hypoevolve가 다루는 리서치/최적화 하네스는 목적이 더 확률적이고 불확실하다.

- 가설 품질 상승
- 탐색 분포 개선
- 재현 가능한 실험 설계
- 데이터 누수 및 과최적화 방지
- 실패 패턴 학습
- 다양한 후보군 유지
- 사람에게 검토 가능한 evidence bundle 생성

따라서 코딩 하네스가 주로 **task completion harness**라면, Hypoevolve는 **belief update + search + validation harness**에 가깝다.

---

## 2. 현재 Hypoevolve 시스템의 하네스적 해석

현재 Hypoevolve는 이미 하네스의 핵심 요소를 상당 부분 갖고 있다.

| 현재 구성 요소 | 하네스 관점 해석 |
|---|---|
| Initial Hypothesis | seed generation |
| ELG, Executable Logic Graph | intermediate representation, IR |
| Mutation | candidate transformation skill |
| LLM Eval | semantic evaluator / critic |
| Score | objective feedback |
| MAP-Elites Archive | memory + diversity-preserving archive |
| UCB-based Parent Sampling | selection policy |
| Controller | orchestrator |
| Worker 1..K | executor pool |
| Mutation History / Score History / Top-K Hypothesis | context and memory |

즉 현재 시스템은 이미 다음 구조를 갖는다.

```text
Initial Hypothesis
  -> ELG compile
  -> parent sampling
  -> mutation
  -> evaluation
  -> score
  -> archive update
  -> next parent selection
  -> repeat
```

이제 필요한 것은 이 구조를 명시적인 소프트웨어 추상화로 분리하는 것이다.

---

## 3. 제안하는 전체 아키텍처

전체 구조는 다음처럼 잡는 것이 좋다.

```text
Research Contract
      ↓
Research Orchestrator
      ↓
Skills
      ↓
Operators / Executors / Evaluators
      ↓
Archive + Failure Memory + Lineage
      ↓
Selection Policy
      ↓
Next Iteration
```

좀 더 레이어별로 나누면 다음과 같다.

```text
┌──────────────────────────────────────────────┐
│ Control Plane                                │
│ - Orchestrator                               │
│ - Scheduler                                  │
│ - Selection Policy                           │
│ - Budget Manager                             │
│ - Hook Dispatcher                            │
│ - Retry / Fallback / Stopping Policy         │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ Research Plane                               │
│ - Hypothesis Generation                      │
│ - ELG Compilation                            │
│ - Mutation                                   │
│ - Experiment Design                          │
│ - Evaluation                                 │
│ - Critique                                   │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ Execution Plane                              │
│ - Feature Computation                        │
│ - Backtest                                   │
│ - Statistical Test                           │
│ - Model Training                             │
│ - Data Snapshot Runtime                      │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ Memory Plane                                 │
│ - MAP-Elites Archive                         │
│ - Candidate Store                            │
│ - Evaluation Store                           │
│ - Failure Memory                             │
│ - Lineage Graph                              │
│ - Artifact Store                             │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ Extension Plane                              │
│ - Skill Registry                             │
│ - Operator Registry                          │
│ - Evaluator Registry                         │
│ - Hook Registry                              │
│ - Domain Plugins                             │
└──────────────────────────────────────────────┘
```

---

## 4. 핵심 루프

리서치/최적화 하네스의 기본 루프는 다음처럼 정의할 수 있다.

```text
observe
  -> hypothesize
  -> formalize
  -> mutate / compose
  -> design experiment
  -> execute
  -> evaluate
  -> critique
  -> archive
  -> select
  -> repeat
```

Hypoevolve에 맞게 바꾸면 다음이다.

```text
ResearchContract 생성
  -> seed hypothesis 생성
  -> hypothesis를 ELG로 compile
  -> archive에서 parent ELG 선택
  -> operator 기반 mutation
  -> schema / leakage / duplicate 검증
  -> experiment 실행
  -> score 및 robustness 평가
  -> critic으로 가설 비판
  -> archive 및 failure memory 업데이트
  -> 다음 parent selection
  -> repeat
```

핵심은 **score가 높다고 바로 좋은 후보로 취급하지 않는 것**이다. 리서치 하네스에서는 좋은 점수의 후보일수록 더 강하게 의심해야 한다.

---

## 5. Research Contract

### 5.1 역할

Research Contract는 한 번의 실험 run이 지켜야 하는 계약이다.

코딩 하네스에서 테스트 스펙이나 issue description이 중요하듯, 리서치 하네스에서는 Research Contract가 중요하다.

Research Contract가 명시하지 않으면 다음 문제가 생긴다.

- objective가 중간에 바뀐다.
- metric이 유리한 방향으로 선택된다.
- 데이터 범위가 불명확해진다.
- 미래 정보 참조가 섞인다.
- publish 기준이 애매해진다.
- 같은 실험을 다시 재현하기 어렵다.

### 5.2 예시 스키마

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class ResearchContract:
    run_id: str
    objective: str
    target: str

    dataset_scope: dict[str, Any]
    time_range: tuple[str, str]
    frequency: str

    allowed_features: list[str]
    forbidden_features: list[str]

    primary_metric: str
    secondary_metrics: list[str]
    validation_protocol: str

    max_iterations: int
    max_cost: float
    max_runtime_seconds: int

    max_elg_depth: int
    max_node_count: int
    max_feature_complexity: int

    novelty_threshold: float
    robustness_criteria: dict[str, Any]
    publish_criteria: dict[str, Any]
```

### 5.3 금융 feature mining 예시

```yaml
run_id: crypto_feature_mining_2026_04
objective: Find robust predictive features for next-day crypto returns
target: returns_t_plus_1
frequency: daily

dataset_scope:
  tickers: [BTC, ETH, XRP, DOGE]
  start_date: "2021-01-01"
  end_date: "2025-12-31"

forbidden_features:
  - future_return
  - future_volume
  - any_column_after_prediction_time

primary_metric: information_coefficient
secondary_metrics:
  - sharpe
  - hit_rate
  - max_drawdown
  - turnover

validation_protocol: rolling_walk_forward
max_iterations: 1000
max_elg_depth: 6
max_node_count: 40
novelty_threshold: 0.8

publish_criteria:
  min_train_score: 0.03
  min_test_score: 0.015
  max_feature_correlation: 0.7
  require_leakage_check: true
  require_ablation: true
```

---

## 6. ELG: Intermediate Representation

### 6.1 역할

ELG는 Hypoevolve의 핵심 IR이다.

자연어 hypothesis는 사람이 이해하기 좋지만, 다음 작업에는 약하다.

- 구조적 mutation
- deterministic validation
- duplicate detection
- lineage tracking
- reproducible execution
- complexity measurement

따라서 자연어 hypothesis를 그대로 굴리기보다, ELG라는 중간 표현으로 변환해야 한다.

### 6.2 ELG 스키마 예시

```python
from dataclasses import dataclass, field
from typing import Any, Literal

NodeType = Literal[
    "feature",
    "condition",
    "operator",
    "aggregation",
    "transform",
    "target",
]

@dataclass
class ELGNode:
    node_id: str
    node_type: NodeType
    name: str
    inputs: list[str]
    params: dict[str, Any]
    output_type: str

@dataclass
class ELG:
    elg_id: str
    version: str
    hypothesis_text: str
    nodes: list[ELGNode]
    edges: list[tuple[str, str]]
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 6.3 ELG에 반드시 포함할 메타데이터

```text
- schema_version
- canonical_hash
- parent_id
- lineage_id
- generated_by_skill
- generated_by_operator
- generation_model
- prompt_template_version
- created_at
- complexity_score
- data_dependencies
- target_horizon
```

### 6.4 Canonicalization

ELG는 canonical form을 가져야 한다.

동일한 의미의 ELG가 표현만 다르게 여러 번 생성되면 다음 문제가 생긴다.

- archive 중복 증가
- score history 오염
- mutation lineage 왜곡
- parent sampling bias 발생

따라서 다음 기능이 필요하다.

```python
def canonicalize_elg(elg: ELG) -> str:
    """ELG를 안정적인 문자열 표현으로 직렬화한다."""
    ...


def hash_elg(elg: ELG) -> str:
    """canonical form 기반 hash를 생성한다."""
    ...
```

---

## 7. Orchestrator

### 7.1 역할

Orchestrator는 하네스의 중심 제어 컴포넌트다.

책임은 다음이다.

- 현재 run state 관리
- 다음 parent 선택
- 어떤 skill을 호출할지 결정
- hook 실행
- 실패 시 retry / fallback 처리
- archive update 지시
- 종료 조건 확인

중요한 점은 orchestrator가 모든 세부 로직을 직접 구현하면 안 된다는 것이다.

Orchestrator는 “실행 흐름을 조정”하고, 실제 작업은 skill, operator, evaluator, hook, policy가 담당해야 한다.

### 7.2 예시 구조

```python
class ResearchOrchestrator:
    def __init__(
        self,
        skills,
        archive,
        selection_policy,
        archive_policy,
        stop_policy,
        hook_bus,
    ):
        self.skills = skills
        self.archive = archive
        self.selection_policy = selection_policy
        self.archive_policy = archive_policy
        self.stop_policy = stop_policy
        self.hooks = hook_bus

    def run(self, contract: ResearchContract):
        state = self.initialize_state(contract)

        self.hooks.emit("on_run_start", state=state, contract=contract)

        while not self.stop_policy.should_stop(state):
            self.hooks.emit("before_parent_sampling", state=state)
            parent = self.selection_policy.select(self.archive, state)
            self.hooks.emit("after_parent_sampling", state=state, parent=parent)

            self.hooks.emit("before_mutation", state=state, parent=parent)
            child = self.skills.mutate_elg(parent, state)
            self.hooks.emit("after_mutation", state=state, child=child)

            self.hooks.emit("before_eval", state=state, candidate=child)
            result = self.skills.evaluate_candidate(child, contract, state)
            self.hooks.emit("after_eval", state=state, candidate=child, result=result)

            decision = self.archive_policy.decide(child, result, state)

            self.hooks.emit(
                "before_archive_insert",
                state=state,
                candidate=child,
                result=result,
                decision=decision,
            )
            self.archive.update(child, result, decision)
            self.hooks.emit(
                "after_archive_insert",
                state=state,
                candidate=child,
                result=result,
                decision=decision,
            )

            self.hooks.emit("on_iteration_end", state=state)
            state.step += 1

        self.hooks.emit("on_run_end", state=state)
        return self.archive.best()
```

---

## 8. Skill 설계

### 8.1 Skill의 정의

Skill은 시스템이 수행할 수 있는 기능 단위다.

예를 들면 다음은 모두 skill이다.

- 초기 가설 생성
- 자연어 hypothesis를 ELG로 변환
- ELG mutation
- 실험 설계
- 후보 평가
- 후보 비판
- report 생성

Skill은 “무엇을 할 수 있는가”의 추상화다.

### 8.2 기본 인터페이스

```python
from typing import Protocol, Any

class Skill(Protocol):
    name: str

    def run(self, input: Any, context: dict[str, Any]) -> Any:
        ...
```

### 8.3 Core Research Skills

| Skill | 입력 | 출력 | 설명 |
|---|---|---|---|
| `generate_seed_hypotheses` | ResearchContract, context | hypothesis list | 초기 가설 생성 |
| `compile_hypothesis_to_elg` | hypothesis text | ELG | 자연어 가설 구조화 |
| `mutate_elg` | parent ELG, context | child ELG | 후보 변형 |
| `design_experiment` | ELG, contract | ExperimentSpec | 검증 프로토콜 생성 |
| `execute_experiment` | ExperimentSpec | RawResult | feature 계산, 백테스트, 통계 검정 실행 |
| `evaluate_candidate` | RawResult, ELG | EvaluationResult | 점수 및 robustness 평가 |
| `critique_candidate` | ELG, EvaluationResult | CritiqueReport | leakage, overfit, novelty, logic 비판 |
| `update_archive` | Candidate, EvaluationResult | ArchiveDecision | 후보 저장 여부 결정 |
| `select_parent` | Archive, State | Candidate | 다음 parent 선택 |
| `generate_report` | Candidate, lineage, evals | Report | 사람이 검토할 수 있는 결과 생성 |

### 8.4 우선 구현할 Skill

초기 MVP에서는 다음 5개를 먼저 안정화하는 것이 좋다.

```text
generate_seed_hypotheses
compile_hypothesis_to_elg
mutate_elg
evaluate_candidate
critique_candidate
```

특히 `critique_candidate`는 중요하다.

리서치 하네스는 좋은 score를 찾는 시스템이 아니라, 좋은 score를 **의심하고 검증하는 시스템**이어야 하기 때문이다.

---

## 9. Operator 설계

### 9.1 Operator의 정의

Operator는 skill 내부에서 쓰이는 더 작은 변환 연산이다.

예를 들어 `mutate_elg`는 skill이고, 그 안에서 쓰는 다음 연산들은 operator다.

- `wrap_not`
- `remove_atomic`
- `change_lookback`
- `adjust_threshold`
- `swap_feature`
- `reformulate_subtree`

### 9.2 기본 인터페이스

```python
from typing import Protocol

class Operator(Protocol):
    name: str
    input_type: str
    output_type: str

    def apply(self, elg: ELG, context: dict) -> ELG:
        ...
```

### 9.3 Operator Library

#### Logical Operators

```text
wrap_not
flip_comparator
add_condition
drop_condition
and_to_or
or_to_and
merge_conditions
split_condition
```

#### Temporal Operators

```text
change_lookback
shift_lag
expand_window
shrink_window
change_smoothing
change_horizon
```

#### Feature-space Operators

```text
swap_feature
replace_factor_family
add_interaction
normalize_feature
drop_redundant_feature
compose_feature
```

#### Numerical Operators

```text
adjust_threshold
change_quantile
rescale_weight
change_zscore_cutoff
clip_outlier_range
```

#### Structural Operators

```text
remove_atomic
reformulate_subtree
rewrite_expression
prune_complexity
introduce_interaction_term
```

#### Experiment Operators

```text
change_metric_bundle
add_ablation
add_placebo_test
change_validation_split
add_regime_split
add_bootstrap_check
```

### 9.4 Operator별 성과 기록

Operator는 단순 코드 조각이 아니라, 하네스가 탐색 분포를 학습하기 위한 중요한 단위다.

따라서 operator별로 다음 통계를 저장해야 한다.

```text
operator_name
applied_count
valid_rate
acceptance_rate
avg_score_lift
median_score_lift
failure_rate
avg_latency
avg_cost
best_candidate_id
```

이 통계가 쌓이면 selection policy가 다음과 같은 결정을 할 수 있다.

- 최근 성과가 좋은 operator를 더 자주 사용한다.
- failure rate가 높은 operator를 일시적으로 비활성화한다.
- score는 낮지만 novelty가 높은 operator를 exploration 목적으로 유지한다.
- 특정 regime에서만 좋은 operator를 조건부로 사용한다.

---

## 10. Hook 설계

### 10.1 Hook의 정의

Hook은 런타임의 특정 지점에 개입할 수 있는 확장 포인트다.

정확히는 다음처럼 구분한다.

```text
hook = 개입 가능한 시점
handler = 그 시점에 실행되는 실제 로직
rule / policy = handler 내부의 조건과 행동
```

예를 들어 `before_mutation`은 hook이고, 여기에 붙는 handler는 다음 같은 일을 할 수 있다.

- parent ELG 복잡도 검사
- 특정 operator 금지
- budget이 낮으면 cheap model로 라우팅
- 동일 후보 반복 방지

### 10.2 HookBus 예시

```python
from collections import defaultdict
from typing import Callable

class HookBus:
    def __init__(self):
        self.handlers: dict[str, list[Callable]] = defaultdict(list)

    def register(self, event_name: str, handler: Callable):
        self.handlers[event_name].append(handler)

    def emit(self, event_name: str, **kwargs):
        for handler in self.handlers[event_name]:
            handler(**kwargs)
```

### 10.3 Lifecycle Hooks

| Hook | 용도 |
|---|---|
| `on_run_start` | run 초기화, config logging |
| `before_seed_generation` | seed 생성 전 context 준비 |
| `after_seed_generation` | seed 중복 제거, novelty check |
| `before_elg_compile` | hypothesis 구체성 검사 |
| `after_elg_compile` | ELG schema validation, hash 생성 |
| `before_parent_sampling` | archive 상태 검사 |
| `after_parent_sampling` | parent lineage 기록 |
| `before_mutation` | operator budget, complexity 제한 |
| `after_mutation` | child ELG validation, duplicate check |
| `before_eval` | experiment spec 고정, data snapshot 선택 |
| `after_eval` | score sanity check, result logging |
| `before_archive_insert` | archive policy 적용 전 검증 |
| `after_archive_insert` | lineage, metrics, artifact 저장 |
| `on_iteration_end` | stagnation, diversity, budget 검사 |
| `on_run_end` | final report, artifact bundle 생성 |

### 10.4 Reliability Hooks

| Hook | 용도 |
|---|---|
| `on_llm_parse_failure` | LLM 출력 파싱 실패 복구 |
| `on_invalid_elg` | invalid ELG quarantine |
| `on_code_execution_error` | 실행 실패 retry/fallback |
| `on_worker_timeout` | worker timeout 처리 |
| `on_repeated_failure` | 특정 operator 일시 비활성화 |
| `on_data_leakage_suspected` | stricter validation 강제 |

### 10.5 Research Quality Hooks

| Hook | 용도 |
|---|---|
| `on_score_spike` | 비정상 고득점 후보 감사 |
| `on_stagnation` | exploration 강화 |
| `on_diversity_collapse` | novelty search 전환 |
| `on_new_best_candidate` | robustness suite 실행 |
| `before_publish` | publish gate 적용 |

### 10.6 중요한 Hook: `on_score_spike`

금융/EDA/feature mining에서는 갑작스러운 score 상승이 가장 위험하다.

좋은 feature를 찾은 것처럼 보이지만 실제로는 다음일 수 있다.

- data leakage
- lookahead bias
- target contamination
- 특정 기간 overfit
- metric artifact
- duplicate target proxy

따라서 `on_score_spike`에서는 다음을 강제하는 것이 좋다.

```text
1. 동일 후보 재실행
2. stricter split 재평가
3. rolling validation 수행
4. ablation test 수행
5. feature-target time alignment 확인
6. 기존 feature와 correlation 확인
7. suspicious feature dependency 검사
8. 통과하지 못하면 publish 후보에서 제외
```

---

## 11. Evaluator Stack

### 11.1 왜 stack이 필요한가

리서치 하네스에서 `evaluate_candidate`를 단일 score 함수로 만들면 위험하다.

score는 높지만 다음 문제가 있을 수 있다.

- 실행은 되지만 schema가 틀림
- 데이터 누수 존재
- train 구간에서만 좋음
- test 구간에서는 약함
- 기존 feature와 거의 동일
- 너무 복잡해서 해석 불가
- 특정 종목이나 특정 기간에만 작동

따라서 evaluator는 stack으로 구성해야 한다.

### 11.2 제안 stack

```text
1. Schema Evaluator
2. Execution Evaluator
3. Statistical Evaluator
4. Robustness Evaluator
5. Novelty Evaluator
6. Leakage Evaluator
7. Complexity Evaluator
8. Semantic Critic
```

### 11.3 각 evaluator 역할

| Evaluator | 역할 | deterministic 여부 |
|---|---|---|
| Schema Evaluator | ELG 구조, type, node validation | deterministic |
| Execution Evaluator | 코드 생성/실행 가능 여부 | deterministic |
| Statistical Evaluator | primary/secondary metric 계산 | deterministic |
| Robustness Evaluator | rolling split, regime split, bootstrap | deterministic |
| Novelty Evaluator | 기존 후보와 중복성 검사 | mixed |
| Leakage Evaluator | 미래 정보, target proxy, time alignment 검사 | deterministic 중심 |
| Complexity Evaluator | node count, depth, expression cost 측정 | deterministic |
| Semantic Critic | 논리적 약점, 해석, failure reason 요약 | LLM-assisted |

### 11.4 EvaluationResult 스키마

```python
from dataclasses import dataclass
from typing import Any, Literal

Decision = Literal[
    "reject",
    "archive",
    "retest",
    "publish_candidate",
]

@dataclass
class EvaluationResult:
    candidate_id: str
    schema_pass: bool
    execution_pass: bool

    primary_score: float
    secondary_scores: dict[str, float]

    robustness_score: float
    novelty_score: float
    leakage_risk: float
    complexity_score: float
    interpretability_score: float

    failure_type: str | None
    critic_notes: str
    decision: Decision

    raw_metrics: dict[str, Any]
    artifacts: dict[str, str]
```

### 11.5 LLM이 담당하면 좋은 부분과 아닌 부분

LLM이 담당하면 좋은 것:

```text
- hypothesis 해석
- semantic novelty 판단 보조
- failure reason 요약
- 다음 mutation 방향 제안
- 사람이 읽을 수 있는 report 생성
- critic note 작성
```

LLM이 담당하면 안 되는 것:

```text
- metric 계산
- schema validation
- code execution 성공 여부
- duplicate hash check
- time alignment check
- leakage rule check
- complexity 계산
```

---

## 12. Archive, Memory, Lineage

### 12.1 Archive의 역할

Archive는 단순 top-k 저장소가 아니다.

리서치 하네스에서 archive는 다음 역할을 한다.

- 좋은 후보 저장
- 실패 후보 저장
- 탐색 다양성 유지
- lineage 추적
- 다음 parent sampling의 기반 제공
- 실패 패턴 학습
- reproducibility 보장

### 12.2 핵심 테이블

최소한 다음 네 테이블은 필요하다.

```text
candidates
evaluations
lineage
failures
```

### 12.3 CandidateRecord

```python
@dataclass
class CandidateRecord:
    candidate_id: str
    run_id: str

    hypothesis_text: str
    elg_hash: str
    elg_version: str

    parent_id: str | None
    operator_name: str | None
    skill_name: str | None

    created_at: str
    status: str  # valid, rejected, failed, archived, publish_candidate

    complexity_score: float
    metadata: dict
```

### 12.4 EvaluationRecord

```python
@dataclass
class EvaluationRecord:
    candidate_id: str
    run_id: str

    primary_score: float
    secondary_scores: dict[str, float]

    robustness_score: float
    novelty_score: float
    leakage_risk: float
    complexity_score: float

    validation_protocol: str
    data_snapshot_id: str
    artifact_refs: dict[str, str]
```

### 12.5 FailureRecord

```python
@dataclass
class FailureRecord:
    candidate_id: str
    run_id: str

    failure_type: str
    failure_stage: str
    message: str
    recoverable: bool

    operator_name: str | None
    skill_name: str | None
    created_at: str
```

### 12.6 Failure 유형

```text
INVALID_ELG
DUPLICATE_CANDIDATE
EXECUTION_ERROR
LEAKAGE_SUSPECTED
OVERFIT_SUSPECTED
LOW_NOVELTY
TOO_COMPLEX
UNSTABLE_SCORE
TRIVIAL_RULE
NO_SIGNAL
TIMEOUT
PARSER_FAILURE
```

### 12.7 Multi-objective Archive

리서치 하네스에서는 단일 best score만 유지하면 위험하다.

다음 축을 함께 봐야 한다.

```text
predictive score
robustness
novelty
simplicity
interpretability
coverage
cost
failure risk
correlation with existing features
```

즉 archive는 “가장 점수 높은 후보 하나”가 아니라, 다양한 특성을 가진 우수 후보군을 유지해야 한다.

---

## 13. Selection Policy

### 13.1 역할

Selection Policy는 다음에 어떤 parent를 탐색할지 결정한다.

이는 단순한 부수 기능이 아니라, 탐색 성능을 좌우하는 핵심 정책이다.

### 13.2 기본 인터페이스

```python
class SelectionPolicy:
    def select(self, archive, state):
        raise NotImplementedError
```

### 13.3 추천 정책

| Policy | 설명 |
|---|---|
| UCBSelectionPolicy | 평균 score와 불확실성을 함께 고려 |
| NoveltySelectionPolicy | 기존 후보와 다른 후보를 우선 탐색 |
| ParetoFrontSelectionPolicy | score, robustness, novelty, simplicity의 Pareto front 선택 |
| DiversityAwareSelectionPolicy | archive coverage가 낮은 영역 우선 선택 |
| BestFirstSelectionPolicy | 현재까지 가장 좋은 후보 주변 exploitation |
| FailureAwareSelectionPolicy | 실패율이 낮은 영역을 우선 선택 |

### 13.4 UCB 예시

```python
import math

class UCBSelectionPolicy:
    def __init__(self, exploration_weight: float = 0.01):
        self.exploration_weight = exploration_weight

    def score(self, record, total_trials: int) -> float:
        mean_score = record.mean_score
        n = max(record.num_trials, 1)
        exploration = math.sqrt(math.log(total_trials + 1) / n)
        return mean_score + self.exploration_weight * exploration

    def select(self, archive, state):
        total_trials = state.iteration + 1
        candidates = archive.get_parent_candidates()
        return max(candidates, key=lambda r: self.score(r, total_trials))
```

---

## 14. Context Layer

### 14.1 역할

LLM이 어떤 정보를 보고 mutation, critique, generation을 수행할지 결정하는 층이다.

현재 Hypoevolve에서도 high context와 low context를 비교하는 실험이 있다. 이를 더 일반화하면 context layer가 된다.

### 14.2 Context Provider

```python
class ContextProvider:
    name: str

    def build(self, state, candidate=None) -> dict:
        raise NotImplementedError
```

### 14.3 추천 Context Provider

```text
ParentELGContextProvider
MutationHistoryContextProvider
ScoreHistoryContextProvider
TopKHypothesisContextProvider
FailureMemoryContextProvider
DataSchemaContextProvider
FeatureMapContextProvider
ResearchContractContextProvider
OperatorStatsContextProvider
```

### 14.4 Context 구성 예시

```text
Mutation Context
- parent ELG
- parent score history
- parent mutation lineage
- recently failed operators
- top-k successful hypotheses
- research contract constraints
- allowed operator list
- forbidden feature list

Critique Context
- candidate ELG
- experiment result
- robustness result
- existing similar candidates
- leakage checklist
- feature dependency list
```

---

## 15. Plugin Architecture

### 15.1 Plugin의 정의

Plugin은 skill, operator, evaluator, hook, context provider 등을 묶어서 시스템에 추가하는 확장 단위다.

Skill은 “무엇을 한다”이고, plugin은 “그 기능들을 어떻게 묶어서 붙인다”이다.

### 15.2 Plugin 인터페이스

```python
class Plugin:
    name: str
    version: str

    def register(self, registry):
        raise NotImplementedError
```

### 15.3 예시

```python
class CryptoResearchPlugin:
    name = "crypto_research"
    version = "0.1.0"

    def register(self, registry):
        registry.operators.register(ChangeLookbackOperator())
        registry.operators.register(SwapMarketMicrostructureFeatureOperator())
        registry.evaluators.register(CryptoBacktestEvaluator())
        registry.hooks.register("on_score_spike", crypto_score_spike_audit)
        registry.context_providers.register(CryptoFeatureMapProvider())
```

### 15.4 Plugin 예시 구성

```text
plugins/
  crypto/
    operators.py
    evaluators.py
    hooks.py
    context.py
    contract_templates.yaml

  macro/
    operators.py
    evaluators.py
    hooks.py
    context.py
    contract_templates.yaml

  equity/
    operators.py
    evaluators.py
    hooks.py
    context.py
    contract_templates.yaml
```

---

## 16. Observability

### 16.1 왜 필요한가

리서치 하네스는 결과만 보면 안 된다.

다음을 알아야 고도화할 수 있다.

- 어떤 operator가 score 개선에 기여했는가
- 어떤 operator가 invalid 후보를 많이 만드는가
- 어느 모델이 어떤 skill에서 잘 작동하는가
- high context가 실제로 도움이 되는가
- archive diversity가 유지되는가
- score spike가 얼마나 자주 발생하는가
- 실패 유형이 특정 stage에 몰리는가

### 16.2 최소 추적 지표

```text
run_id
iteration
candidate_id
parent_id
operator_name
skill_name
model_name
prompt_template_version
latency
cost
primary_score
robustness_score
novelty_score
leakage_risk
complexity_score
failure_type
archive_decision
```

### 16.3 Dashboard 지표

```text
Best Score over Iteration
Mean Score over Iteration
Archive Coverage
Operator Acceptance Rate
Operator Average Score Lift
Failure Type Distribution
Score Spike Count
Model Cost per Accepted Candidate
Context Size vs Score
Robustness Score Distribution
```

---

## 17. Runtime / Execution

### 17.1 필요한 기능

리서치 하네스가 커지면 runtime layer가 필요하다.

```text
worker queue
async execution
timeout
retry
checkpoint
cache
model router
sandbox execution
data snapshot management
artifact store
```

### 17.2 Worker 구조

```python
class Worker:
    def __init__(self, worker_id, skill_registry, hook_bus):
        self.worker_id = worker_id
        self.skills = skill_registry
        self.hooks = hook_bus

    def execute(self, task):
        self.hooks.emit("before_task", task=task, worker_id=self.worker_id)
        try:
            result = self.skills.run(task.skill_name, task.input, task.context)
            self.hooks.emit("after_task", task=task, result=result)
            return result
        except Exception as e:
            self.hooks.emit("on_worker_failure", task=task, error=e)
            raise
```

### 17.3 Checkpoint

각 run은 언제든 재시작 가능해야 한다.

저장해야 할 것:

```text
ResearchContract
current iteration
archive snapshot
candidate records
evaluation records
failure records
random seed
model versions
prompt versions
data snapshot id
operator stats
```

---

## 18. 추천 폴더 구조

```text
hypoevolve/
  core/
    orchestrator.py
    state.py
    contract.py
    events.py
    registry.py

  schema/
    elg.py
    candidate.py
    experiment.py
    evaluation.py
    lineage.py

  skills/
    seed_generation.py
    elg_compile.py
    mutation.py
    experiment_design.py
    execution.py
    evaluation.py
    critique.py
    reporting.py

  operators/
    logical.py
    temporal.py
    feature.py
    numerical.py
    structural.py
    experiment.py
    registry.py

  evaluators/
    schema_eval.py
    execution_eval.py
    statistical_eval.py
    robustness_eval.py
    leakage_eval.py
    novelty_eval.py
    complexity_eval.py
    semantic_critic.py

  policies/
    selection.py
    archive_policy.py
    retry_policy.py
    routing_policy.py
    stopping_policy.py
    publish_policy.py

  hooks/
    hook_bus.py
    lifecycle.py
    reliability.py
    research_quality.py
    observability.py

  memory/
    archive.py
    lineage.py
    failure_memory.py
    artifact_store.py
    candidate_store.py

  context/
    providers.py
    feature_map.py
    hypothesis_map.py
    failure_map.py

  runtime/
    worker.py
    queue.py
    checkpoint.py
    model_router.py
    sandbox.py

  observability/
    tracing.py
    metrics.py
    dashboard.py

  plugins/
    crypto/
    macro/
    equity/

  cli/
    run.py
    inspect.py
    report.py
```

핵심 원칙은 `core`를 작게 유지하는 것이다.

복잡한 로직은 다음으로 분리한다.

```text
skills
operators
evaluators
hooks
policies
plugins
```

---

## 19. 오픈소스 코딩 하네스에서 가져올 수 있는 추상화

최근 코딩 하네스들은 이름은 다르지만 비슷한 추상화를 갖는다.

| 코딩 하네스 구성 | 리서치 하네스 대응 |
|---|---|
| Agent Loop | Hypothesis Evolution Loop |
| Tool | Skill / Evaluator / Executor |
| Action | Mutation / Experiment / Critique Action |
| Observation | Score / Error / Evaluation Result |
| Event Stream | Research Event Log |
| Hook / Middleware | Validation, Audit, Retry, Guardrail Hook |
| Plugin | Domain Pack / Operator Pack / Evaluator Pack |
| Context Provider | Data, Feature, Archive, Failure Context Provider |
| Repo Map | Feature Map / Hypothesis Map |
| Sandbox | Experiment Runtime / Data Snapshot Runtime |
| Reviewer Agent | Critic / Robustness Evaluator |
| Checkpoint | Run Snapshot / Archive Snapshot |
| Human Approval | Publish Gate / Expensive Experiment Approval |

즉 Hypoevolve도 다음 패턴을 가져올 수 있다.

```text
Action / Observation 구조
Event stream 기반 trace
Tool / Skill registry
Hook / Middleware 주입
Context provider 분리
Sandbox runtime
Checkpoint / restore
Plugin packaging
Reviewer / Critic 역할 분리
```

---

## 20. MVP 개발 로드맵

### Phase 1. Schema와 재현성

목표: 같은 run을 다시 돌릴 수 있게 만든다.

구현 항목:

```text
ResearchContract 도입
ELG schema 고정
canonical hash 도입
CandidateRecord 정의
EvaluationRecord 정의
run_id / candidate_id 체계 도입
prompt template version 기록
model version 기록
```

### Phase 2. Evaluator Stack

목표: 나쁜 후보를 빨리 버린다.

구현 항목:

```text
Schema Evaluator
Execution Evaluator
Basic Statistical Evaluator
Leakage Rule Checker
Complexity Evaluator
Duplicate Checker
```

### Phase 3. Operator Registry

목표: mutation을 실험 가능한 단위로 분리한다.

구현 항목:

```text
Operator interface
Operator registry
Logical operators
Temporal operators
Feature operators
Structural operators
Operator별 성능 logging
```

### Phase 4. Hook Bus

목표: controller를 더럽히지 않고 정책을 추가한다.

구현 항목:

```text
HookBus
before_mutation
after_mutation
before_eval
after_eval
on_score_spike
on_stagnation
on_worker_failure
before_publish
```

### Phase 5. Archive 고도화

목표: 탐색 분포가 좋아지고 있는지 추적 가능하게 한다.

구현 항목:

```text
MAP-Elites Archive 정리
Failure Memory 추가
Lineage Graph 추가
Multi-objective ranking
Archive coverage metric
Failed candidate context provider
```

### Phase 6. Plugin화

목표: 도메인이 바뀌어도 core harness는 유지한다.

구현 항목:

```text
Plugin interface
Crypto plugin
Macro plugin
Equity plugin
Domain-specific evaluator
Domain-specific operator pack
Domain-specific contract template
```

---

## 21. 설계 원칙

### 21.1 Orchestrator는 작게 유지한다

Orchestrator는 흐름을 조정하는 곳이지, 모든 로직을 넣는 곳이 아니다.

잘못된 구조:

```text
orchestrator 안에 mutation, validation, retry, logging, evaluation, archive logic이 모두 들어감
```

좋은 구조:

```text
orchestrator는 skill 호출과 state transition만 관리
세부 로직은 skill, operator, evaluator, hook, policy로 분리
```

### 21.2 LLM은 생성과 해석에 쓰고, 검증은 가능한 한 deterministic하게 한다

LLM이 잘하는 것:

```text
가설 생성
구조화 보조
semantic critique
failure summary
다음 탐색 방향 제안
report 작성
```

deterministic하게 해야 하는 것:

```text
schema validation
metric calculation
leakage rule check
duplicate hash check
complexity calculation
execution success check
```

### 21.3 Best score와 publish candidate를 구분한다

높은 점수는 publish의 필요조건일 수 있지만 충분조건은 아니다.

publish 전에는 다음을 통과해야 한다.

```text
robustness check
leakage check
ablation
novelty check
complexity check
human review
```

### 21.4 실패를 저장한다

성공 후보만 기억하면 같은 실패를 반복한다.

실패 memory는 mutation context와 selection policy에 다시 들어가야 한다.

### 21.5 탐색 분포를 관리한다

리서치 하네스의 목적은 단일 최고 후보를 찾는 것만이 아니다.

다음이 중요하다.

```text
좋은 후보군 유지
다양한 search basin 유지
실패 영역 회피
유망한 operator 강화
새로운 영역 exploration
```

---

## 22. 최종 요약

Hypoevolve를 하네스 엔진으로 만든다는 것은 다음을 의미한다.

```text
1. 연구 문제를 ResearchContract로 고정한다.
2. 자연어 가설을 ELG라는 IR로 변환한다.
3. Operator Registry를 통해 후보를 변형한다.
4. Evaluator Stack으로 후보를 의심하고 검증한다.
5. Archive와 Failure Memory에 성공과 실패를 모두 저장한다.
6. Selection Policy가 다음 탐색 분포를 조정한다.
7. Hook이 runtime 중간중간 guardrail, retry, audit, logging을 담당한다.
8. Lineage와 Observability로 왜 좋은 결과가 나왔는지 추적한다.
9. Plugin 구조로 도메인별 skill/operator/evaluator를 확장한다.
```

한 문장으로 정리하면 다음과 같다.

> Hypoevolve Research Harness는 LLM이 후보를 잘 만들게 하는 시스템이 아니라, 후보 생성·의심·검증·기억·재탐색을 모두 재현 가능하고 확장 가능한 소프트웨어 루프로 만드는 시스템이다.
