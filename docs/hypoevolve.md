# HypoEvolve 시스템 이해 문서

이 문서는 **향후 대규모 리팩토링/재구현의 기준점**으로 작성한 현재 HypoEvolve 시스템 해부 문서다.

목표는 단순 코드 요약이 아니다.

- **사용자 관점에서는 아무것도 달라지지 않게**
- **내부 구조는 엔터프라이즈급으로 재설계할 수 있게**
- 현재 시스템의 **동작 계약(contract)**, **암묵적 제약**, **숨은 결합점**, **테스트가 고정하는 사실들**, **리팩토링 시 절대 깨지면 안 되는 것들**을 정리하는 것이 목적이다.

---

## 1. 한 줄 요약

HypoEvolve는 **자연어 가설을 ELG(Executable Logic Graph)라는 구조적 중간표현으로 변환하고**,
LLM이 **구조적으로 근접한 새 가설을 제안**하고,
또 다른 LLM이 **평가용 Python 코드를 생성/실행**해 점수를 매긴 뒤,
그 결과를 **MAP-Elites 스타일 아카이브 + 런 아티팩트 + 리포트**로 축적하는 시스템이다.

즉 이 프로젝트의 본질은:

> **“코드 진화”가 아니라 “가설 진화”를 하는 LLM-오케스트레이션 실험 시스템**

이다.

---

## 2. 내가 현재 이해한 시스템의 핵심 목적

현재 시스템은 다음 문제를 풀려고 한다.

1. 사용자가 자연어로 가설을 준다.
2. 그 가설을 LLM이 **ELG 구조**로 바꾼다.
3. 그 ELG를 더 **측정 가능(measurable)** 하게 다시 쓴다.
4. 현재 가설의 점수와 최근 이력, 상위 가설들을 참고해 LLM이 **다음 child hypothesis** 를 만든다.
5. 또 다른 LLM이 그 가설을 평가하는 **Python evaluator code** 를 생성한다.
6. 실제 데이터셋에 대해 그 코드를 실행해서 `combined_score`, `precision`, `coverage`, `uplift` 등을 얻는다.
7. 그 결과를 아카이브에 넣고, 최고 가설과 전체 진행 상황을 저장한다.

즉 HypoEvolve는 다음 세 층이 결합된 시스템이다.

- **표현 계층**: ELG
- **탐색 계층**: mutation steering + archive
- **평가 계층**: LLM-generated evaluator code + subprocess execution

---

## 2.5 이번에 만든 문서 세트의 역할 분담

재구현 전에 문서가 여러 개로 늘어났기 때문에, 각각의 역할을 명확히 해두는 것이 좋다.

- `docs/hypoevolve.md`
  - 현재 시스템을 **이해하기 위한 기준 문서**
  - “지금 무엇이 어떻게 동작하는가”에 집중
- `.omx/plans/prd-behavior-preserving-reimplementation.md`
  - 재구현의 **제품/아키텍처 목표 문서**
  - “무엇을 어떤 원칙으로 바꿀 것인가”에 집중
- `.omx/plans/test-spec-behavior-preserving-reimplementation.md`
  - 재구현의 **검증 전략 문서**
  - “같은 행동임을 어떻게 증명할 것인가”에 집중
- `.omx/plans/verified-vs-inferred-ledger.md`
  - 현재 이해 중 **검증된 사실과 해석을 분리하는 문서**
- `.omx/plans/artifact-consumer-inventory.md`
  - 어떤 persisted artifact가 실제로 어디서 소비되는지 정리한 문서
- `.omx/plans/phase-0-behavior-freeze-task-list.md`
  - 가장 먼저 해야 할 **실행 작업 목록**
  - “무엇부터 잠글 것인가”에 집중
- `.omx/plans/first-patch-set-behavior-freeze.md`
  - 바로 착수 가능한 **첫 패치 제안서**
  - “지금 당장 어떤 안전한 변경을 넣을 것인가”에 집중

권장 읽기 순서:

1. `docs/hypoevolve.md`
2. `prd-behavior-preserving-reimplementation.md`
3. `test-spec-behavior-preserving-reimplementation.md`
4. `verified-vs-inferred-ledger.md`
5. `artifact-consumer-inventory.md`
6. `phase-0-behavior-freeze-task-list.md`
7. `first-patch-set-behavior-freeze.md`

---

## 3. 사용자 관점에서 절대 유지해야 하는 외부 계약

대규모 재구현을 하더라도 아래는 깨지면 안 된다.

### 3.1 CLI 계약

현재 CLI 엔트리포인트는 `hypoevolve` 이고, 주요 명령은:

- `hypoevolve run [hypothesis]`
- `hypoevolve seed`
- `hypoevolve render`
- `hypoevolve inspect`
- `hypoevolve doctor`
- `hypoevolve runs latest`
- `hypoevolve runs status`
- `hypoevolve runs report`
- `hypoevolve status`
- `hypoevolve report`

테스트가 고정하는 사용자 경험:

- bare `hypoevolve` 는 **도움말을 출력하고 exit code 1** 로 종료한다.
- `--help` 에는 배너, quick start, commands 가 나온다.
- `run` 은 summary / metric highlights / initial hypothesis / best hypothesis 를 출력한다.
- `run` 에 seed hypothesis 를 직접 안 주면 **자동 seed 생성 경로** 로 간다.
- `render --tree` 는 ASCII tree 를 출력한다.
- `runs latest/status/report` 는 persisted artifact 기준으로 작동한다.

### 3.2 출력 아티팩트 계약

한 번의 실행은 기본적으로 아래 구조를 만든다.

```text
.hypoevolve/runs/<run-id>/
  trace.jsonl
  checkpoint.json
  best.json
  run_summary.json
  score_history.json
  hypoevolve.log
  artifacts/
    seed.json
    iteration_0001.json
    ...
    top_evaluators.json
    top_evaluators/
      rank_.._candidate.py
      rank_.._wrapper.py
      rank_.._evaluation.json
  report/
    report.md
    assets/
      score_progression.svg
      seed_vs_best_metrics.svg
      archive_distribution.svg
```

즉 재구현 후에도 적어도 **동일한 의미의 파일 시스템 계약**은 유지하는 것이 안전하다.

최소 payload 의미도 사실상 계약이다.

#### `best.json`
- 현재 best hypothesis
- 해당 metrics

#### `checkpoint.json`
- 마지막 iteration 번호
- archive snapshot
- current best hypothesis / best metrics

#### `run_summary.json`
- requested iteration 수
- worker 설정
- duplicate skip 집계
- best score / best fingerprint / best hypothesis NL

#### `score_history.json`
- iteration별 상태
- score 계열 값
- best update 여부
- mutation summary / rationale 일부

#### `artifacts/top_evaluators.json`
- 상위 evaluator candidate code/wrapper/metadata manifest

즉 재구현 후에도 단순히 파일명만 맞추는 것이 아니라,

> **“후속 명령과 분석 도구가 기대하는 의미를 가진 payload shape”**

를 유지해야 한다.

### 3.3 점수 의미 계약

시스템은 현재 다음 평가 개념을 중심으로 동작한다.

- `precision = P(target | condition)`
- `baseline = P(target)`
- `coverage = P(condition)`
- `uplift = precision - baseline`
- `combined_score = uplift * coverage`

이 scoring 의미는 프롬프트, report, archive, tests, docs 전반에 박혀 있다.

### 3.4 “겉보기 동일성”에 포함되는 것

사용자가 체감하는 동일성은 단순 CLI 명령만이 아니다.

- 같은 prompt 구조
- 같은 ELG schema
- 같은 artifact 형태
- 같은 score 의미
- 같은 duplicate skip semantics
- 같은 best/report 계산
- 같은 seed 생성 경로
- 같은 worker on/off 동작
- prompt 제약에 의해 유도되는 **LLM 출력 행동**

까지 포함된다.

즉 재구현 대상은 “코드”가 아니라 사실상 **행동 전체**다.

---

## 4. 현재 아키텍처의 큰 그림

```text
User/CLI
  -> hypoevolve.cli
    -> hypoevolve.config
    -> hypoevolve.controller
      -> hypoevolve.parser
        -> hypoevolve.llm
        -> prompts/parser, prompts/measurable, prompts/nl
      -> hypoevolve.mutation
        -> hypoevolve.llm
        -> prompts/steering*
      -> hypoevolve.evaluator
        -> hypoevolve.llm
        -> prompts/evaluator*
        -> hypoevolve.executor
        -> hypoevolve.dataset
      -> hypoevolve.archive
      -> hypoevolve.artifacts
      -> hypoevolve.runtime
      -> hypoevolve.reporting
      -> hypoevolve.workers

Core IR
  -> elg/*

Optional seed generation path
  -> hypoevolve.hypo/*
```

이 구조에서 진짜 중심은 `HypoEvolveController.run()` 이다.

---

## 5. 코드베이스 디렉토리별 역할

## 5.1 `elg/`

이 프로젝트의 **가설 중간표현(IR)** 핵심이다.

주요 역할:

- `ir.py`: ELG 노드 타입 정의
- `codec.py`: dict/json 직렬화
- `normalize.py`: canonicalization
- `metrics.py`: 구조 metrics + fingerprint
- `render.py`: pretty / tree render
- `mutate.py`: immutable tree edit primitive

이 디렉토리는 향후 재구현에서도 **가장 안정적인 domain core** 로 볼 수 있다.

## 5.2 `hypoevolve/`

애플리케이션 계층이다.

주요 역할:

- config loading
- CLI
- controller orchestration
- parser / mutation steering / evaluator orchestration
- dataset schema access
- worker process execution
- artifacts / runtime persistence / report generation

즉 `elg/` 가 domain core 라면, `hypoevolve/` 는 application/runtime shell 이다.

## 5.3 `hypoevolve/hypo/`

이건 현재 메인 ELG 파이프라인과는 약간 다른 **legacy/보조 subsystem** 으로 봐야 한다.

역할:

- feature tree 랜덤 생성
- tree pair 를 보고 자연어 hypothesis seed 생성

현재 실제 사용 지점:

- `hypoevolve run` 에 hypothesis 가 없을 때 seed 자동 생성
- `hypoevolve seed`

즉 이 서브시스템은 **메인 ELG mutation loop의 일부가 아니라 seed 공급기** 에 가깝다.

## 5.4 `prompts/`

시스템 성능과 행동을 사실상 정의하는 **행동 계약 레이어**다.

중요 prompt:

- `parser/system.md`
- `measurable/system.md`
- `nl/system.md`
- `steering/system.md`
- `steering-random/system.md`
- `evaluator/system.md`
- 각 user prompt 들

이 프로젝트는 prompt 가 단순 리소스가 아니라 **실질적 business logic** 의 일부다.

## 5.5 `tests/`

가장 중요한 “진짜 사양” 중 하나다.

특히 다음을 고정한다.

- CLI UX
- artifact shape
- archive semantics
- parser normalization behavior
- steering prompt 제약
- evaluator retry / repair 흐름
- worker skip semantics

향후 재구현에서 **문서보다 tests 를 더 강한 진실** 로 봐야 한다.

---

## 6. 현재 end-to-end 실행 흐름

## 6.1 `run` 명령 진입

`hypoevolve.cli.run()`

순서:

1. config 로드
2. logger 설정
3. `--workers` override 반영
4. `HypoEvolveController.run(hypothesis)` 호출
5. 결과를 사람이 읽기 좋은 형태로 출력

## 6.2 seed 결정

`HypoEvolveController._resolve_seed_input_text()`

- 사용자가 hypothesis text 를 줬으면 그대로 사용
- 없으면 `generate_random_tree_pair_hypothesis()` 로 자동 생성

즉 시스템은 항상 **자연어 hypothesis string** 에서 시작한다.

## 6.3 run directory 생성

`create_run_dir()`

- 기본 위치: `.hypoevolve/runs/<8-char-id>/`
- `artifacts/` 서브디렉토리를 미리 만든다.

## 6.4 자연어 → ELG parse

`parse_hypothesis_text() -> llm_parse_hypothesis()`

- parser prompt 사용
- LLM 이 ELG root JSON 생성
- payload 검증
- `hypothesis_from_dict(...)`
- `normalize_hypothesis(...)`

## 6.5 measurable rewrite

`llm_make_hypothesis_measurable()`

- measurable prompt 사용
- 원 hypothesis 를 더 측정 가능하게 rewriting
- structure 최대한 유지
- parameter slot (`{HORIZON}`, `{NEG_Z_THRESHOLD}` 등) 사용 강제

## 6.6 seed evaluation

`Evaluator.evaluate(...)`

- 기본 evaluator 는 `LLMEvaluator`
- evaluator LLM 이 Python code 생성
- subprocess 로 실행
- stdout JSON 을 metrics 로 받음

참고:
- 과거 Python helper `evaluate_hypothesis(hypothesis, evaluator)` 는 제거되었고,
- 현재 호출 surface 는 `evaluator.evaluate(hypothesis)` 이다.

## 6.7 archive 초기화

`MAPElitesArchive`

- coverage bin
- complexity bin
- per-cell top-k
- parent sampling mode

seed 가 iteration 0 으로 archive 에 들어간다.

## 6.8 iteration loop

solo mode:

1. archive 에서 parent sample
2. `steer_mutation(...)`
3. child fingerprint 확인
4. duplicate 면 skip
5. 아니면 evaluate
6. archive 반영
7. artifact 기록

worker mode:

1. leader 가 parent sampling / task assembling
2. worker process 가 mutation + evaluation
3. leader 가 결과 merge
4. duplicate / steering error / best update 처리

## 6.9 finalize

마지막에:

- `score_history.json`
- `run_summary.json`
- top evaluator artifacts
- markdown report + SVG assets

를 생성한다.

---

## 7. ELG 표현 계층의 정확한 이해

현재 ELG 는 매우 작고 엄격한 schema 를 갖는다.

### 7.1 node 종류

- `AtomicNode`
- `LogicalNode`
- `RelationNode`

### 7.2 logical operator

코드 validator 차원에서는:

- `AND`
- `OR`
- `NOT`

을 지원한다.

하지만 **현재 parser / steering prompt 는 사실상 `AND`, `NOT` 중심으로 유도** 하고 있다.
즉 코드 레벨 허용 범위와 prompt 레벨 유도 범위가 완전히 동일하지는 않다.

### 7.3 relation type

- `IMPLIES`
- `SUPPORT`
- `CONTRADICT`
- `CORRELATE`

### 7.4 핵심 불변조건

- atomic 은 leaf 여야 한다.
- NOT 은 input 1개
- AND/OR 는 input 2개 이상
- relation 은 input 2개 정확히
- hypothesis root 는 사실상 relation root 가 기대된다.

테스트와 prompts 를 보면 실제 운영상의 기본 가정은:

> **“가설은 condition side 와 target side 를 가지는 relation-level proposition”**

이다.

즉 ELG 자체는 더 일반적이지만, 시스템 전체는 relation-root hypothesis 를 중심으로 설계돼 있다.

### 7.5 normalization 이 중요한 이유

`normalize_hypothesis()` 는:

- nested same-op flatten
- duplicate removal
- stable ordering
- double NOT collapse

를 수행한다.

이 때문에 fingerprint 는 **표현 차이보다 의미에 가까운 구조 차이** 를 추적한다.

즉 duplicate detection 은 raw text 가 아니라 **normalized structure** 기준이다.

이건 절대 보존해야 하는 핵심 계약이다.

---

## 8. Archive 계층의 정확한 의미

현재 archive 는 이름 그대로 완전한 MAP-Elites 구현은 아니고,

> **“MAP-Elites 느낌의 descriptor-binned top-k archive”**

에 가깝다.

### 8.1 descriptor 축

- coverage
- complexity = `count_nodes(hypothesis)`

### 8.2 binning 규칙

- coverage: `bisect_right`
- complexity: `bisect_left`

즉 경계 처리 방식이 서로 다르다.

예:

- coverage `0.05` 는 다음 bin 으로 간다.
- complexity `3` 은 현재 bin 에 남는다.

이 경계 의미는 유지해야 한다.

### 8.3 per-cell top-k

각 cell 마다 상위 `k` 개 elite 를 유지한다.

### 8.4 fingerprint dedup

같은 cell 내부에서 동일 fingerprint 가 다시 들어오면:

- 기존 점수가 더 높으면 유지
- 새 점수가 더 높으면 교체

### 8.5 parent sampling

두 모드가 있다.

- `random`: 전체 retained entry 중 uniform
- `map_elites_ucb`:
  - occupied cell 을 uniform 하게 고른 뒤
  - 그 cell 안에서 UCB 로 parent 선택

즉 UCB 가 global archive 전체가 아니라 **cell 내부** 에만 걸린다.

### 8.6 reward 의미

`record_parent_outcome()` 에 들어가는 reward 는 대체로:

- `child_score - parent_score`

즉 “그 parent 를 뽑았을 때 개선이 있었는가” 를 추적한다.

### 8.7 중요한 숨은 사실

`len(archive)` 는 **entries 수가 아니라 occupied cells 수** 다.

따라서 현재 `archive_size` 라는 이름으로 저장/출력되는 값은 실제로는
“retained hypothesis 총수”가 아니라 **occupied cell count** 에 더 가깝다.

이건 재구현 시 함부로 의미를 바꾸면 안 되는 부분이다.
이름이 어색해도 **현재 행동** 은 보존해야 한다.

---

## 9. Parser / measurable rewrite 의 실제 의미

## 9.1 parser 는 단순 문법 파서가 아니다

현재 parser 는 deterministic parser 가 아니라:

- LLM 호출
- JSON schema validation
- ELG normalization

으로 구성된 **semantic parser** 다.

즉 parser 성능은 코드보다 prompt 품질과 모델 행동에 크게 의존한다.

## 9.2 measurable 단계는 매우 중요하다

이 단계는 가설을 evaluator-friendly 한 형태로 강제한다.

주요 규칙:

- threshold/window/horizon 은 숫자 literal 대신 parameter slot 으로 표현
- predictive hypothesis 면 condition 은 `@t`, target 은 `@t+{HORIZON}`
- 비-boolean measurable atomic 은 z-score scale 선호
- structure 는 최대한 보존

즉 measurable rewrite 는 현재 시스템에서

> **“자연어 가설을 평가 가능한 operational form 으로 바꾸는 핵심 정규화 단계”**

다.

이 단계가 흔들리면 evaluator prompt, generated code, score 비교 모두 흔들린다.

---

## 10. Mutation steering 의 실제 의미

현재 mutation 은 `elg/mutate.py` 의 primitive 를 직접 사용하는 방식이 아니다.

실제로는:

- parent hypothesis
- current metrics
- recent history
- top hypotheses

를 prompt 로 넘겨서,

> **LLM 이 full child ELG root 전체를 직접 생성**

한다.

즉 현재 mutation 은 “primitive execution” 이 아니라 **prompt-constrained whole-child generation** 이다.

### 10.1 guided mode

요구 출력:

- `domain_reason`
- `score_reason`
- `operation_score_rankings`
- `child_hypothesis`
- `mutation_summary`

### 10.2 random steering mode

요구 출력:

- `child_hypothesis`
- `mutation_summary`

즉 random mode 는 rationale 을 요구하지 않는다.

### 10.3 prompt 가 강제하는 핵심 제약

테스트가 특히 고정하는 제약:

- target/conclusion side 에 `remove_atomic` 금지
- target side 는 반드시 남아야 함
- target atomic 이 condition side atomic 과 동일하면 안 됨
- relation root / 양변 proposition 구조를 유지해야 함

즉 이 시스템은 child 생성 자체를 LLM 에 맡기지만,
prompt 로 **“로컬 mutation처럼 보이는 full rewrite”** 를 강제하고 있다.

### 10.4 중요한 해석

재구현 시 naive 하게 “primitive mutation engine” 으로 바꾸면
표면상 비슷해 보여도 실제 LLM behavior distribution 이 달라질 수 있다.

즉 이 부분은 단순 리팩토링 대상이 아니라 **행동 재현 대상** 이다.

---

## 11. Evaluator 계층의 정확한 이해

이 시스템에서 evaluator 는 매우 특이하다.

### 11.1 evaluator 가 하는 일

1. hypothesis + dataset schema 를 prompt 로 준다.
2. LLM 이 `evaluate_hypothesis(accessor, parameters=None)` 함수를 생성한다.
3. wrapper script 가 schema / accessor / parameters 를 준비한다.
4. subprocess 에서 candidate code 를 실행한다.
5. stdout JSON 을 metrics 로 읽는다.

즉 evaluator 는 사실상:

> **“LLM에게 hypothesis-specific scoring program을 쓰게 하는 meta-evaluator”**

다.

### 11.2 required metric contract

핵심 키:

- `combined_score`
- `precision`
- `baseline`
- `coverage`
- `uplift`
- `support_count`
- `total_count`
- `rationale`
- `used_parameters`

### 11.3 evaluator prompt 가 강제하는 것

- pandas 외 third-party 금지
- lazy import
- exact column names만 사용
- 만들어내지 않은 컬럼 참조 금지
- `parameters.get(..., default)` 패턴 사용
- JSON-serializable 반환
- NaN/inf 반환 금지
- 미래 누수(leakage) 금지
- target horizon double-shift 금지

즉 evaluator prompt 도 사실상 런타임 규격서다.

### 11.4 repair loop

실패하면:

- failure message
- previous candidate code
- specific repair requirements

를 넣어 다시 LLM 에 요청한다.

이 retry/repair loop 는 중요한 품질 계약이다.

### 11.5 sanitation

실행 성공 후에도 payload 를 그대로 믿지 않는다.

- 누락 키는 default 로 채움
- non-finite 값은 0 / 0.0 으로 sanitize
- rationale 에 `non_finite_metrics_sanitized` prefix 부여

즉 evaluator 는 LLM output + code execution + post-sanitize 의 3단 구조다.

### 11.6 보안/운영 관점에서의 의미

현재 generated code 는 local subprocess 에서 실행된다.

이건 곧:

- generated code sandbox 가 매우 약함
- filesystem / CPU / memory / import / side effect 위험이 있음
- enterprise-grade 재구현에서는 별도 sandbox/execution boundary 가 매우 중요함

을 뜻한다.

이건 리팩토링이 아니라 **재설계 포인트** 다.

---

## 12. Worker 병렬 실행의 정확한 의미

worker mode 는 leader 가 archive 를 공유하는 진짜 분산 탐색이 아니라,

> **leader-owned archive + worker-side mutation/evaluation**

구조다.

### 12.1 leader 책임

- parent sampling
- task 구성
- known fingerprint 관리
- archive 반영
- artifact 기록

### 12.2 worker 책임

- steer_mutation
- duplicate fingerprint 사전 확인
- evaluation
- WorkerResult 반환

### 12.3 중요한 의미

archive state 의 single source of truth 는 leader 다.
worker 는 archive 를 직접 갱신하지 않는다.

이 구조는 좋은데, 현재는 각 worker task 마다:

- LLMClient 생성
- dataset schema 로드
- evaluator 생성

을 반복한다.

즉 성능/구조 측면에서 재구현 여지가 크다.

### 12.4 테스트가 고정하는 worker semantics

- steering error 는 fatal 이 아니라 skip
- duplicate child 는 evaluation 전에 skip
- worker mode 켜져도 artifact/report 는 동일하게 생성

---

## 13. Seed generation subsystem (`hypoevolve.hypo`) 의 위치

이 부분은 메인 ELG 시스템과 철학이 다소 다르다.

### 13.1 내부 모델

`hypoevolve.hypo` 는:

- DATA leaf
- transform/operator 노드
- 랜덤 트리 generator

를 사용해 feature tree 를 만든다.

그리고 `prompts/hypo/*` 를 사용해:

- 두 feature tree 간의 관계를 설명하는 자연어 hypothesis

를 만든다.

### 13.2 현재 역할

이 subsystem 은 **seed 문장 생성기** 역할이다.

즉:

- ELG core 의 일부 아님
- mutation loop 의 일부 아님
- hypothesis search 의 bootstrapper

에 가깝다.

### 13.3 재구현 시 판단

이건 메인 런타임과 분리된 bounded context 로 다루는 것이 좋다.

---

## 14. Dataset / config / YAML 계층

## 14.1 dataset schema

`dataset.yaml` 은 다음을 정의한다.

- description
- index name/dtype
- files (entity -> parquet path)
- columns (name + description)

이 schema 는:

- evaluator prompt
- DatasetAccessor
- seed tree DATA node label 공급

에 모두 영향을 준다.

즉 단순 config 가 아니라 **도메인 계약 파일** 이다.

## 14.2 DatasetAccessor

역할:

- entity 목록
- column 목록/설명
- parquet dataframe load
- summary

중요 제약:

- `.parquet` 만 지원
- pandas import 실패 시 에러

## 14.3 simple_yaml

현재 YAML 파서는 full YAML parser 가 아니라 **작은 subset parser** 다.

지원 범위는 제한적이다.

- indentation 기반 dict/list
- basic scalar
- comment skip

즉 향후 PyYAML/ruamel 등으로 갈아타더라도 **현재 읽히는 파일의 행동 호환성** 을 검증해야 한다.

---

## 15. Prompt 시스템은 사실상 코드다

이 프로젝트에서 prompt 는 부가 자료가 아니라 **실행 로직의 일부** 다.

그 이유:

1. parser behavior 를 결정한다.
2. measurable rewrite behavior 를 결정한다.
3. mutation locality 제약을 결정한다.
4. evaluator code shape 를 결정한다.
5. natural-language rendering fidelity 를 결정한다.

즉 향후 재구현에서도 prompt 는:

- source-controlled
- versioned
- contract-tested
- domain-reviewed

되어야 한다.

현재 tests 는 실제로 prompt 내용까지 검증한다.

예:

- evaluator prompt 에 pandas-only 규칙이 있는지
- steering prompt 에 conclusion-side mutation 제약이 있는지
- prompt 예시가 domain-neutral 한지

이건 매우 중요한 사양이다.

---

## 16. Runtime artifact / reporting 계층의 정확한 역할

`RunArtifactRecorder` 는 controller 로직을 단순화하기 위해 존재하지만,
실제로는 시스템의 **감사(audit) 레이어** 다.

### 16.1 저장하는 것

- trace
- best
- checkpoint
- per-iteration artifact
- score history
- run summary
- top evaluator code

### 16.2 의도적으로 저장하지 않는 것

테스트가 고정하는 사실:

- child natural language (`hypothesis_nl`) 는 archive metadata 에 저장하지 않음
- worker task 에 parent hypothesis natural language 를 실어 보내지 않음

이건 아마도:

- metadata 부피 감소
- worker payload 단순화
- archive purity 유지

를 위한 현재 설계 의도로 보인다.

### 16.3 report 의 의미

report 는 단순 요약이 아니다.

- best hypothesis
- structure summary
- score progression
- archive distribution
- top entries

를 제공한다.

즉 재구현 후에도 report 는 **persistent product surface** 로 취급하는 것이 맞다.

---

## 17. 현재 시스템의 중요한 암묵적 제약 / 숨은 결합점

아래는 코드만 대충 보면 놓치기 쉬운 부분들이다.

### 17.1 relation-root assumption

ELG 는 일반 구조를 허용하지만, system-level behavior 는 거의 항상 relation-root hypothesis 를 가정한다.

### 17.2 parser/measurable/steering/evaluator 간 강한 prompt coupling

이 네 단계는 느슨하게 연결된 게 아니라 매우 강하게 묶여 있다.

- parser 가 만든 구조
- measurable 가 만든 parameter slot / time notation
- steering 이 보존하려는 relation/condition/target semantics
- evaluator 가 기대하는 measurable syntax

가 한 체인이다.

한 부분만 바꾸면 전체 distribution 이 흔들린다.

### 17.3 duplicate detection 은 normalized structure 기준

raw JSON/text equality 가 아니다.

### 17.4 report/status 는 persisted file 존재 여부에 의존

예:

- `run_summary.json` 있으면 completed
- 없고 checkpoint 있으면 running

즉 runtime state 는 DB 가 아니라 **파일 존재 + JSON payload** 기준이다.

### 17.5 CLI example 와 실제 runtime 사이의 작은 불일치 가능성

예를 들어 README/CLI example 은 `.hypoevolve/runs/latest/...` 예시를 보여주지만,
현재 코드에는 `latest` symlink/materialization 로직이 없다.
대신 `runs latest` 명령이 최신 run directory 를 찾는다.

즉 문서/예시/코드 간 미세한 불일치도 존재한다.

### 17.6 outdated docs 존재 가능성

`docs/llm-integration-points.md` 같은 보조 문서는 주기적으로 현재 구현에 맞춰 갱신해야 한다.
과거에는 placeholder/fallback 중심 설명이 섞여 있었지만, 현재는 주요 parser/evaluator/steering 경로 설명을 최신 구현에 맞추는 방향으로 정리되고 있다.

그래도 우선순위는 여전히:

> **현행 사양은 문서보다 코드+테스트 쪽이 더 강한 진실이다.**

### 17.7 “현재 존재하는 결함”과 “보존해야 할 동작”은 다르다

아주 중요하다.

지금 시스템에 관측되는 모든 현상을 재구현 후에도 그대로 유지해야 하는 것은 아니다.

예를 들어 테스트 실행 중 관찰되는 다음과 같은 현상:

- logger sink lifecycle 문제로 인한 closed-stream logging noise

같은 것은 **현재 구현상의 결함/운영상 잡음** 으로 보는 편이 맞고,
사용자 가치가 있는 product contract 로 취급하면 안 된다.

즉 재구현 시에는 아래 둘을 분리해서 다뤄야 한다.

1. **반드시 유지해야 하는 계약**
   - CLI behavior
   - artifact shape
   - ELG semantics
   - score semantics
   - prompt-driven behavioral constraints
2. **현존 결함이지만 굳이 보존할 필요는 없는 것**
   - 내부 noisy logging
   - 우연한 temp path 표현
   - 테스트 환경 특유 부수 출력

이 구분이 없으면 “동등성”을 잘못 정의하게 된다.

### 17.8 Verified vs Inferred Ledger 가 필요하다

현재 이 문서는 상당 부분 코드/테스트 기반으로 작성됐지만,
일부는 여전히 **작성자의 해석(inference)** 이 섞여 있다.

재구현 착수 전에 아래 같은 ledger 를 별도 표로 유지하는 것이 좋다.

| Claim | Type | Primary anchor | Secondary anchor | Migration risk if wrong | Re-verify before Phase 1 |
|---|---|---|---|---|---|
| seed bootstrap 순서는 parse -> measurable -> evaluate -> archive insert 이다 | verified-by-code | `hypoevolve/controller.py` | `tests/test_hypoevolve_controller.py` | 높음 | fixture-backed run 재검증 |
| measurable rewrite 는 현재 사실상 필수 단계다 | inferred + partially verified | `hypoevolve/controller.py` | prompt/evaluator 계약 | 높음 | measurable step bypass 실험 금지 여부 확인 |
| duplicate skip 시 iteration artifact는 쓰지 않는다 | verified-by-code | `hypoevolve/controller.py`, `hypoevolve/artifacts.py` | duplicate tests | 중간 | duplicate fixture 생성 |
| worker merge의 source of truth 는 leader archive 다 | verified-by-code | `hypoevolve/controller.py`, `hypoevolve/workers.py` | worker tests | 높음 | worker shadow test |
| report regeneration은 persisted artifacts만으로 가능하다 | verified-by-test | `hypoevolve/cli.py`, `hypoevolve/reporting.py` | CLI report tests | 중간 | persisted-run fixture로 재검증 |
| `hypoevolve.hypo` 는 seed 공급기 역할로만 분리 가능하다 | inferred | `hypoevolve/cli.py`, `hypoevolve/controller.py` | hypo tests | 중간~높음 | dependency map 재확인 |

이 표는 지금 당장 완벽할 필요는 없지만,
Phase 1 전에 최소한 위 항목들은 정리돼 있어야 한다.

현재 별도 산출물:

- `.omx/plans/verified-vs-inferred-ledger.md`

---

## 18. 현재 구조의 강점

대규모 재구현의 기반으로서 좋은 점도 분명하다.

### 18.1 ELG core 가 작고 명확하다

IR 가 매우 단순하고 설명 가능하다.

### 18.2 application concerns 가 대체로 모듈 분리되어 있다

- controller
- archive
- evaluator
- runtime/artifacts/reporting

가 이미 어느 정도 나뉘어 있다.

### 18.3 prompt contracts 가 테스트로 잠겨 있다

이건 아주 큰 장점이다.

### 18.4 artifact 중심 구조라 regression 비교가 쉽다

재구현 후에도 같은 run artifact 를 비교하면서 행동 등가성을 검증하기 쉽다.

### 18.5 worker merge ownership 이 leader 에 집중되어 있다

archive consistency 관점에서 유리하다.

---

## 19. 현재 구조의 약점 / 리팩토링 포인트

## 19.1 도메인 코어와 orchestration 이 아직 완전히 분리되지 않았다

controller 가 비교적 많은 책임을 가진다.

## 19.2 evaluator 보안 경계가 약하다

generated Python code 를 local subprocess 에서 직접 실행한다.

## 19.3 prompt-driven behavior 가 강하지만 prompt abstraction layer 는 얇다

현재 `prompts.py` 는 거의 file loader 수준이다.

## 19.4 legacy subsystem (`hypoevolve.hypo`) 과 ELG mainline 의 철학이 다르다

bounded context 를 더 명확히 쪼갤 필요가 있다.

## 19.5 일부 naming/meaning mismatch 존재

대표적으로 `archive_size` 가 실제 entry count 가 아니라 occupied cell count 에 가깝다.

## 19.6 config / dataset parsing 이 minimal custom parser 에 묶여 있다

운영성/에러 메시지/표준성 측면에서 한계가 있다.

## 19.7 worker task payload 에 사용되지 않는 정보가 일부 있다

예: `parser_retries` 는 현재 worker 실행 경로에서 사실상 쓰이지 않는다.

## 19.8 evaluator / parser / mutation retry semantics 가 모듈별로 분산되어 있다

재사용 가능한 failure policy abstraction 이 아직 없다.

---

## 20. 재구현 시 절대 보존해야 할 behavioral invariants

아래는 **사용자가 바뀐 걸 체감하지 않기 위해** 특히 중요하다.

1. **ELG schema**
2. **normalization + fingerprint semantics**
3. **score formula semantics**
4. **CLI command / option / exit-code behavior**
5. **artifact file names + payload shape**
6. **report generation outputs**
7. **prompt text가 강제하는 제약의 실질적 의미**
8. **duplicate skip semantics**
9. **worker skip/merge semantics**
10. **seed generation path when hypothesis is omitted**
11. **status/report commands reading persisted artifacts**
12. **measurable parameter-slot notation**
13. **condition@t / target@t+h temporal interpretation**

---

## 21. 재구현 시 추천되는 아키텍처 분해 방향

현재 코드를 깔끔하게 재구현하려면 아래 bounded context 로 나누는 것이 좋다.

### 21.1 Domain Core

- ELG node model
- normalization
- fingerprint
- structural metrics
- render
- archive descriptor logic

### 21.2 Prompted Semantics Layer

- parser contract
- measurable rewrite contract
- NL rendering contract
- mutation steering contract
- evaluator codegen contract

### 21.3 Search Engine Layer

- run session
- iteration state
- parent selection
- duplicate policy
- worker coordination

### 21.4 Evaluation Runtime Layer

- code generation
- execution sandbox
- retry/repair policy
- payload validation/sanitization

### 21.5 Persistence & Reporting Layer

- run directory management
- trace/checkpoint/best/history
- report generation
- top evaluator materialization

### 21.6 Delivery Surface Layer

- CLI
- config loading
- doctor/status/report UX

### 21.7 Seed Synthesis Layer

- feature-tree generator
- tree-pair hypothesis synthesis

즉 특히 `hypoevolve.hypo` 는 메인 search engine 과 분리된 별도 subdomain 으로 보는 것이 맞다.

---

## 22. 내가 생각하는 리팩토링/재구현 전략

사용자 체감 0 변화가 목표라면, 추천 전략은 다음과 같다.

### 22.1 1단계: 사양 고정

- 현재 tests 유지/확대
- artifact golden test 추가
- prompt golden snapshot 추가
- sample run 비교 harness 추가

### 22.2 2단계: 읽기 전용 구조화

- package boundary 재정리
- domain/application/infrastructure 분리
- 현재 public contracts 를 adapter 로 감싼다.

### 22.3 3단계: 내부 구현 치환

- controller 분해
- evaluator runtime 분리
- persistence layer 추상화
- worker orchestration 분리

### 22.4 4단계: 안전한 품질 상승

- sandbox 강화
- richer diagnostics
- deterministic replay / seed reproducibility 강화
- prompt/version metadata 관리

즉 처음부터 “새 시스템” 을 만드는 게 아니라,

> **동일한 public behavior를 가진 새 내부 구조를 단계적으로 끼워 넣는 방식**

이 맞다.

---

## 23. 앞으로 꼭 더 파악해야 할 것들

이번 문서화는 현재 소스+테스트 기준의 1차 해부다.
대규모 재구현 전에 아래는 추가로 더 조사해야 한다.

### 23.1 실제 sample run 비교

- 동일 config / 동일 prompt / 동일 model 로 기존 구현의 run artifact 수집
- 재구현 후보와 diff 비교

### 23.2 prompt-output distribution 분석

- parser
- measurable
- steering
- evaluator

각 단계에서 실제 모델이 어떤 분포의 출력을 내는지 파악해야 한다.

### 23.3 report consumers 확인

누가 `report.md`, `run_summary.json`, `score_history.json` 을 소비하는지 확인 필요.

### 23.4 generated evaluator code 위험 분석

- import 범위
- file/network side effect
- subprocess isolation 수준

### 23.5 `hypoevolve.hypo` 의 유지 필요성 판단

이 subsystem 을 계속 유지할지, seed 전용 standalone package 로 뺄지 결정 필요.

### 23.6 parity comparison normalization 규칙 확정

golden artifact 비교를 하려면 “무엇은 exact match, 무엇은 normalized compare, 무엇은 tolerance compare” 인지 먼저 정해야 한다.

예:

- exact:
  - ELG JSON shape
  - prompt sections
  - artifact key set
- normalized:
  - temp dir path
  - run id
  - timestamp
- tolerance:
  - live LLM을 쓸 경우 score/selection 분포 일부

이 규칙이 없으면 이후 테스트 스펙이 흔들린다.

### 23.7 report consumer 실사용 여부 확인

`status`, `report`, `runs status`, `runs report` 외에
사람이나 다른 스크립트가 `run_summary.json`, `score_history.json`, `report.md`, `top_evaluators.json`
을 실제로 소비하는지 확인이 필요하다.

이게 확인되면:

- 진짜 public contract
- 사실상 internal-but-stable contract

를 더 정확히 나눌 수 있다.

---

## 23.8 Phase-1 Entry Blockers

아래가 정리되지 않았다면 Phase 1 구현 착수는 막는 것이 맞다.

- [ ] golden fixture 6종 생성됨
- [ ] artifact normalization policy 문서화됨
- [ ] prompt snapshot 고정됨
- [ ] generated-seed path용 deterministic harness 확보됨
- [ ] legacy/new shadow diff 기준 정의됨
- [ ] report consumer 확인됨 또는 “없음”으로 명시됨
- [ ] Verified vs Inferred Ledger 최소 버전 작성됨

즉 “좋은 계획이 있음”과 “구현에 들어가도 됨”은 다른 상태다.

---

## Appendix A. Persisted Artifact Contract Summary

아래 표는 Phase 0 fixture 설계용 최소 계약 요약이다.

| file | required keys / sections | optional keys | volatile fields to normalize | consumer commands/tests | compatibility rule |
|---|---|---|---|---|---|
| `checkpoint.json` | `iteration`, `archive_size`, `archive` | `best_hypothesis`, `best_metrics` | 없음 또는 일부 path-free metadata | controller/report/status flows | key set + semantic meaning 유지 |
| `best.json` | `hypothesis`, `metrics` | 없음 | 없음 | inspect/report | exact schema 유지 |
| `run_summary.json` | `iterations_requested`, `best_score`, `archive_size`, `duplicate_skips_total` | worker/best metadata fields | 없음 | status/report | required summary keys 유지 |
| `score_history.json` | iteration records list | richer metadata fields | 없음 | report/status | status/value semantics 유지 |
| `artifacts/top_evaluators.json` | manifest list, per-entry rank/fingerprint/score | candidate/wrapper/metadata paths | generated relative filenames 일부 | artifacts tests/report debugging | relative manifest semantics 유지 |
| `report/report.md` | executive summary, best ELG, search overview 류 섹션 | extra explanatory prose | run id/path 텍스트 일부 | report command/tests | section structure 유지 |
| `trace.jsonl` | event-per-line JSON | extra metadata fields | none if using stubbed fixture | trace/debugging | event shape compatibility 유지 |

이 표는 implementation spec 이라기보다,

> **golden fixture를 설계할 때 무엇을 exact compare 하고 무엇을 normalize 해야 하는지 정하는 출발점**

이다.

---

## Appendix B. Prompt Input Contract Summary

prompt parity는 raw prompt text만으로 충분하지 않다.
현재 코드 기준으로 최소한 아래 입력 규칙도 계약으로 봐야 한다.

### Steering prompt input contract

- parent hypothesis: measurable ELG pretty render
- current metrics: current parent metrics JSON
- recent history: **최근 2개만 사용**
- top hypotheses: **global archive 상위 3개**
- excluded metadata:
  - child/archive metadata 안의 `hypothesis_nl`
  - worker task 에서의 parent natural-language text

즉 mutation prompt parity는:

1. raw prompt file
2. variable names
3. included/excluded metadata
4. truncation window

를 함께 고정해야 한다.

### Evaluator prompt input contract

- hypothesis pretty render
- dataset description / index / entities / column specs
- accessor API 설명
- no sample dataframe payload

### Parser / measurable / NL contract

- parser: raw NL input -> ELG root JSON
- measurable: ELG root -> more measurable ELG root
- NL render: ELG root -> plain text sentence

즉 “prompt contract”는 단순히 문장 지시가 아니라,

> **무슨 데이터를 어떤 shape로 모델에 넘기느냐까지 포함한다**

고 봐야 한다.

---

## Appendix C. Run Lifecycle / CLI Persisted-Run Contract

현재 persisted run을 둘러싼 CLI 의미 계약은 다음과 같다.

### run directory creation

- `create_run_dir()` 가 run dir + `artifacts/` 를 만든다.

### `runs latest`

- 최신 run dir 선택 기준은 **이름순이 아니라 mtime 기준** 이다.

### `status` / `runs status`

- `run_summary.json` 존재 -> `completed`
- `checkpoint.json` 만 존재 -> `running`
- 둘 다 없으면 -> `failed`

### `report` / `runs report`

- `report/report.md` 가 없으면 **disk artifacts로부터 regenerate** 한다.

### worker fallback

- workers disabled 이거나 `worker_count == 1` 이면 single-process path 로 간다.

이 규칙들은 작아 보여도,
재구현 시 status/report UX와 regression 결과를 흔들 수 있는 핵심 휴리스틱이다.

---

## Appendix D. Artifact Record-Type Summary

특히 `score_history.json` 은 파일 하나지만,
status별로 record 의미가 다르다.

### `seed`
- iteration = 0
- evaluated row
- best_updated = true

### `evaluated`
- normal evaluated child
- score 계열 값 존재
- mutation metadata 일부 존재 가능

### `skipped_duplicate`
- **evaluated row 아님**
- duplicate skip 기록
- report에서 evaluated count 로 세면 안 된다

### `skipped_steering_error`
- **evaluated row 아님**
- steering failure 기록
- error text 포함 가능

중요:

> skip entry를 `score = 0` 으로 쓰면 안 된다.

현재 report 쪽은 사실상 `score is not None` 류 기준으로 evaluated history 를 해석하므로,
skip row를 zero-score evaluated row로 바꾸면 report/plot/count가 drift 한다.

### `trace.jsonl`

각 줄은 JSON event 이며, 최소한:

- iteration
- parent
- child
- metrics
- metadata

의 의미를 유지해야 한다.

---

## 24. 최종 결론

현재 HypoEvolve는 겉보기보다 단순한 프로젝트가 아니다.

이건 단순히:

- ELG library 하나
- LLM wrapper 하나
- CLI 하나

가 아니라,

> **“prompt가 정의하는 의미 계약” + “ELG 구조 계약” + “generated code evaluator” + “artifact/report runtime” 이 결합된 탐색 시스템**

이다.

따라서 앞으로의 대규모 리팩토링/재구현은 다음 원칙을 따라야 한다.

1. **사용자-visible contract 먼저 고정**
2. **prompt behavior 를 코드만큼 중요하게 취급**
3. **ELG/fingerprint/scoring/artifacts 를 핵심 불변량으로 간주**
4. **legacy seed subsystem 과 mainline search engine 을 분리 인식**
5. **보안/확장성/운영성은 내부에서 크게 개선하되 외부 행동은 보존**

이 문서는 앞으로의 재구현 작업에서

- 무엇을 그대로 지켜야 하는지
- 무엇을 내부적으로 바꿔도 되는지
- 어디가 핵심이고 어디가 부수적인지

를 판단하는 기준 문서로 사용하면 된다.
