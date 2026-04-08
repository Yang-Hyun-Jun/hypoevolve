# HypoEvolve System Methodology

## 1. Introduction

본 문서는 현재 `hypoevolve` 저장소 구현을 기준으로, HypoEvolve를 **자연어 가설을 구조화된 논리 표현으로 변환하고, 데이터 기반 점수를 최대화하도록 반복적으로 변이·평가하는 가설 진화 시스템**으로 기술한다. 문서의 목표는 단순 사용 설명을 넘어서, 시스템의 표현 체계, 탐색 알고리즘, 평가 프로토콜, 병렬 실행 방식, 그리고 런타임 산출물을 하나의 일관된 방법론으로 정식화하는 데 있다.

현 구현에서 HypoEvolve는 다음 세 요소의 결합으로 이해할 수 있다.

1. **ELG(Executable Logic Graph)**: 가설을 표현하는 구조적 중간표현
2. **LLM-guided proposal/evaluation**: 가설 파싱, measurable rewrite, mutation steering, evaluator code generation을 담당하는 생성 계층
3. **Archive-based search loop**: 구조 복잡도와 경험적 커버리지를 기준으로 후보를 유지·샘플링하는 탐색 계층

이 문서는 구현과 밀착된 기술 문서이므로, 모든 서술은 현재 코드베이스가 실제 수행하는 동작을 기준으로 한다.

### 1.1 System synopsis

| 항목 | 내용 |
| --- | --- |
| 입력 | 자연어 가설, 구성 파일, 데이터셋 스키마 |
| 핵심 표현 | ELG tree (`atomic`, `logical`, `relation`) |
| 탐색 목적 | 데이터셋 위에서 경험적 점수 $s(H;D)$ 를 최대화하는 가설 탐색 |
| proposal 계층 | LLM 기반 parser, measurable rewrite, mutation steering |
| evaluation 계층 | LLM이 생성한 evaluator code의 로컬 실행 |
| memory 계층 | coverage/complexity 기반 top-k archive |
| 선택 전략 | occupied cell 균등 샘플링 + UCB-style parent reuse |
| 산출물 | `trace.jsonl`, `checkpoint.json`, `best.json`, iteration artifacts |

### 1.2 Architecture sketch

```text
Natural-language hypothesis
        │
        ▼
  ELG parsing (LLM)
        │
        ▼
Measurable rewrite (LLM)
        │
        ▼
Evaluator code generation (LLM)
        │
        ▼
Local execution on dataset
        │
        ▼
Score / archive update / trace persistence
        │
        └───────────────► mutation steering (LLM) ───────────────┐
                                                                  │
                                                                  ▼
                                                           next child ELG
```

---

## 2. Problem Formulation

HypoEvolve의 입력은 자연어 가설 서술 $x$ 와 데이터셋 $D$ 이다. 시스템은 $x$ 를 구조적 가설 $H$ 로 변환한 뒤, 반복적인 제안-평가 루프를 통해 더 높은 점수를 갖는 가설 집합을 탐색한다.

형식적으로 시스템의 목적은 다음과 같이 쓸 수 있다.

$$
H^* = \arg\max_{H \in \mathcal{H}} s(H; D)
$$

여기서:

- $\mathcal{H}$ 는 ELG로 표현 가능한 가설 공간
- $s(H;D)$ 는 데이터셋 $D$ 에 대해 경험적으로 계산되는 평가 점수
- $H^*$ 는 현재 탐색 예산 아래에서 발견된 최고 점수 가설

중요한 점은 HypoEvolve가 직접 코드 공간을 탐색하지 않는다는 것이다. 대신 시스템은 **가설의 구조 자체**를 탐색하고, 평가 시점에만 LLM이 생성한 실행 코드를 통해 데이터에 대한 점수를 계산한다.

---

## 3. Hypothesis Representation: Executable Logic Graph (ELG)

### 3.1 ELG 타입 체계

HypoEvolve의 핵심 표현은 ELG이며, 이는 트리 형태의 불변 구조로 구현된다. 루트는 하나의 `Hypothesis` 객체이며, 내부 노드는 세 가지 범주로 나뉜다.

1. **Atomic node**
   - 단일 명제 또는 측정 가능한 사건을 표현한다.
   - 필드: `name`
2. **Logical node**
   - 자식 명제를 논리적으로 결합한다.
   - 연산자는 공통 필드 `name` 에 저장된다. (`AND`, `OR`, `NOT`)
3. **Relation node**
   - 조건 측과 목표 측 사이의 관계를 나타낸다.
   - 관계 타입도 공통 필드 `name` 에 저장된다. (`IMPLIES`, `SUPPORT`, `CONTRADICT`, `CORRELATE`)

루트 가설은 일반적으로 하나의 relation node이며, condition side와 target side를 각각 하나씩 가진다.

### 3.2 ELG 문법

현재 구현의 ELG 문법은 개략적으로 다음과 같다.

```text
Hypothesis := RelationNode | LogicalNode | AtomicNode

AtomicNode := {
  kind: "atomic",
  name: str
}

LogicalNode := {
  kind: "logical",
  name: "AND" | "OR" | "NOT",
  inputs: [Node, ...]
}

RelationNode := {
  kind: "relation",
  name: "IMPLIES" | "SUPPORT" | "CONTRADICT" | "CORRELATE",
  inputs: [condition_node, target_node]
}
```

### 3.3 Structural validity constraints

구현 수준의 구조 제약은 다음과 같다.

- `AtomicNode.name` 는 공백이 아닌 문자열이어야 한다.
- `LogicalNode.NOT` 는 정확히 하나의 입력을 가져야 한다.
- `LogicalNode.AND` 와 `LogicalNode.OR` 는 최소 두 개 이상의 입력을 가져야 한다.
- `RelationNode` 는 정확히 두 개의 입력을 가져야 한다.

이 제약은 타입 생성 시점과 parser validation 시점 모두에서 적용된다. 따라서 ELG는 단순한 JSON 직렬화 포맷이 아니라, **명시적 well-formedness 규칙을 갖는 실행 가능한 구조 언어**로 볼 수 있다.

### 3.4 Normalization and structural identity

ELG는 탐색 중 중복 후보가 자주 발생할 수 있기 때문에, HypoEvolve는 정규화(normalization)를 통해 구조적 동형성을 완화한다. 정규화 연산은 다음을 수행한다.

1. 재귀적으로 하위 노드를 정규화
2. 연속된 동일 logical operator를 평탄화(flatten)
3. 이중 부정 `NOT(NOT(x))` 제거
4. 동일 자식 제거(deduplication)
5. 안정된 정렬 키(JSON 직렬화 기반)를 이용한 자식 순서 정렬

정규화된 가설은 stable JSON 으로 직렬화되고 SHA-256 해시를 통해 fingerprint를 생성한다. 따라서 archive는 단순 텍스트 비교가 아니라 **정규화된 구조 해시**를 사용해 중복을 관리한다.

### 3.5 Structural complexity

현재 archive descriptor에서 복잡도는 단순하면서도 안정적인 정의를 사용한다.

$$
\mathrm{complexity}(H) = \text{count\_nodes}(H)
$$

즉 원자, 논리, 관계 노드를 모두 포함한 전체 노드 수가 구조 복잡도이다.

---

## 4. Dataset Model and Evaluation Interface

HypoEvolve는 evaluator를 데이터셋에 직접 결합하지 않고, `DatasetSchema` 와 `DatasetAccessor` 를 통해 느슨하게 연결한다.

### 4.1 Dataset schema

데이터셋 스키마는 다음 정보를 담는다.

- 데이터셋 설명(description)
- 인덱스 이름과 dtype
- 엔티티별 parquet 파일 경로
- 사용 가능한 컬럼과 각 컬럼 설명

현재 기본 실험 설정에서는 Binance perpetual futures 데이터가 사용되며, 엔티티는 `BTCUSDT`, `DOGEUSDT`, `XRPUSDT` 이다.

### 4.2 Dataset accessor abstraction

Evaluator가 사용할 수 있는 데이터 인터페이스는 제한적이고 명시적이다.

- `entities()`
- `file_map()`
- `column_names()`
- `column_descriptions()`
- `load_dataframe(entity)`
- `load_all_dataframes()`
- `head(entity, n=5)`
- `summary()`

이 추상화는 evaluator prompt에 그대로 삽입되어, LLM이 실제 사용 가능한 데이터 API 범위를 벗어나지 않도록 유도한다.

---

## 5. End-to-End Pipeline Overview

HypoEvolve의 단일 실행(run)은 자연어 시드 가설 하나를 입력으로 받아 다음 절차를 수행한다.

### Algorithm 1. Overall run loop

```text
Input:
  natural-language hypothesis x
  config c
  dataset D

1. create run directory R
2. configure logger
3. parse x into ELG hypothesis H0
4. rewrite H0 into measurable ELG H0'
5. render H0' back to natural language for metadata
6. evaluate H0' on D to obtain metrics m0
7. add (H0', m0) to archive A
8. persist seed trace / checkpoint / best artifact
9. for t = 1 .. T:
10.   sample parent Hp from archive A
11.   generate child Hc via mutation steering
12.   evaluate Hc to obtain metrics mc
13.   compute reward Δ = score(mc) - score(Hp)
14.   add (Hc, mc) to archive A
15.   update parent sampling statistics with reward Δ
16.   persist trace / checkpoint / best / iteration artifact
17. return best hypothesis in archive A
```

이 루프는 단일 프로세스(serial mode) 또는 다중 프로세스(worker mode)로 실행될 수 있다.

---

## 6. Stage I: Natural-Language Parsing and Measurable Rewrite

### 6.1 Natural-language to ELG parsing

입력 문장 $x$ 는 `llm_parse_hypothesis` 를 통해 ELG root JSON으로 변환된다. 이 단계의 핵심은 LLM을 단순 자유서술 생성기가 아니라 **구조화된 JSON 제안기**로 사용하는 것이다.

절차는 다음과 같다.

1. parser system prompt 로 초기 요청 수행
2. 응답을 JSON으로 파싱
3. ELG schema validation 수행
4. `Hypothesis` 객체로 변환
5. 정규화 수행
6. 실패 시 JSON retry prompt를 부착해 재시도

이 과정은 최대 `retries + 1` 회 수행된다.

### 6.2 Parser validation

Parser 출력은 단순 JSON 성공 여부만이 아니라 다음 구조 검증을 통과해야 한다.

- 유효한 `kind`
- 유효한 logical / relation `name`
- `inputs` 의 길이 제약
- atomic node의 non-empty `name`

즉 HypoEvolve에서 parser는 “자연어를 바로 해석한다”기보다, **타입이 보장된 ELG 객체로의 투영(projection)** 을 수행한다.

### 6.3 Measurable rewrite

초기 ELG는 여전히 추상적일 수 있으므로, HypoEvolve는 즉시 두 번째 LLM 단계를 수행한다. 이 단계는 기존 ELG를 입력으로 받아 **더 측정 가능(measurable)한 ELG**로 재작성한다.

이때의 설계 원칙은 다음과 같다.

- hypothesis structure는 가능한 한 유지
- 데이터에서 직접 평가 가능한 형태로 원자 명제를 구체화
- 출력 역시 동일한 ELG schema validation을 통과해야 함

따라서 실제 탐색의 출발점은 “raw parse result”가 아니라 “measurable rewrite result”라고 보는 편이 정확하다.

### 6.4 Natural-language rendering for metadata

시스템은 archive metadata와 trace 해석 가능성을 높이기 위해 measurable ELG를 다시 자연어로 렌더링하는 보조 단계를 둔다. 이 렌더링은 탐색 그 자체의 필수 단계는 아니지만, 다음 두 용도에 사용된다.

1. archive entry metadata 저장
2. 이후 mutation steering prompt에 parent hypothesis의 자연어 서술 제공

렌더링에 실패하면 시스템은 pretty-printed ELG 문자열로 폴백한다.

---

## 7. Stage II: Data-Driven Scoring via LLM-Generated Evaluator Code

### 7.1 Motivation

HypoEvolve는 evaluator를 고정된 수식 엔진 하나로 제한하지 않는다. 대신 measurable ELG와 데이터셋 컨텍스트를 읽고, LLM이 해당 가설을 평가할 Python 함수를 생성하도록 한다. 이는 강한 의미의 완전한 프로그램 합성이라기보다, **가설별 평가 코드를 생성하는 constrained code generation procedure** 로 이해하는 편이 정확하다.

### 7.2 Function contract

LLM이 생성하는 evaluator는 정확히 하나의 함수를 정의해야 한다.

```python
def evaluate_hypothesis(accessor, parameters: dict | None = None) -> dict:
    ...
```

이 함수는 다음 출력을 반드시 반환해야 한다.

- `combined_score`
- `precision`
- `baseline`
- `coverage`
- `uplift`
- `support_count`
- `total_count`
- `rationale`
- `used_parameters`

### 7.3 Scoring definition

현재 구현의 핵심 점수는 다음과 같이 정의된다.

- precision: $P(T \mid C)$
- baseline: $P(T)$
- coverage: $P(C)$
- uplift: $P(T \mid C) - P(T)$
- combined score:

$$
\mathrm{combined\_score} = \mathrm{uplift} \times \mathrm{coverage}
$$

이를 경험적 카운트로 쓰면,

$$
\mathrm{precision} = \frac{n(C \land T)}{n(C)}
$$

$$
\mathrm{baseline} = \frac{n(T)}{N}
$$

$$
\mathrm{coverage} = \frac{n(C)}{N}
$$

$$
\mathrm{uplift} = \mathrm{precision} - \mathrm{baseline}
$$

$$
\mathrm{combined\_score} = \left(\frac{n(C \land T)}{n(C)} - \frac{n(T)}{N}\right) \cdot \frac{n(C)}{N}
$$

여기서:

- $C$ 는 condition event
- $T$ 는 target event
- $n(C)$ 는 condition이 참인 시점 수
- $n(T)$ 는 target이 참인 시점 수
- $N$ 은 유효 평가 표본 수

이 스코어는 “조건이 baseline 대비 얼마나 target 확률을 끌어올리는가”와 “그 조건이 얼마나 자주 발생하는가”를 동시에 반영한다.

### 7.4 Prompt-grounded evaluator generation

Evaluator LLM prompt는 다음 정보로 구성된다.

- readable ELG hypothesis
- dataset description
- index metadata
- entity 목록
- column specification
- DatasetAccessor API documentation
- parameter-slot handling rule
- 시간 정렬 및 leakage 방지 규칙
- 출력 계약

특히 시간 정렬 규칙은 다음을 강하게 요구한다.

- timestamp index를 유일한 time axis로 사용
- condition은 시점 $t$ 까지의 정보만 사용
- target은 반드시 미래 시점 $t+h$ 에 대해 계산
- look-ahead bias 금지
- resampling 금지

### 7.5 Execution wrapper

LLM이 생성한 코드는 직접 실행되지 않는다. 시스템은 wrapper script를 생성하여 다음 절차를 수행한다.

1. 프로젝트 루트를 `sys.path` 에 추가
2. dataset schema를 로드
3. `DatasetAccessor` 생성
4. `candidate.py` 의 `evaluate_hypothesis` 호출
5. 결과 dict를 JSON 문자열로 출력

이 wrapper 방식은 evaluator 코드가 최소한 동일한 실행 환경과 데이터 인터페이스를 사용하게 만들며, evaluator 자체는 오직 함수 구현에만 집중하도록 한다.

### 7.6 Syntax check, execution, and fallback

Evaluator는 코드 생성 후 즉시 Python AST 파싱을 통해 문법 검사를 수행한다. 그 뒤 로컬 임시 디렉토리에서 subprocess로 실행된다.

실패 가능성은 세 단계로 분류할 수 있다.

1. **generation failure**: LLM이 코드 자체를 잘못 생성
2. **syntax failure**: AST parse 실패
3. **runtime failure**: timeout, non-zero exit, empty stdout, invalid JSON

각 실패는 다음 generation attempt의 prompt에 반영된다. 모든 재시도가 실패하면 evaluator는 전체 required key를 갖는 zero-valued payload를 반환하며, `rationale` 에 실패 원인을 기록한다.

### 7.7 Numerical sanitization

Evaluator 출력은 후처리를 통해 다음을 강제한다.

- required key 존재
- numeric field의 타입 안정성 확보
- `NaN`, `inf`, `-inf` 제거
- 잘못된 `used_parameters` 를 빈 dict로 대체

이로써 archive와 search loop는 외부 LLM의 불안정한 출력을 받더라도 최소한의 폐쇄성을 유지한다.

### 7.8 Same-run duplicate skip before evaluation

현재 구현은 **evaluation 자체를 캐시하기보다, 이미 같은 run 안에서 관측된 ELG를 evaluation 직전에 건너뛰는 novelty gate** 를 둔다. 핵심 아이디어는 다음과 같다.

1. child ELG가 생성되면 정규화 기반 fingerprint를 계산한다.
2. 현재 run에서 이미 archive에 반영된 fingerprint 집합과 비교한다.
3. 이미 알려진 fingerprint이면 evaluator code generation과 subprocess execution을 수행하지 않고 즉시 skip한다.

이 정책은 특히 evaluator 단계가 가장 큰 병목이라는 점을 반영한 것이다. 같은 ELG를 다시 score해도 새 정보가 거의 없으므로, 현재 시스템은 “같은 run에서 이미 본 ELG는 기본적으로 다시 평가하지 않는다”는 쪽을 택한다.

다만 이 최적화는 **run-local** 하다. 즉 한 번의 `controller.run(...)` 내부에서만 적용되며, persisted checkpoint를 다음 run의 evaluation cache로 복구하지는 않는다.

---

## 8. Stage III: Archive, Diversity Descriptor, and Parent Sampling

### 8.1 Archive structure

HypoEvolve의 archive 클래스는 `MAPElitesArchive` 라는 이름을 사용하지만, 현재 구현은 전형적인 full MAP-Elites 시스템이라기보다 **coverage와 structural complexity의 2차원 descriptor 공간에 top-k 후보를 유지하는 compact archive** 에 가깝다. 따라서 본 문서에서는 이를 “MAP-Elites-like quality-diversity memory”로 해석한다.

각 candidate는 다음 속성을 가진다.

- hypothesis
- metrics
- fingerprint
- iteration
- metadata
- coverage
- complexity
- cell

### 8.2 Descriptor definition

현재 descriptor는 다음 두 요소로 정의된다.

1. **coverage**: evaluator가 보고한 경험적 condition coverage
2. **complexity**: hypothesis node count

coverage는 $[0,1]$ 범위로 coercion되며, complexity는 정수 노드 수이다.

archive cell은 다음과 같이 계산된다.

$$
\mathrm{cell}(H) = (b_{cov}(\mathrm{coverage}(H)), b_{cmp}(\mathrm{complexity}(H)))
$$

여기서:

- $b_{cov}$ 는 `bisect_right` 로 계산되는 coverage bin index
- $b_{cmp}$ 는 `bisect_left` 로 계산되는 complexity bin index

### 8.3 Per-cell elite retention

각 cell에는 최대 `per_cell_top_k` 개의 후보만 유지된다. 동일 cell 내부에서 같은 fingerprint를 가진 후보가 다시 들어오면, 더 높은 score를 가진 버전만 유지된다.

이 archive dedup은 insertion 시점에 일어나는 반면, search loop의 novelty gate는 그보다 앞선 단계에서 작동한다. 따라서 현재 시스템은

- **pre-eval duplicate skip**: 이미 알려진 ELG면 evaluation 자체를 생략
- **archive dedup**: 그럼에도 archive에 들어오려는 동일 fingerprint 후보를 cell 내부에서 정리

의 두 층을 가진다.

이 규칙은 다음 목적을 가진다.

- 단순 중복 제거
- 동일 구조의 score 향상본만 보존
- descriptor diversity 유지
- 메모리 사용량 상한 제공

### 8.4 Parent selection

Parent selection은 2단계로 수행된다.

1. **occupied cell을 균등 샘플링**
2. 선택된 cell 내부에서 UCB-style score가 최대인 entry 선택

cell 내부 선택 점수는 다음과 같이 정의된다.

$$
\mathrm{UCB}(i) =
\begin{cases}
\infty, & \text{if } pulls_i = 0 \\
\bar{r}_i + c \sqrt{\frac{\log P}{pulls_i}}, & \text{otherwise}
\end{cases}
$$

여기서:

- $pulls_i$ 는 해당 후보가 parent로 선택된 횟수
- $\bar{r}_i$ 는 평균 reward
- $P$ 는 같은 cell 내 후보들의 총 parent-pull 수
- $c = 0.01$ 은 exploration weight

무한대 초기값을 사용하므로, 아직 한 번도 parent로 선택되지 않은 entry는 우선적으로 탐색된다.

### 8.5 Reward update

Parent selection 후 child가 평가되면, parent의 reward는 다음과 같이 기록된다.

$$
\Delta = \mathrm{score}(H_{child}) - \mathrm{score}(H_{parent})
$$

즉 reward는 절대 score가 아니라 **부모 대비 개선량(delta)** 이다. 이 설계는 “어떤 후보를 확장하는 것이 탐색 관점에서 유리한가”를 측정하는 bandit-style 해석과 잘 맞는다.

---

## 9. Stage IV: Mutation Steering

### 9.1 Input context

Child proposal은 단순 random tree edit가 아니라, parent-aware prompt를 받는 LLM steering 단계로 생성된다. 입력 컨텍스트는 다음과 같다.

- parent measurable ELG
- parent natural-language rendering
- current metrics
- recent mutation history (최대 최근 3개)
- top archive hypotheses (상위 3개)

### 9.2 Two steering modes

현재 구현은 두 가지 steering 모드를 사용한다.

1. **Score-directed steering**
   - 현재 metrics를 반영해 precision/coverage trade-off를 개선하려는 child를 생성
   - `domain_reason`, `score_reason`, `operation_score_rankings`, `mutation_summary` 를 포함
2. **Random exploratory steering**
   - `random_steering_prob` 확률로 호출
   - 더 적은 제약과 더 적은 메타데이터를 갖는 exploratory mutation을 생성

이때 random steering 여부는 config에 의해 제어된다.

### 9.3 Full-rewrite but local-mutation policy

Mutation steering은 child hypothesis를 부분 diff로 반환하지 않는다. 대신 **child 전체 ELG root node** 를 다시 생성한다. 그러나 prompt 수준에서는 이것이 parent에 대한 “국소적 mutation” 으로 행동하도록 요구한다. 즉 구현 관점에서는 full rewrite이고, 알고리즘 관점에서는 **locally constrained proposal distribution** 으로 볼 수 있다.

### 9.4 Prompted mutation family

Score-directed steering prompt는 다음 mutation family를 우선 후보로 제시한다.

- `replace_atomic_feature`
- `replace_atomic_reformulate`
- `append_atomic`
- `remove_atomic`
- `change_relation_type`
- `wrap_not`

중요한 점은 ELG core 자체는 이보다 더 일반적인 구조 변환을 표현할 수 있지만, 현재 steering prompt는 탐색 안정성을 위해 더 좁은 mutation family를 사용한다는 것이다.

### 9.5 Structural constraint on child hypothesis

Steering 결과는 다음 구조 제약을 만족해야 한다.

- child는 완전한 relation-level proposition이어야 함
- root는 relation node를 유지해야 함
- condition side와 target side를 모두 보존해야 함
- measurable atomic 표현을 우선해야 함

이 제약은 search loop가 fragment-level proposal이나 degenerate hypothesis로 붕괴하는 것을 방지한다.

### 9.6 Output validation

Mutation steering 출력은 JSON parse 후 다음을 검증한다.

- `domain_reason`, `score_reason`, `mutation_summary` 의 non-empty 여부
- `operation_score_rankings` 의 dict 여부 및 rank 정수성
- `child_hypothesis` 의 dict 여부
- resulting ELG object의 정상 생성 가능 여부

이 validation을 통과한 child만 archive 반영 단계로 이동한다.

---

## 10. Serial Search Procedure

Serial mode는 가장 직접적인 구현이며, 검색 동작을 이해하기 위한 기준선이다.

### Algorithm 2. Serial local search update

```text
Initialize recent_history = []
For iteration t in {1, ..., T}:
  1. parent <- sample_parent(archive)
  2. child <- steer_mutation(parent, recent_history[-3:], archive_top3)
  3. metrics <- evaluate(child)
  4. delta <- score(metrics) - score(parent)
  5. reflect_result(child, metrics, delta)
  6. record_parent_outcome(parent.fingerprint, delta)
  7. append result summary to recent_history
```

`recent_history` 에는 현재까지의 mutation summary, domain/score reason, 결과 hypothesis, score delta가 누적되며, subsequent steering prompt는 이 중 최근 3개만 참조한다. 이 설계는 장기 메모리 폭주를 피하면서도, 아주 최근의 탐색 실패·성공 패턴은 반영하도록 한다.

---

## 11. Parallel Worker Procedure

### 11.1 Motivation

Worker mode는 evaluator와 steering이 모두 LLM 및 subprocess를 포함하기 때문에, iteration 간 대기 시간이 상대적으로 큰 상황에서 wall-clock throughput을 높이기 위해 도입되었다.

### 11.2 Worker task payload

각 worker는 다음 정보를 직렬화된 task로 전달받는다.

- parent hypothesis
- parent hypothesis natural language
- parent metrics
- iteration id
- parent score
- random steering flag
- LLM config
- dataset schema path
- evaluator parameters
- parser/steering retry budget
- recent history
- top hypotheses snapshot
- seen fingerprint snapshot

### 11.3 Worker-side execution

Worker는 독립적으로 다음 절차를 수행한다.

1. parent ELG 복원
2. 로컬 LLM client 구성
3. dataset schema 로드
4. worker-local evaluator 생성
5. 필요 시 parent hypothesis natural-language rendering 재생성
6. mutation steering 수행
7. child fingerprint 계산 및 task에 포함된 seen snapshot과 비교
8. 이미 알려진 fingerprint이면 evaluation 없이 duplicate-skip result 반환
9. 아니면 child hypothesis 평가
10. result payload 반환

### 11.4 Controller-side scheduling

Parallel controller는 처음에 `min(worker_count, total_iterations)` 개의 task를 제출한다. 이후 완료된 future를 감지할 때마다 다음을 수행한다.

1. worker result 회수
2. duplicate-skip 여부 확인
3. skip이 아니면 archive 반영 및 reward update
4. recent history 갱신
5. 남은 iteration budget이 있으면 새 task 제출

이 방식은 strict synchronous generation이 아니라 **completion-driven asynchronous refill** 에 가깝다.

### Algorithm 3. Parallel worker scheduling

```text
1. submit up to min(W, T) worker tasks
2. while pending futures exist:
3.   wait for any completed future
4.   if result is duplicate-skip:
5.      do not evaluate/integrate child
6.   else:
7.      integrate returned child and metrics into archive
8.      update parent reward statistics
9.      append new result to recent_history
10.  if iteration budget remains:
11.     build a fresh worker task from current archive snapshot
12.     submit the next worker
```

### 11.5 Trade-off

Parallel mode는 throughput 측면에서는 유리하지만, 완전히 직렬적인 “한 step의 결과가 다음 step의 prompt context에 즉시 반영되는” 이상적인 closed-loop search와는 다르다. 즉 이미 제출된 worker task는 이전 archive snapshot을 기반으로 작동하므로, 일부 stale context가 존재한다. 현 구현은 이 trade-off를 감수하고도 practical speedup을 얻는 쪽을 선택한다.

따라서 worker duplicate-skip 역시 완전한 global dedup은 아니다. 이미 archive에 들어온 fingerprint에 대해서는 높은 확률로 evaluation을 피할 수 있지만, 서로 다른 worker가 **동시에 아직 archive에 반영되지 않은 동일 child** 를 생성한 경우에는 중복 evaluation이 남을 수 있다. 현재 설계는 이 남은 race를 허용하는 대신 구현 복잡도를 낮추는 쪽을 택한다.

---

## 12. Runtime Persistence and Observability

HypoEvolve는 각 run마다 독립 디렉토리를 생성하고, 탐색 과정을 복수의 artifact로 저장한다.

### 12.1 Run directory

기본 출력 경로는 다음과 같다.

```text
.hypoevolve/runs/<run-id>/
```

여기서 `run-id` 는 기본적으로 난수 UUID prefix이다.

### 12.2 Persisted artifacts

각 run은 다음 파일을 생성한다.

- `trace.jsonl`: iteration별 event append-only log
- `checkpoint.json`: 최신 archive snapshot
- `best.json`: 현재 최고 가설과 metrics
- `artifacts/seed.json`: 시드 가설 artifact
- `artifacts/iteration_XXXX.json`: 각 iteration 산출물
- `hypoevolve.log`: 실행 로그 파일

### 12.3 Trace event schema

각 trace event는 다음 정보를 포함한다.

- `iteration`
- `parent`
- `child`
- `metrics`
- `metadata`

metadata에는 예컨대 다음이 들어간다.

- hypothesis natural language
- parent score
- score delta
- mutation summary
- domain reason
- score reason
- map-elites descriptor
- random steering 여부
- worker mode 여부

즉 trace는 단순 로그가 아니라, **탐색 trajectory를 재구성할 수 있는 실험 기록** 역할을 한다.

또한 `hypoevolve.log` 에는 duplicate-skip 관련 요약 로그가 남는다.

- iteration-level: `iter.skip_duplicate`, `worker.skip_duplicate`
- run-level summary: `run.duplicate_summary`

이 로그를 통해 한 run에서 novelty gate가 얼마나 자주 evaluation을 절약했는지 빠르게 확인할 수 있다.

---

## 13. Configuration Space

HypoEvolve는 경량 YAML config를 사용하며, 주요 제어 변수는 다음과 같다.

### 13.1 LLM configuration

- `model`
- `temperature`
- `max_tokens`
- `api_key`
- `api_base`
- `timeout`
- `retries`
- `retry_delay`

### 13.2 Parser and evaluator configuration

- `parser.retries`
- `evaluator.dataset_schema_path`
- `evaluator.parameters`
- `evaluator.seed`

### 13.3 Search configuration

- `search.iterations`
- `search.steering_retries`
- `search.random_steering_prob`
- `search.random_seed`

### 13.4 Archive configuration

- `archive.coverage_bins`
- `archive.complexity_bins`
- `archive.per_cell_top_k`

### 13.5 Runtime configuration

- `output.base_dir`
- `logging.level`
- `workers.enabled`
- `workers.count`

이 구성은 “탐색 예산”, “proposal stochasticity”, “quality-diversity granularity”, “병렬성”, “LLM backend”를 서로 독립적으로 제어할 수 있게 한다.

---

## 14. Design Rationale

### 14.1 Why ELG instead of raw text?

자연어 가설만으로는 중복 판정, 구조 변형, complexity 측정이 어렵다. ELG는 다음 장점을 제공한다.

- 구조적 정규화 가능
- 논리 연산 단위 mutation 가능
- fingerprint 기반 dedup 가능
- readable rendering과 machine execution 사이의 매개체 역할

### 14.2 Why use LLMs for both proposal and evaluator generation?

Hypothesis space가 넓고 도메인별 operationalization이 다르기 때문에, 고정 템플릿 evaluator 하나로는 충분하지 않다. HypoEvolve는 LLM을 두 곳에 배치한다.

1. **proposal model**: hypothesis structure를 제안
2. **code generator**: measurable hypothesis를 dataset-aware evaluator code로 operationalize

즉 LLM은 단순 judge가 아니라, **proposal distribution과 evaluation implementation을 동시에 제공하는 생성 계층**으로 기능한다.

### 14.3 Why uplift × coverage?

정밀도(precision)만 최대화하면 극도로 희귀한 패턴이 과대평가될 수 있다. 반대로 coverage만 높이면 baseline과 구별되지 않는 일반 사건에 높은 점수가 갈 수 있다. `uplift × coverage` 는 다음 균형을 제공한다.

- baseline 대비 개선(uplift)
- 조건의 실질적 적용 범위(coverage)

따라서 이 점수는 “드물지만 완벽한 규칙”과 “흔하지만 무의미한 규칙”을 동시에 억제한다.

### 14.4 Why archive-based selection instead of greedy hill climbing?

가설 탐색은 다봉성(multimodality)을 갖고, 서로 다른 구조가 서로 다른 방식으로 유망할 수 있다. archive는 다음 효과를 제공한다.

- 단일 best candidate로의 조기 수렴 완화
- coverage-complexity 공간의 다양성 유지
- bandit-style parent reuse를 통한 proposal budget 재배분

---

## 15. Positioning Relative to Adjacent Systems

HypoEvolve는 세 가지 계열의 시스템과 인접해 있다.

1. **rule discovery systems**: 데이터 위에서 설명 가능한 규칙을 찾는다는 점에서 유사하지만, HypoEvolve는 규칙 후보를 자연어-구조-코드의 다층 표현으로 다룬다.
2. **program evolution systems**: proposal, archive, iterative selection의 구조를 공유하지만, 진화 대상이 실행 코드가 아니라 구조화된 가설이라는 점이 다르다.
3. **LLM-only reasoning systems**: LLM을 중심에 두지만, 최종 평가는 자유 텍스트 판단이 아니라 데이터셋 위의 실행 결과로 환원된다는 점이 다르다.

따라서 HypoEvolve는 가장 적절하게는 **symbolic hypothesis representation, LLM-guided generation, and dataset-grounded empirical scoring을 결합한 hybrid research system** 으로 위치지을 수 있다.

---

## 16. Current Limitations

본 절은 현재 구현의 한계를 방법론 수준에서 명시한다.

### 16.1 Evaluator execution trust boundary

Evaluator 코드는 timeout과 출력 검증을 가지지만, 본질적으로 LLM이 생성한 Python 코드를 로컬 subprocess에서 실행한다. 따라서 현재 시스템은 **실험용 신뢰 경계(trust boundary)** 를 가지며, 강한 샌드박스 보안 모델을 전제로 하지 않는다.

### 16.2 Prompt-level search space restriction

ELG core는 `AND`, `OR`, `NOT` 을 지원하지만, 현재 steering prompt는 주로 `AND`, `NOT` 기반 child generation을 유도한다. 따라서 이론적 hypothesis space와 실제 proposal distribution 사이에는 불일치가 존재한다.

### 16.3 Descriptor simplicity

Archive descriptor는 coverage와 node complexity 두 차원만 사용한다. 이는 계산이 간단하고 안정적이지만, 의미론적 novelty나 temporal regime diversity를 직접 반영하지는 못한다.

### 16.4 Partial staleness in parallel mode

Parallel mode에서 이미 제출된 worker task는 archive의 최신 상태를 즉시 반영하지 못한다. 따라서 완전한 synchronous closed-loop search와 비교하면 context staleness가 존재한다.

### 16.5 External model dependence

Parser, measurable rewrite, natural-language rendering, mutation steering, evaluator codegen 모두 외부 LLM backend에 의존한다. 따라서 재현성과 비용은 모델 응답 품질 및 API 상태에 영향을 받는다.

---

## 17. Reproducibility and Experimental Reading

HypoEvolve 결과를 읽을 때에는 다음 세 층위를 분리해 해석하는 것이 바람직하다.

1. **Representation layer**: ELG가 실제로 어떤 구조를 표현했는가
2. **Proposal layer**: LLM이 어떤 mutation 방향을 선택했는가
3. **Execution layer**: evaluator code가 그 가설을 데이터 위에서 어떻게 operationalize했는가

이 세 층위를 분리하면, score 개선이 진짜 hypothesis quality 개선인지, evaluator operationalization 차이 때문인지, 또는 prompt stochasticity 때문인지 더 정교하게 분석할 수 있다.

실험 기록 측면에서 최소한 다음 파일을 함께 읽는 것이 권장된다.

- `best.json`
- `checkpoint.json`
- `trace.jsonl`
- `artifacts/seed.json`
- `artifacts/iteration_XXXX.json`
- `hypoevolve.log`

---

## 18. Summary

HypoEvolve는 현재 구현 기준으로 다음과 같이 요약될 수 있다.

> HypoEvolve는 자연어 가설을 ELG라는 구조적 논리 표현으로 변환하고, LLM이 생성한 mutation proposal과 evaluator code를 이용해 데이터셋 위에서 가설의 경험적 유용성을 반복적으로 최적화하는 archive-based hypothesis evolution system이다.

핵심 특징은 다음 네 가지다.

1. **가설을 코드가 아닌 구조 객체로 직접 다룬다.**
2. **LLM을 파서이자 proposal model이자 evaluator code generator로 사용한다.**
3. **uplift × coverage 기반 점수로 경험적 유용성을 측정한다.**
4. **coverage-complexity archive와 UCB-style parent sampling으로 탐색을 조직한다.**

따라서 HypoEvolve는 단순한 규칙 엔진도, 단순한 prompt optimizer도 아니다. 현재 시스템은 보다 정확히 말해, **구조적 가설 표현, 생성적 코드 작성, quality-diversity memory를 결합한 연구용 hypothesis search runtime** 이다.

---

## Appendix A. Implementation Anchors

본 문서와 직접적으로 대응되는 구현 파일은 다음과 같다.

- ELG types: `elg/ir.py`
- ELG codec: `elg/codec.py`
- ELG normalization and fingerprinting: `elg/normalize.py`, `elg/metrics.py`
- Structural mutation helpers: `elg/mutate.py`
- Config model: `hypoevolve/config.py`
- CLI entrypoint: `hypoevolve/cli.py`
- LLM client: `hypoevolve/llm.py`
- Parser and measurable rewrite: `hypoevolve/parser.py`
- Evaluator orchestration: `hypoevolve/evaluator.py`
- Executor sandbox wrapper: `hypoevolve/executor.py`
- Prompt helpers: `hypoevolve/prompts.py`, `hypoevolve/helper.py`
- Archive and parent sampling: `hypoevolve/archive.py`
- Search controller: `hypoevolve/controller.py`
- Worker runtime: `hypoevolve/workers.py`
- Dataset abstraction: `hypoevolve/dataset.py`
- Runtime persistence: `hypoevolve/runtime.py`
