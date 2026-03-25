# HypoEvolve LLM Integration Points

이 문서는 **현재 HypoEvolve 구현 기준**으로, 향후 LLM이 들어가야 하거나 들어갈 가능성이 높은 지점을 파일/함수 단위로 정리한 문서다.

목적은 단순하다.
- 어디에 LLM을 붙여야 하는지 빠르게 파악하기
- 어떤 함수가 parser / evaluator / mutation steering의 진입점인지 명확히 하기
- 이후 프롬프트 엔지니어링과 구현 우선순위를 잡기 쉽게 하기

---

## 1. 자연어 가설 → ELG 변환

### 파일
- `hypoevolve/parser.py`

### 핵심 함수
- `parse_hypothesis_text(...)`
- `fallback_parse_hypothesis(...)`

### 현재 상태
- `parse_hypothesis_text()`는 **injectable parser callable**을 받을 수 있다.
- custom parser가 없으면 `fallback_parse_hypothesis()`가 실행된다.
- `fallback_parse_hypothesis()`는 단순한 규칙 기반 fallback parser다.

### LLM이 들어갈 자리
가장 직접적인 LLM 주입 포인트는 다음과 같다.

```python
parse_hypothesis_text(text, parser=your_llm_parser)
```

즉 향후 아래 같은 함수가 들어갈 수 있다.

```python
def llm_parse_hypothesis(text: str) -> Hypothesis:
    ...
```

### 역할
- 자연어 가설 읽기
- ELG 구조(JSON 또는 객체) 생성
- 실패 시 retry / validation / fallback 처리

---

## 2. 가설 평가(Evaluation)

### 파일
- `hypoevolve/evaluator.py`

### 핵심 함수 / 인터페이스
- `Evaluator.evaluate(...)`
- `evaluate_hypothesis(...)`
- 현재 구현체: `PlaceholderEvaluator.evaluate(...)`

### 현재 상태
- 현재는 placeholder evaluator만 있고, deterministic random-ish metric을 반환한다.
- 실제 데이터 기반 hypothesis 검증 로직은 아직 구현되어 있지 않다.

### LLM이 들어갈 자리
핵심 진입점은 `Evaluator.evaluate(...)` 구현체다.

즉 향후 이런 구현이 가능하다.

```python
class LLMDataDrivenEvaluator:
    def evaluate(self, hypothesis: Hypothesis) -> Dict[str, float]:
        ...
```

### 역할
- hypothesis + 데이터/EDA context 기반 score 계산
- support / plausibility / novelty / critique 생성
- 필요 시 evaluation evidence까지 생성

---

## 3. mutation 방향 결정 / orchestration

### 파일
- `hypoevolve/controller.py`

### 핵심 함수
- `HypoEvolveController.run(...)`

### 현재 상태
현재 `run()`은 대략 아래 흐름으로 돈다.
- parser 호출
- archive 초기화
- parent 선택
- `sample_mutation(...)`
- evaluator 호출
- archive 반영
- runtime trace/checkpoint/artifact 기록

### LLM이 들어갈 자리
LLM critique 또는 mutation steering이 들어갈 수 있는 지점은:
- parent selection 이후
- 실제 mutation sample 선택 전

즉 구조적으로는 여기가 mutation 방향을 더 지능적으로 조절할 자리다.

예시:
- 현재 hypothesis critique
- 너무 강한 relation인지 판단
- 어떤 subtree를 바꾸는 게 좋을지 제안
- target horizon이나 condition complexity 조절 제안

### 역할
- 단순 random mutation을 넘어서, **LLM-guided search orchestration**을 담당하게 될 가능성이 높다.

---

## 4. mutation 샘플링 계층

### 파일
- `elg/sampler.py`

### 핵심 함수
- `generate_mutation_candidates(...)`
- `sample_mutation(...)`

### 현재 상태
- 현재는 legal mutation candidate를 생성하고
- seeded RNG로 하나를 샘플링하는 random sampler다.

### LLM이 들어갈 자리
이 계층은 두 가지 방식으로 확장될 수 있다.

#### A. 기존 sampler를 대체
예:
```python
def sample_mutation_guided(hypothesis, context, critique) -> MutationSample:
    ...
```

#### B. 기존 sampler를 보강
예:
- 후보를 모두 만든 뒤
- LLM이 ranking / filtering / weighting
- 최종 mutation 선택

### 역할
- legal mutation 후보 공간은 ELG가 유지하고
- 그 중 어떤 방향이 더 promising한지 LLM이 판단하게 만드는 계층

---

## 5. 앞으로 생길 가능성이 큰 LLM 전용 모듈

### 현재 없음
현재 `hypoevolve/`에는 아직 별도 LLM orchestration/prompt 모듈이 없다.

### 나중에 생길 가능성이 높은 파일
- `hypoevolve/prompts.py`
- `hypoevolve/llm.py`
- 혹은 `hypoevolve/critique.py`

### 여기서 관리할 가능성이 큰 것
- parser system prompt
- evaluator system prompt
- mutation critique prompt
- mutation recommendation schema
- retry / parse / validation prompt 전략

이 파일은 지금 당장 필수는 아니지만, parser/evaluator/mutation 모두에 LLM을 붙이기 시작하면 거의 필요해질 가능성이 높다.

---

## 6. 우선순위 기준 요약

### 가장 먼저 붙일 자리
1. `hypoevolve/parser.py`
2. `hypoevolve/evaluator.py`

이 둘이 제일 핵심이다.
- parser: NL → ELG 진입점
- evaluator: data-driven hypothesis scoring 핵심

### 그 다음
3. `hypoevolve/controller.py`
4. `elg/sampler.py`

이 둘은 mutation steering / critique 기반 검색 품질을 높이는 단계다.

---

## 7. 최소 리스트업

### 파일 기준
- `hypoevolve/parser.py`
- `hypoevolve/evaluator.py`
- `hypoevolve/controller.py`
- `elg/sampler.py`

### 함수 기준
- `parse_hypothesis_text(...)`
- `Evaluator.evaluate(...)`
- `HypoEvolveController.run(...)`
- `sample_mutation(...)`

---

## 8. 한 문장 결론

현재 코드 기준으로 **LLM이 원래 들어가야 하는 핵심 자리는 parser, evaluator, controller, sampler의 네 군데**이며, 우선순위는 보통:

> **parser → evaluator → controller → sampler**

순으로 보는 게 가장 자연스럽다.

---

## 9. 구현 체크리스트

### Parser
- [ ] `hypoevolve/parser.py`에 LLM parser callable 구현 추가
- [ ] LLM parser 출력 형식(JSON/ELG object) 계약 정의
- [ ] LLM parser 결과를 ELG validator로 검증
- [ ] parser retry 전략을 LLM 응답 실패 유형에 맞게 조정
- [ ] fallback parser와 LLM parser의 우선순위/전환 규칙 정의

### Evaluator
- [ ] `hypoevolve/evaluator.py`에 실제 LLM/data-driven evaluator 구현 추가
- [ ] evaluator 입력 계약 확정 (`Hypothesis` + dataset/context 등)
- [ ] evaluator 출력 metric schema 확정
- [ ] evaluator가 critique / evidence를 반환할지 여부 결정
- [ ] placeholder evaluator를 실제 evaluator로 교체하거나 공존 정책 정의

### Controller / Orchestration
- [ ] `hypoevolve/controller.py`에 LLM critique 호출 지점 명시
- [ ] parent selection 이후 mutation steering 훅 추가
- [ ] evaluator 결과를 다음 iteration prompt/context에 반영하는 규칙 정의
- [ ] single-process / worker mode 모두에서 LLM 흐름이 일관되게 동작하는지 점검

### Mutation Sampler
- [ ] `elg/sampler.py`를 LLM-guided sampler로 대체 또는 보강할지 결정
- [ ] legal mutation candidates를 LLM이 ranking/filtering할 수 있는 인터페이스 정의
- [ ] critique → mutation primitive 매핑 규칙 정의
- [ ] random sampler와 guided sampler의 fallback 정책 정의

### Prompt / LLM Layer
- [ ] 전용 프롬프트 모듈(`hypoevolve/prompts.py` 또는 `hypoevolve/llm.py`) 추가 여부 결정
- [ ] parser용 system prompt 작성
- [ ] evaluator용 system prompt 작성
- [ ] mutation critique / mutation proposal prompt 작성
- [ ] LLM 호출 공통 유틸(재시도, validation, logging) 필요 여부 결정

### Validation / Reliability
- [ ] LLM 출력 validation 전략 정의
- [ ] malformed output 처리 정책 정의
- [ ] deterministic test를 위한 mock LLM 경로 준비
- [ ] runtime trace/checkpoint에 LLM 관련 metadata를 얼마나 저장할지 결정

### MVP Readiness
- [ ] parser에 LLM을 붙인 최소 happy path 확보
- [ ] evaluator에 LLM을 붙인 최소 happy path 확보
- [ ] mutation steering까지 MVP에 포함할지 여부 최종 결정
- [ ] end-to-end: NL hypothesis → ELG → evolve → best hypothesis 출력 검증
