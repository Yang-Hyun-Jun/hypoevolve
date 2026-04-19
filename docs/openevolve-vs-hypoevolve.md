# OpenEvolve vs HypoEvolve 기능 비교

이 문서는 **현재 구현 기준**으로 `OpenEvolve`와 `HypoEvolve`를 기능 단위로 비교한 문서다.  
핵심 질문은 다음 세 가지다.

1. 지금 HypoEvolve는 OpenEvolve 대비 **어디까지 와 있는가?**
2. 어떤 기능은 아직 **OpenEvolve보다 부족한가?**
3. 반대로 어떤 부분은 HypoEvolve가 **이미 더 잘 정렬되어 있거나 강한가?**

비교 기준은 로컬 저장소에 있는 현재 구현체다.  
- OpenEvolve 구현 참조: `openevolve/openevolve/`  
- HypoEvolve 구현 참조: `hypoevolve/`, `hypoevolve/elg/`

---

## 1. 한 문장 요약

- **OpenEvolve**는 코드 진화를 위한 비교적 완성도 높은 end-to-end evolutionary system이다.  
- **HypoEvolve**는 가설 진화를 위한 도메인 특화 코어와, 그 위에 얹은 compact MVP app layer까지 구현된 상태다.

즉,
- **시스템 완성도 총량**은 OpenEvolve가 더 높고,
- **가설 표현/변형 적합성**은 HypoEvolve가 더 높다.

---

## 2. 전체 비교 요약표

| 기능 영역 | OpenEvolve | HypoEvolve | 현재 판단 |
|---|---|---|---|
| 후보 표현 | 코드 문자열 중심 | ELG 구조 중심 | **HypoEvolve 우위** |
| 후보 직렬화/복원 | Program dict 저장 | ELG codec + JSON round-trip | **HypoEvolve 우위** |
| 후보 정규화 | 코드 기준 없음/약함 | normalize / fingerprint 있음 | **HypoEvolve 우위** |
| mutation primitive | 코드 diff/rewrite 중심 | 구조적 immutable mutation 제공 | **HypoEvolve 우위** |
| 랜덤 mutation sampler | 명시적 primitive sampler 없음 | 없음 (현재는 LLM이 child ELG를 직접 생성) | 비슷 |
| LLM proposal integration | 강함 | parser / steering / evaluator에 실제 LLM 경로 존재 | **OpenEvolve 우위, 격차 축소** |
| evaluator 성숙도 | 높음 | LLM-generated evaluator runtime + retry/repair 있음 | **OpenEvolve 우위, 격차 축소** |
| archive/best tracking | 강함 | compact top-k archive | **OpenEvolve 우위** |
| quality-diversity (MAP-Elites/islands) | 있음 | 없음 | **OpenEvolve 우위** |
| runtime persistence | checkpoint/trace/artifact 풍부 | checkpoint/trace/report/artifact 구현 있음 | **OpenEvolve 우위** |
| CLI / UX | 있음 | MVP용 서브커맨드 있음 | **비슷, 목적 다름** |
| 병렬화/worker | 있음 | worker mode 있음 | **OpenEvolve 우위** |
| 도메인 적합성(가설) | 낮음 | 높음 | **HypoEvolve 우위** |

---

## 3. 후보 표현(Representation)

### OpenEvolve
OpenEvolve의 후보는 본질적으로 `Program` 객체이고, 중심 필드는 `code: str`이다. 즉 후보의 정체성이 결국 **코드 문자열**에 있다. 이 구조는 코드 최적화에는 잘 맞지만, 가설/명제/논리 구조를 직접 다루기에는 부자연스럽다. (`openevolve/openevolve/database.py`)

### HypoEvolve
HypoEvolve는 `ELG`를 후보 표현의 중심으로 둔다. `AtomicNode`, `LogicalNode`, `RelationNode`, `Hypothesis`가 있고, hypothesis는 구조적 트리로 표현된다. 즉 후보가 처음부터 **논리 구조 객체**다. (`hypoevolve/elg/ir.py`)

### 판단
가설 진화라는 문제에 한정하면, 이 영역은 **HypoEvolve가 이미 더 잘 설계되어 있다.**

---

## 4. 직렬화 / 복원 / 정규화

### OpenEvolve
OpenEvolve는 `Program.to_dict()`, `Program.from_dict()`를 통해 저장/복원은 가능하다. 하지만 이것은 코드 후보 객체 저장에 가까우며, 구조적 의미의 canonicalization은 거의 없다. (`openevolve/openevolve/database.py`)

### HypoEvolve
HypoEvolve는 다음을 이미 갖췄다.
- `node_to_dict`, `node_from_dict`
- `hypothesis_to_json`, `hypothesis_from_json`
- `normalize_node`, `normalize_hypothesis`
- `fingerprint`

즉 저장/복원뿐 아니라, **구조적 동치에 가까운 정규화 기반 식별**이 가능하다. (`hypoevolve/elg/codec.py`, `hypoevolve/elg/normalize.py`, `hypoevolve/elg/metrics.py`)

### 판단
이 부분은 **HypoEvolve가 OpenEvolve보다 가설 도메인에 더 잘 맞고, 기술적으로도 더 정교한 코어를 갖고 있다.**

---

## 5. Mutation 시스템

### OpenEvolve
OpenEvolve는 mutation primitive를 명시적으로 제공하지 않는다. 대신 worker 루프에서 LLM이 diff 또는 full rewrite를 생성하고, 그 결과를 텍스트로 적용한다. 즉 mutation은 **LLM-generated code edit**다. (`openevolve/openevolve/process_parallel.py`, `openevolve/openevolve/utils/code_utils.py`)

### HypoEvolve
HypoEvolve는 mutation primitive를 명시적으로 갖고 있다.
- path traversal
- subtree replacement
- logical op change
- relation type change
- wrap/unwrap NOT
- append/remove child

또한 이 mutation은 **immutable 구조 변환**으로 구현돼 있다. (`hypoevolve/elg/mutate.py`)

### 판단
가설 구조를 다루는 관점에서는 **HypoEvolve의 mutation core가 훨씬 직접적이고 강하다.**  
다만 OpenEvolve는 이 위에 LLM proposal을 실전적으로 얹어놓았고, 그 부분은 아직 HypoEvolve가 약하다.

---

## 6. 랜덤 mutation sampler

### OpenEvolve
OpenEvolve는 parent sampling은 확률적으로 하지만, 명시적인 구조적 mutation sampler는 없다. 실제 변형은 prompt와 LLM 응답을 통해 만들어진다. (`openevolve/openevolve/database.py`, `openevolve/openevolve/process_parallel.py`)

### HypoEvolve
현재 HypoEvolve에는 별도 random mutation sampler가 없다.
실제 mutation steering은 parent measurable ELG와 metric 문맥을 바탕으로
LLM이 full child ELG를 직접 생성하는 방식이다. (`hypoevolve/mutation.py`, `hypoevolve/prompts/steering/system.md`)

### 판단
이 부분은 현재 기준으로 **둘 다 명시적 primitive sampler는 없다**고 보는 편이 정확하다.

---

## 7. LLM integration

### OpenEvolve
OpenEvolve는 여전히 이 부분이 강하다.
- `LLMEnsemble`
- OpenAI-compatible generation backend
- PromptSampler
- mutation content를 LLM이 직접 생성
- evaluator에 optional LLM feedback도 가능

즉 OpenEvolve는 **LLM이 실전 검색 루프 안에 깊게 박혀 있다.** (`openevolve/openevolve/llm/ensemble.py`, `openevolve/openevolve/llm/openai.py`, `openevolve/openevolve/prompt/sampler.py`, `openevolve/openevolve/evaluator.py`)

### HypoEvolve
HypoEvolve는 이제 구조만 준비된 상태는 아니다.
- `LLMConfig` 기반 설정이 실제 parser / evaluator / steering 경로에 연결되어 있다.
- parser는 자연어 → ELG 변환과 measurable rewrite에 LLM을 사용한다.
- evaluator는 hypothesis별 Python evaluator code를 LLM이 생성하고 subprocess에서 실행한다.
- mutation steering도 LLM이 child ELG와 mutation rationale을 직접 생성한다.
- 다만 orchestration breadth, ensemble sophistication, sampler richness는 여전히 OpenEvolve 쪽이 더 넓다.

(`hypoevolve/config.py`, `hypoevolve/parser.py`, `hypoevolve/evaluator.py`, `hypoevolve/mutation.py`)

### 판단
이 영역은 **OpenEvolve가 훨씬 앞서 있다.**  
HypoEvolve는 아직 "LLM-friendly architecture"이지, "LLM-driven system"은 아니다.

---

## 8. Evaluator

### OpenEvolve
OpenEvolve evaluator는 상당히 성숙하다.
- evaluator file 동적 로딩
- timeout
- retry
- artifacts
- cascade evaluation
- optional LLM feedback blending

즉 fitness adapter로서 기능이 많다. (`openevolve/openevolve/evaluator.py`, `openevolve/openevolve/config.py`)

### HypoEvolve
HypoEvolve evaluator는 intentionally thin 하다.
- `Evaluator` protocol
- `LLMEvaluator`
- LLM-generated evaluator code execution
- retry / repair / artifact capture

즉 OpenEvolve만큼 넓은 evaluator feature set은 아니지만, 실제 데이터 기반 hypothesis 검증 경로는 이미 구현되어 있다. (`hypoevolve/evaluator.py`)

### 판단
이 부분은 여전히 **OpenEvolve 우위**지만, 예전처럼 placeholder-only 상태는 아니다.

---

## 9. Archive / best tracking

### OpenEvolve
OpenEvolve는 단순 best tracking이 아니라:
- archive
- top programs
- island population
- MAP-Elites feature map
- migration
- best program tracking

까지 있다. (`openevolve/openevolve/database.py`)

### HypoEvolve
HypoEvolve는 MVP 기준으로 compact archive를 갖고 있다.
- top-k=5
- fingerprint dedup
- best tracking
- weighted parent sampling

즉 최소한의 evolutionary memory는 있다. (`hypoevolve/archive.py`)

### 판단
OpenEvolve가 더 풍부하지만, **HypoEvolve MVP는 최소 실행 가능한 archive를 이미 갖고 있다.**

---

## 10. Quality-diversity / MAP-Elites / islands

### OpenEvolve
이건 OpenEvolve의 큰 강점이다.
- feature dimensions
- feature bins
- island populations
- migration
- archive replacement logic

즉 다양성을 적극적으로 보존하는 quality-diversity search다. (`openevolve/openevolve/database.py`, `openevolve/openevolve/config.py`)

### HypoEvolve
현재는 없음.
- top-k archive만 있음
- MAP-Elites 없음
- island model 없음
- migration 없음

### 판단
이 영역은 **OpenEvolve가 훨씬 앞서 있다.**

---

## 11. 병렬화 / worker runtime

### OpenEvolve
- process-based parallel controller
- worker init
- DB snapshot
- process pool execution

(`openevolve/openevolve/process_parallel.py`)

### HypoEvolve
- single-process loop
- 병렬 worker 없음
- distributed runtime 없음

(`hypoevolve/controller.py`)

### 판단
이건 명확히 **OpenEvolve 우위**다.

---

## 12. Runtime persistence: trace / checkpoint / artifact

### OpenEvolve
OpenEvolve는 trace/export 계층이 더 풍부하다.
- evolution trace module
- format 선택(jsonl/json/hdf5)
- artifact handling이 더 깊음
- checkpoint/resume도 성숙함

(`openevolve/openevolve/evolution_trace.py`, `openevolve/openevolve/database.py`, `openevolve/openevolve/controller.py`)

### HypoEvolve
HypoEvolve도 MVP 수준의 runtime persistence는 이미 있다.
- `.hypoevolve/runs/<run-id>/trace.jsonl`
- `checkpoint.json`
- `best.json`
- `artifacts/`

하지만 구조는 아주 가볍고, resume/orchestration sophistication은 거의 없다. (`hypoevolve/runtime.py`, `hypoevolve/controller.py`)

### 판단
OpenEvolve가 더 풍부하지만, **HypoEvolve는 MVP에 필요한 최소 운영 흔적은 이미 갖춤**.

---

## 13. Config / CLI / UX

### OpenEvolve
OpenEvolve는 config surface가 넓고, CLI와 library API를 둘 다 제공한다. 다만 코드 진화 중심이라 HypoEvolve에 그대로 맞지는 않는다. (`openevolve/openevolve/config.py`, `openevolve/openevolve/cli.py`, `openevolve/openevolve/api.py`)

### HypoEvolve
HypoEvolve는 더 작지만, 사용성은 MVP 기준으로 나쁘지 않다.
- `hypoevolve.yaml`
- CLI subcommands:
  - `run`
  - `render`
  - `inspect`
  - `doctor`
- console script wiring

(`hypoevolve/config.py`, `hypoevolve/cli.py`, `pyproject.toml`)

### 판단
절대적인 기능량은 OpenEvolve가 많지만, **HypoEvolve는 현재 목적에 맞는 compact CLI/UX를 이미 갖췄다.**

---

## 14. 도메인 적합성: 코드 vs 가설

### OpenEvolve
코드 진화에는 매우 잘 맞다. 그러나 후보가 코드 문자열이라는 전제가 강해서, hypothesis evolution에는 그대로 맞지 않는다. (`openevolve/openevolve/database.py`, `openevolve/openevolve/process_parallel.py`)

### HypoEvolve
가설 도메인에 맞춘 설계다.
- ELG 구조
- logical/relation/atomic 분리
- structural mutation
- canonicalization
- fingerprint
- top-k hypothesis archive

즉 도메인 적합성은 HypoEvolve 쪽이 훨씬 높다. (`hypoevolve/elg/ir.py`, `hypoevolve/elg/mutate.py`, `hypoevolve/elg/normalize.py`, `hypoevolve/archive.py`)

### 판단
이 영역은 **HypoEvolve가 OpenEvolve보다 분명히 더 잘 맞다.**

---

## 15. 기능 단위 최종 평가

### OpenEvolve보다 아직 약한 것
1. 실제 evaluator 성숙도  
2. LLM mutation/proposal integration  
3. quality-diversity search (MAP-Elites/islands/migration)  
4. 병렬화/worker runtime  
5. 운영성(trace/export/checkpoint sophistication)

### OpenEvolve보다 이미 강하거나 더 적합한 것
1. hypothesis representation  
2. structural mutation primitives  
3. LLM direct mutation steering  
4. canonicalization / fingerprint  
5. 가설 도메인 적합성

### OpenEvolve와 비슷한 감각으로 최소 구현된 것
1. controller orchestration  
2. archive / best tracking  
3. config + CLI  
4. runtime persistence의 최소 형태

---

## 16. 한 문장 결론

**시스템 완성도 총량은 아직 OpenEvolve가 더 높다. 하지만 HypoEvolve는 가설 진화라는 문제에 맞는 표현층과 구조 변형 계층을 이미 잘 갖췄고, 현재 구현 기준으로는 “도메인 적합한 MVP 엔진”까지는 올라온 상태다.**

즉 OpenEvolve는 **더 크고 더 완성된 코드 진화 시스템**,  
HypoEvolve는 **더 작지만 가설 진화에 맞게 잘 정렬된 시스템**이라고 보면 된다.
