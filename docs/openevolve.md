# OpenEvolve 구현 및 방법론 정리

이 문서는 `OpenEvolve` 구현체를 **HypoEvolve 설계의 참고 자료**로 삼기 위해, OpenEvolve가 실제로 어떤 구조와 흐름으로 동작하는지 한국어로 정리한 문서다. 설명은 가능한 한 구현과 직접 대응되도록 적었고, 각 항목 끝에 관련 코드 파일을 괄호로 표기했다.

---

## 1. OpenEvolve를 한 문장으로 요약하면

OpenEvolve는 **LLM이 코드 후보를 생성/수정하고, 외부 evaluator가 그것을 실행해 점수를 매긴 뒤, 그 결과를 MAP-Elites + island model 기반 데이터베이스에 축적하면서 반복적으로 더 나은 코드를 탐색하는 시스템**이다. 즉, 단순한 "코드 생성기"가 아니라, **탐색(search) + 평가(evaluation) + 선택(selection) + 다양성 유지(diversity preservation)**를 모두 포함하는 진화형 최적화 루프다. (`openevolve/openevolve/controller.py`, `openevolve/openevolve/process_parallel.py`, `openevolve/openevolve/database.py`)

---

## 2. OpenEvolve의 가장 중요한 추상화

OpenEvolve에서 진화 대상은 이름상 "프로그램(program)"이지만, 더 본질적으로 보면 **평가 가능한 후보(candidate)** 다. 구현체에서는 이것이 `Program` dataclass로 표현되며, 핵심 필드는 `id`, `code`, `metrics`, `parent_id`, `generation`, `metadata`다. 즉 시스템은 결국 "코드 문자열을 가진 후보 객체"를 저장·샘플링·평가하는 구조다. (`openevolve/openevolve/database.py`)

이 관점은 HypoEvolve로 확장할 때 매우 중요하다. OpenEvolve의 본질은 "코드만 진화한다"가 아니라, **후보를 표현하고 평가하는 인터페이스가 코드 중심으로 구현되어 있다**는 점이다. (`openevolve/openevolve/database.py`, `openevolve/openevolve/process_parallel.py`)

---

## 3. 최상위 오케스트레이션 구조

OpenEvolve의 전체 실행은 `OpenEvolve` 클래스가 담당한다. 생성자에서 다음을 초기화한다.

1. 출력 디렉토리 및 로그
2. 시드 설정 및 재현성 관련 파라미터
3. 초기 프로그램 로드 및 언어/확장자 추론
4. LLM ensemble
5. Prompt sampler
6. Program database
7. Evaluator
8. Evolution tracer (옵션)
9. Process-based parallel controller (후속 실행 시)

즉 `OpenEvolve`는 여러 하위 모듈을 조합하는 **controller/orchestrator** 역할을 한다. (`openevolve/openevolve/controller.py`)

이 구현의 장점은 평가기, 프롬프트, 저장소, 병렬화가 모두 분리되어 있다는 점이다. 덕분에 HypoEvolve에서도 동일한 층위를 유지할 수 있다. 예를 들어 Hypothesis IR, Hypothesis evaluator, Hypothesis sampler를 각각 분리하는 설계가 가능하다. (`openevolve/openevolve/controller.py`)

---

## 4. 실행 흐름: 한 번의 iteration에서 무슨 일이 일어나는가

OpenEvolve의 실제 iteration은 `process_parallel.py`의 worker 루프에서 가장 잘 드러난다.

대략적인 흐름은 다음과 같다.

1. DB snapshot에서 부모(parent)와 inspiration 후보를 읽는다.
2. 부모가 속한 island의 프로그램들 중 top/diverse 후보를 모은다.
3. PromptSampler가 현재 코드, 이전 성능, top history, inspiration, artifact 등을 섞어서 프롬프트를 만든다.
4. LLM이 diff 또는 full rewrite를 생성한다.
5. 생성된 결과를 코드에 적용한다.
6. evaluator가 새 코드 후보를 실행해 metric을 계산한다.
7. 새 `Program` 객체를 만들고 DB에 추가한다.
8. DB는 MAP-Elites / island / best tracking / archive를 갱신한다.

즉 mutation은 단순 랜덤 비트 플립이 아니라 **LLM-guided proposal**이다. 다만 어떤 부모를 샘플링할지, 어떤 island에서 어떤 후보를 볼지 등은 확률적이다. 그래서 OpenEvolve는 전통적 진화 알고리즘과 비교하면, **선택은 확률적이고 변형 내용은 LLM이 방향성 있게 제안하는 hybrid 구조**라고 볼 수 있다. (`openevolve/openevolve/process_parallel.py`, `openevolve/openevolve/prompt/sampler.py`, `openevolve/openevolve/database.py`)

---

## 5. ProgramDatabase: OpenEvolve의 핵심 저장소

OpenEvolve에서 가장 중요한 구현 중 하나는 `ProgramDatabase`다. 이 모듈이 단순 리스트가 아니라 **MAP-Elites + island population model**을 구현하고 있기 때문이다.

핵심 기능은 다음과 같다.

- 프로그램 저장 (`programs`)
- island별 population 관리 (`islands`)
- island별 feature map 관리 (`island_feature_maps`)
- archive 관리 (`archive`)
- 전역 best program 추적 (`best_program_id`)
- island별 best 추적 (`island_best_programs`)
- migration 관리
- novelty/embedding 기반 거부 샘플링 옵션

즉 OpenEvolve는 단순히 best만 계속 덮어쓰는 탐욕적 탐색이 아니다. **다양한 feature cell에 좋은 해를 유지하는 quality-diversity search**다. (`openevolve/openevolve/database.py`)

이 부분은 HypoEvolve에서 특히 중요하다. 가설 탐색은 정답 하나만 찾는 문제가 아니라, **여러 형태의 plausible hypothesis frontier를 유지하는 문제**가 되기 쉽다. 따라서 MAP-Elites적 관점은 HypoEvolve에 매우 잘 맞는다. (`openevolve/openevolve/database.py`)

---

## 6. 선택(selection)은 어떻게 이루어지는가

DB는 parent sampling에서 exploration / exploitation / weighted sampling을 섞는다. current island에서 랜덤 샘플링을 하기도 하고, archive에서 엘리트를 가져오기도 하고, fitness-weighted sampling을 하기도 한다. 또한 island 내부 inspiration 후보도 함께 샘플링한다. (`openevolve/openevolve/database.py`)

즉 OpenEvolve의 selection은 다음 세 가지를 동시에 고려한다.

- 성능이 좋은 후보
- 현재 island 내의 구조적 다양성
- 다른 island들과의 적절한 분리 및 migration

이 구조 덕분에 local optimum에 너무 빨리 수렴하는 것을 완화한다. (`openevolve/openevolve/database.py`)

---

## 7. mutation은 랜덤인가, 지시적(guided)인가

OpenEvolve는 전통적 유전 알고리즘처럼 mutation 내용을 랜덤하게 직접 뒤흔들지 않는다. 실제 코드 변경 내용은 LLM이 생성한다. 하지만 LLM에게 주어지는 정보는 매우 풍부하다.

프롬프트에 들어가는 것들:
- 현재 프로그램 코드
- 현재 metrics
- improvement areas
- previous attempts
- top programs
- inspiration programs
- artifacts (실패 정보, timeout 등)
- feature dimensions

즉 mutation의 **내용**은 LLM이 제안하지만, mutation의 **탐색 경로**는 selection과 prompt construction에 의해 간접적으로 유도된다. (`openevolve/openevolve/prompt/sampler.py`, `openevolve/openevolve/process_parallel.py`, `openevolve/openevolve/evaluator.py`)

따라서 OpenEvolve는 전통적 EA보다 훨씬 더 **reflection-driven mutation**에 가깝다. 이 부분은 HypoEvolve에서 더 강화할 수 있다. 예를 들어 LLM이 가설 구조에 대한 critique를 먼저 내고, 그 critique를 바탕으로 ELG mutation sampler를 편향시키는 방식이 가능하다.

---

## 8. PromptSampler의 역할

`PromptSampler`는 단순 문자열 포맷터가 아니다. 현재 후보의 상태와 과거의 탐색 맥락을 LLM이 활용하기 좋은 형태로 재구성하는 역할을 한다.

핵심 포인트:
- diff-based prompt vs full rewrite prompt 선택
- improvement areas 계산
- previous attempts / top programs / inspirations 포맷팅
- feature dimension / fitness information 반영
- artifact section 삽입
- large-codebase mode에서 changes description 중심 표현도 지원

즉 LLM이 무의미한 랜덤 편집을 하지 않도록, **검색 이력과 메트릭을 prompt에 반영하는 계층**이다. (`openevolve/openevolve/prompt/sampler.py`, `openevolve/openevolve/prompt/templates.py`)

HypoEvolve로 옮기면 이 계층은 매우 중요해진다. 코드 대신 가설을 mutation할 때도, 단순히 현재 ELG만 주는 것보다 **현재 평가, 실패 원인, top hypothesis, inspiration hypothesis**를 같이 줘야 mutation quality가 올라간다.

---

## 9. Evaluator는 무엇을 하는가

`Evaluator`는 OpenEvolve의 외부 세계와 연결되는 관문이다. 시스템은 실제로 코드가 좋은지 자체적으로 알 수 없고, evaluator가 그것을 판정한다.

기본 구조:
- evaluation file을 동적으로 로드
- `evaluate(program_path)` 함수를 찾음
- 코드 후보를 임시 파일로 쓴 뒤 실행
- dict 또는 `EvaluationResult` 형태의 결과를 metric으로 변환
- timeout / retry / artifact 수집 처리
- 옵션으로 LLM feedback metric도 섞을 수 있음

즉 evaluator는 OpenEvolve에서 **fitness function adapter**다. (`openevolve/openevolve/evaluator.py`, `openevolve/openevolve/evaluation_result.py`)

HypoEvolve에서도 같은 구조가 자연스럽다. 다만 program 대신 hypothesis/ELG를 입력으로 받아, 데이터 기반 가설 검증 점수를 내는 evaluator로 바뀌면 된다.

---

## 10. 병렬화 방식

OpenEvolve는 `ProcessPoolExecutor` 기반의 process parallelism을 사용한다. worker는 lazy init으로 LLM ensemble, prompt sampler, evaluator를 초기화하고, database snapshot을 받아 한 번의 iteration을 수행한다. (`openevolve/openevolve/process_parallel.py`)

중요한 점은 OpenEvolve가 이론상 분산 연구 시스템처럼 보일 수 있지만, 실제 구현은 꽤 실용적인 Python 병렬 처리다. 이 점은 HypoEvolve에서도 장점이다. 처음부터 거대한 분산 시스템을 짤 필요 없이, iteration worker + snapshot + evaluator 조합으로 충분히 시작할 수 있다. (`openevolve/openevolve/process_parallel.py`)

---

## 11. LLM 계층

`LLMEnsemble`은 모델 여러 개를 확률적으로 섞어서 사용할 수 있게 해준다. 그리고 `OpenAILLM`은 실제 OpenAI-compatible endpoint를 호출하는 어댑터다. 여기에는 reasoning model 분기, seed 전달, retry, timeout, manual mode까지 들어 있다. (`openevolve/openevolve/llm/ensemble.py`, `openevolve/openevolve/llm/openai.py`)

즉 OpenEvolve는 특정 모델에 고정된 시스템이 아니라, **OpenAI-compatible generation backend** 위에 올라가는 진화 루프다. HypoEvolve도 이 층위를 재활용할 수 있다. 코드 생성 대신 hypothesis critique/generation에 같은 backend를 쓸 수 있다. (`openevolve/openevolve/llm/openai.py`)

---

## 12. Config 구조

OpenEvolve는 config를 통해 시스템의 거의 모든 동작을 조절한다.

대표 항목:
- LLM config
- prompt config
- database config
- evaluator config
- evolution trace config
- diff_based_evolution 여부
- max_code_length
- early stopping
- feature dimensions
- number of islands
- migration interval/rate

즉 구현은 코드에 하드코딩되어 있지 않고, 상당 부분 **실험 가능한 연구용 파라미터 시스템**으로 열려 있다. (`openevolve/openevolve/config.py`)

HypoEvolve에서도 같은 설계를 유지하는 게 좋다. 단, 초기에 너무 많은 config를 열기보다는 작은 범위부터 시작해야 한다.

---

## 13. API / CLI 레이어

OpenEvolve는 라이브러리 API와 CLI를 둘 다 제공한다.

- `run_evolution`
- `evolve_function`
- `evolve_algorithm`
- `evolve_code`
- CLI entrypoint `openevolve-run`

이 API들은 결국 초기 후보를 파일 또는 문자열로 받고, evaluator를 파일 또는 callable로 받아, 내부적으로 `OpenEvolve` controller를 실행하는 thin wrapper다. (`openevolve/openevolve/api.py`, `openevolve/openevolve/cli.py`, `openevolve/pyproject.toml`)

HypoEvolve도 같은 식으로 갈 수 있다. 초기에는 `run_hypothesis_evolution(initial_hypothesis, evaluator, ...)` 수준만 있어도 충분하다.

---

## 14. OpenEvolve의 장점

### 14.1 구현 층위가 비교적 분리되어 있다
controller / prompt / evaluator / database / llm / process worker가 구분되어 있어 확장하기 좋다. (`openevolve/openevolve/controller.py`, `openevolve/openevolve/evaluator.py`, `openevolve/openevolve/prompt/sampler.py`, `openevolve/openevolve/database.py`)

### 14.2 quality-diversity 관점이 강하다
MAP-Elites + islands 덕분에 탐색 공간을 넓게 본다. (`openevolve/openevolve/database.py`)

### 14.3 연구 코드치고는 재현성과 tracing을 신경 쓴다
seed propagation, checkpoint, evolution trace 지원이 있다. (`openevolve/openevolve/controller.py`, `openevolve/openevolve/evolution_trace.py`)

### 14.4 실용적인 OpenAI-compatible backend를 쓴다
특정 벤더 종속성이 상대적으로 약하다. (`openevolve/openevolve/llm/openai.py`)

---

## 15. OpenEvolve의 한계와 HypoEvolve로 넘어갈 때의 주의점

### 15.1 구현체는 코드 문자열 중심이다
`Program.code`가 문자열이고, worker 루프도 diff/full rewrite를 텍스트에 적용하는 방식이다. 즉 OpenEvolve는 후보 객체가 본질적으로 문자열이라는 전제를 강하게 가진다. (`openevolve/openevolve/database.py`, `openevolve/openevolve/process_parallel.py`)

이건 HypoEvolve에서 그대로 가져오기보다, **초기 통합은 string-backed serialization**로 하고 점진적으로 ELG-native pipeline으로 넘어가야 한다는 뜻이다.

### 15.2 mutation primitive가 코드 편집에 특화되어 있다
현재의 diff parser, full rewrite, code length 제한 등은 코드용이다. 가설 구조 mutation은 별도 구현이 필요하다. (`openevolve/openevolve/process_parallel.py`, `openevolve/openevolve/utils/code_utils.py`)

### 15.3 evaluator semantics는 외부로 밀려 있다
이건 장점이기도 하지만, 가설 진화에서는 evaluator가 더 어렵다. 코드보다 hypothesis truthfulness/support evaluation이 훨씬 애매하기 때문이다. (`openevolve/openevolve/evaluator.py`)

### 15.4 novelty와 semantic equivalence는 아직 거칠다
embedding + LLM novelty judge 옵션이 있지만, hypothesis equivalence까지 다루기엔 부족하다. (`openevolve/openevolve/database.py`, `openevolve/openevolve/novelty_judge.py`)

---

## 16. HypoEvolve 관점에서 OpenEvolve를 어떻게 재해석해야 하는가

OpenEvolve를 그대로 베끼는 것보다, 아래 층위를 분리해서 참고하는 것이 좋다.

### 재사용 가치가 높은 층위
- controller/orchestration 패턴
- evaluator abstraction
- database / archive / best tracking
- MAP-Elites / islands
- tracing / checkpoint
- OpenAI-compatible LLM backend

### 교체가 필요한 층위
- `Program.code` 중심 candidate representation
- diff/full rewrite 기반 mutation
- 코드 실행 evaluator
- prompt wording (코드 최적화 중심)

즉 HypoEvolve는 OpenEvolve의 **탐색 시스템 골격**은 많이 참고하되, **후보 표현과 mutation 계층은 ELG 중심으로 다시 설계해야 한다**. (`openevolve/openevolve/controller.py`, `openevolve/openevolve/database.py`, `openevolve/openevolve/process_parallel.py`, `openevolve/openevolve/evaluator.py`)

---

## 17. HypoEvolve에 대한 직접적 시사점

1. **후보 표현을 먼저 고정하라**  
   OpenEvolve는 후보를 `Program`으로 고정했기 때문에 전체 루프가 돌아간다. HypoEvolve에서도 ELG가 그 역할을 해야 한다. (`openevolve/openevolve/database.py`)

2. **mutation primitive를 explicit하게 구현하라**  
   OpenEvolve는 mutation content를 LLM에게 맡기지만, HypoEvolve는 ELG mutation primitive와 sampler를 먼저 갖춘 게 강점이다. (`openevolve/openevolve/process_parallel.py`)

3. **evaluation은 나중에 붙여도 된다**  
   OpenEvolve도 evaluator abstraction이 외부에 있기 때문에, HypoEvolve는 ELG core를 먼저 완성하는 전략이 타당하다. (`openevolve/openevolve/evaluator.py`)

4. **다양성을 유지하는 저장소 설계가 중요하다**  
   hypothesis search는 best 하나보다 frontier 관리가 더 중요할 수 있다. OpenEvolve의 MAP-Elites는 좋은 출발점이다. (`openevolve/openevolve/database.py`)

5. **초기 통합은 string-backed가 현실적이다**  
   OpenEvolve 런타임은 문자열 중심이므로, ELG를 바로 object로 꽂으려 하기보다 serialized hypothesis를 intermediate string으로 다루는 편이 안전하다. (`openevolve/openevolve/database.py`, `openevolve/openevolve/process_parallel.py`, `openevolve/openevolve/controller.py`)

---

## 18. 결론

OpenEvolve는 단순한 코드 생성기가 아니라, **LLM 기반 proposal generation + 외부 evaluator + quality-diversity archive**를 결합한 실용적인 진화 탐색 시스템이다. HypoEvolve는 이 구조를 많이 참고할 수 있지만, 그대로 복제하는 것보다 다음처럼 보는 것이 정확하다.

- OpenEvolve = **코드 후보 탐색 엔진**
- HypoEvolve = **가설 후보 탐색 엔진**

둘의 공통점은 탐색 루프이고, 차이는 후보 표현과 evaluator semantics다. 따라서 HypoEvolve는 OpenEvolve의 controller/database/selection 철학을 재사용하면서, mutation과 evaluation을 ELG 중심으로 다시 설계하는 방향이 가장 자연스럽다. (`openevolve/openevolve/controller.py`, `openevolve/openevolve/database.py`, `openevolve/openevolve/process_parallel.py`, `openevolve/openevolve/evaluator.py`)
