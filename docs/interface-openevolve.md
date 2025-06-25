# OpenEvolve 인터페이스 문서

OpenEvolve 패키지의 모든 클래스와 주요 퍼블릭 메서드들의 사용법을 설명합니다.

## 1. OpenEvolve (메인 컨트롤러)

### 클래스 개요
`openevolve.controller.OpenEvolve`는 진화형 코드 최적화 시스템의 메인 컨트롤러입니다.

### 초기화

```python
from openevolve import OpenEvolve
from openevolve.config import Config

# 기본 설정으로 초기화
openevolve = OpenEvolve(
    initial_program_path="my_program.py",
    evaluation_file="evaluate.py"
)

# 커스텀 설정으로 초기화
config = Config()
config.llm.api_key = "your-openai-api-key"
config.max_iterations = 100

openevolve = OpenEvolve(
    initial_program_path="my_program.py",
    evaluation_file="evaluate.py",
    config=config,
    output_dir="./results"
)
```

### run(iterations, target_score)

코드 진화 프로세스를 실행합니다.

```python
# 최대 50회 반복으로 진화 실행
best_program = await openevolve.run(iterations=50)

# 목표 점수 0.95에 도달할 때까지 진화 실행
best_program = await openevolve.run(target_score=0.95)

print(f"최고 점수: {best_program.metrics}")
print(f"최적화된 코드:\n{best_program.code}")

# 출력:
# 최고 점수: {'accuracy': 0.95, 'efficiency': 0.88}
# 최적화된 코드:
# def optimized_function(x):
#     return x * 2 + 1
```

## 2. Config (설정 관리)

### 클래스 개요
`openevolve.config.Config`는 OpenEvolve의 모든 설정을 관리합니다.

### 기본 초기화

```python
from openevolve.config import Config

config = Config()
print(f"최대 반복 횟수: {config.max_iterations}")
print(f"인구 크기: {config.database.population_size}")

# 출력:
# 최대 반복 횟수: 10000
# 인구 크기: 1000
```

### from_yaml(path)

YAML 파일에서 설정을 로드합니다.

```python
config = Config.from_yaml("config.yaml")
print(f"API 키: {config.llm.api_key}")
print(f"모델: {config.llm.models[0].name}")

# 출력:
# API 키: sk-your-api-key
# 모델: gpt-4
```

### to_yaml(path)

설정을 YAML 파일로 저장합니다.

```python
config.to_yaml("output_config.yaml")
# 출력: None (파일 저장 완료)
# 파일 내용:
# max_iterations: 10000
# llm:
#   api_base: https://api.openai.com/v1
#   temperature: 0.7
```

## 3. Program (프로그램 표현)

### 클래스 개요
`openevolve.database.Program`은 진화 과정의 프로그램을 나타냅니다.

### 기본 초기화

```python
from openevolve.database import Program

program = Program(
    id="prog-001",
    code="def hello(): return 'world'",
    language="python"
)
print(f"프로그램 ID: {program.id}")
print(f"생성 시간: {program.timestamp}")

# 출력:
# 프로그램 ID: prog-001
# 생성 시간: 1699123456.789
```

### to_dict()

프로그램을 딕셔너리로 변환합니다.

```python
program_dict = program.to_dict()
print(program_dict)

# 출력:
# {
#     'id': 'prog-001',
#     'code': "def hello(): return 'world'",
#     'language': 'python',
#     'parent_id': None,
#     'generation': 0,
#     'timestamp': 1699123456.789,
#     'iteration_found': 0,
#     'metrics': {},
#     'complexity': 0.0,
#     'diversity': 0.0,
#     'metadata': {},
#     'artifacts_json': None,
#     'artifact_dir': None
# }
```

### from_dict(data)

딕셔너리에서 프로그램을 생성합니다.

```python
data = {
    'id': 'prog-002',
    'code': 'def add(a, b): return a + b',
    'language': 'python',
    'metrics': {'score': 0.85}
}
program = Program.from_dict(data)
print(f"코드: {program.code}")
print(f"메트릭: {program.metrics}")

# 출력:
# 코드: def add(a, b): return a + b
# 메트릭: {'score': 0.85}
```

## 4. ProgramDatabase (프로그램 데이터베이스)

### 클래스 개요
`openevolve.database.ProgramDatabase`는 MAP-Elites와 아일랜드 기반 진화를 지원하는 프로그램 데이터베이스입니다.

### 초기화

```python
from openevolve.database import ProgramDatabase
from openevolve.config import DatabaseConfig

config = DatabaseConfig(
    population_size=500,
    num_islands=3,
    archive_size=50
)
db = ProgramDatabase(config)
print(f"인구 크기: {db.config.population_size}")
print(f"아일랜드 수: {len(db.islands)}")

# 출력:
# 인구 크기: 500
# 아일랜드 수: 3
```

### add(program, iteration, target_island)

프로그램을 데이터베이스에 추가합니다.

```python
program = Program(
    id="prog-001",
    code="def test(): pass",
    metrics={'score': 0.7}
)
program_id = db.add(program, iteration=1)
print(f"추가된 프로그램 ID: {program_id}")
print(f"총 프로그램 수: {len(db.programs)}")

# 출력:
# 추가된 프로그램 ID: prog-001
# 총 프로그램 수: 1
```

### get_best_program(metric)

최고 성능 프로그램을 반환합니다.

```python
best = db.get_best_program()
if best:
    print(f"최고 점수: {best.metrics}")
    print(f"코드 길이: {len(best.code)}")
else:
    print("프로그램이 없습니다")

# 출력:
# 최고 점수: {'score': 0.95, 'efficiency': 0.88}
# 코드 길이: 156
```

### sample()

부모 프로그램과 영감 프로그램들을 샘플링합니다.

```python
parent, inspirations = db.sample()
print(f"선택된 부모 점수: {parent.metrics}")
print(f"영감 프로그램 수: {len(inspirations)}")
for i, prog in enumerate(inspirations):
    print(f"영감 {i+1}: {prog.metrics}")

# 출력:
# 선택된 부모 점수: {'score': 0.85}
# 영감 프로그램 수: 5
# 영감 1: {'score': 0.95}
# 영감 2: {'score': 0.87}
# 영감 3: {'score': 0.73}
```

### set_current_island(island_idx)

현재 진화 중인 아일랜드를 설정합니다.

```python
print(f"현재 아일랜드: {db.current_island}")
db.set_current_island(2)
print(f"변경된 아일랜드: {db.current_island}")

# 출력:
# 현재 아일랜드: 0
# 변경된 아일랜드: 2
```

### save(path) / load(path)

데이터베이스를 파일로 저장하거나 로드합니다.

```python
# 저장
db.save("database.json")
print("데이터베이스 저장 완료")

# 로드
db.load("database.json")
print(f"로드된 프로그램 수: {len(db.programs)}")

# 출력:
# 데이터베이스 저장 완료
# 로드된 프로그램 수: 25
```

## 5. Evaluator (프로그램 평가자)

### 클래스 개요
`openevolve.evaluator.Evaluator`는 프로그램을 평가하고 점수를 할당합니다.

### 초기화

```python
from openevolve.evaluator import Evaluator
from openevolve.config import EvaluatorConfig

config = EvaluatorConfig(
    timeout=60,
    max_retries=3,
    parallel_evaluations=4
)
evaluator = Evaluator(
    config=config,
    evaluation_file="evaluate.py"
)
```

### evaluate_program(program_code, program_id)

프로그램을 평가합니다.

```python
code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""

metrics = await evaluator.evaluate_program(code, "prog-001")
print(f"평가 결과: {metrics}")

# 출력:
# 평가 결과: {
#     'correctness': 1.0,
#     'efficiency': 0.75,
#     'readability': 0.85,
#     'overall_score': 0.87
# }
```

### evaluate_multiple(programs)

여러 프로그램을 병렬로 평가합니다.

```python
programs = [
    ("def add(a, b): return a + b", "prog-001"),
    ("def sub(a, b): return a - b", "prog-002")
]

results = await evaluator.evaluate_multiple(programs)
for i, result in enumerate(results):
    print(f"프로그램 {i+1}: {result}")

# 출력:
# 프로그램 1: {'score': 0.95}
# 프로그램 2: {'score': 0.88}
```

## 6. LLMEnsemble (LLM 앙상블)

### 클래스 개요
`openevolve.llm.ensemble.LLMEnsemble`는 여러 LLM 모델을 앙상블로 관리합니다.

### 초기화

```python
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.config import LLMModelConfig

models = [
    LLMModelConfig(name="gpt-4", weight=0.7),
    LLMModelConfig(name="gpt-3.5-turbo", weight=0.3)
]
ensemble = LLMEnsemble(models)
print(f"앙상블 모델 수: {len(ensemble.models)}")

# 출력:
# 앙상블 모델 수: 2
```

### generate(prompt, **kwargs)

프롬프트에서 텍스트를 생성합니다.

```python
prompt = "다음 함수를 최적화하세요: def slow_func(n): return sum(range(n))"
response = await ensemble.generate(prompt)
print(f"생성된 응답: {response[:100]}...")

# 출력:
# 생성된 응답: 다음은 최적화된 버전입니다:
# 
# def fast_func(n):
#     return n * (n - 1) // 2...
```

### generate_multiple(prompt, n)

동일한 프롬프트로 여러 응답을 병렬 생성합니다.

```python
responses = await ensemble.generate_multiple(prompt, n=3)
print(f"생성된 응답 수: {len(responses)}")
for i, response in enumerate(responses):
    print(f"응답 {i+1}: {response[:50]}...")

# 출력:
# 생성된 응답 수: 3
# 응답 1: 다음은 최적화된 버전입니다: def fast_func(n):...
# 응답 2: 성능을 개선하려면: def optimized_func(n):...
# 응답 3: 더 효율적인 구현: def efficient_func(n):...
```

## 7. OpenAILLM (OpenAI LLM 클라이언트)

### 클래스 개요
`openevolve.llm.openai.OpenAILLM`은 OpenAI API와 호환되는 LLM 클라이언트입니다.

### 초기화

```python
from openevolve.llm.openai import OpenAILLM
from openevolve.config import LLMModelConfig

model_config = LLMModelConfig(
    name="gpt-4",
    api_key="sk-your-api-key",
    temperature=0.7,
    max_tokens=2048
)
llm = OpenAILLM(model_config)
```

### generate_with_context(system_message, messages)

시스템 메시지와 대화 컨텍스트를 사용하여 텍스트를 생성합니다.

```python
system_message = "당신은 코드 최적화 전문가입니다."
messages = [
    {"role": "user", "content": "이 함수를 최적화해주세요: def slow(n): return sum(i for i in range(n))"}
]

response = await llm.generate_with_context(system_message, messages)
print(f"응답: {response}")

# 출력:
# 응답: 다음은 최적화된 버전입니다:
# 
# def fast(n):
#     return n * (n - 1) // 2
# 
# 이 버전은 O(1) 시간 복잡도를 가집니다.
```

## 8. PromptSampler (프롬프트 샘플러)

### 클래스 개요
`openevolve.prompt.sampler.PromptSampler`는 코드 진화를 위한 프롬프트를 생성합니다.

### 초기화

```python
from openevolve.prompt.sampler import PromptSampler
from openevolve.config import PromptConfig

config = PromptConfig(
    num_top_programs=5,
    num_diverse_programs=3
)
sampler = PromptSampler(config)
```

### build_prompt(current_program, program_metrics, ...)

LLM을 위한 프롬프트를 구성합니다.

```python
current_code = "def add(a, b): return a + b"
metrics = {"accuracy": 0.85, "efficiency": 0.70}
top_programs = [
    {"code": "def fast_add(x, y): return x + y", "metrics": {"accuracy": 0.95}}
]

prompt = sampler.build_prompt(
    current_program=current_code,
    program_metrics=metrics,
    top_programs=top_programs,
    language="python",
    evolution_round=5
)

print(f"시스템 메시지: {prompt['system'][:100]}...")
print(f"사용자 메시지: {prompt['user'][:100]}...")

# 출력:
# 시스템 메시지: You are an expert programmer tasked with evolving code...
# 사용자 메시지: Current program metrics:
# - accuracy: 0.8500
# - efficiency: 0.7000...
```

### set_templates(system_template, user_template)

커스텀 템플릿을 설정합니다.

```python
sampler.set_templates(
    system_template="custom_system",
    user_template="custom_user"
)
print("커스텀 템플릿 설정 완료")

# 출력:
# 커스텀 템플릿 설정 완료
```

## 9. EvaluationResult (평가 결과)

### 클래스 개요
`openevolve.evaluation_result.EvaluationResult`는 프로그램 평가 결과를 나타냅니다.

### 초기화

```python
from openevolve.evaluation_result import EvaluationResult

# 메트릭만 포함
result = EvaluationResult(
    metrics={"accuracy": 0.95, "speed": 0.88}
)

# 메트릭과 아티팩트 포함
result_with_artifacts = EvaluationResult(
    metrics={"score": 0.92},
    artifacts={
        "output_log": "Program executed successfully",
        "debug_info": b"binary debug data"
    }
)
```

### has_artifacts()

아티팩트 포함 여부를 확인합니다.

```python
print(f"아티팩트 포함: {result.has_artifacts()}")
print(f"아티팩트 키: {result_with_artifacts.get_artifact_keys()}")

# 출력:
# 아티팩트 포함: False
# 아티팩트 키: ['output_log', 'debug_info']
```

### get_total_artifact_size()

전체 아티팩트 크기를 반환합니다.

```python
size = result_with_artifacts.get_total_artifact_size()
print(f"총 아티팩트 크기: {size} 바이트")

# 출력:
# 총 아티팩트 크기: 45 바이트
```

## 10. 코드 유틸리티 함수들

### apply_diff(original_code, diff_text)

SEARCH/REPLACE 형식의 diff를 적용합니다.

```python
from openevolve.utils.code_utils import apply_diff

original = "def add(a, b): return a + b"
diff = """<<<<<<< SEARCH
def add(a, b): return a + b
=======
def add(a, b):
    \"\"\"두 수를 더합니다\"\"\"
    return a + b
>>>>>>> REPLACE"""

result = apply_diff(original, diff)
print(result)

# 출력:
# def add(a, b):
#     """두 수를 더합니다"""
#     return a + b
```

### extract_diffs(diff_text)

diff 텍스트에서 diff 블록들을 추출합니다.

```python
from openevolve.utils.code_utils import extract_diffs

diff_blocks = extract_diffs(diff)
print(f"diff 블록 수: {len(diff_blocks)}")
for i, (search, replace) in enumerate(diff_blocks):
    print(f"블록 {i+1}: '{search}' -> '{replace}'")

# 출력:
# diff 블록 수: 1
# 블록 1: 'def add(a, b): return a + b' -> 'def add(a, b):
#     """두 수를 더합니다"""
#     return a + b'
```

### parse_full_rewrite(llm_response, language)

LLM 응답에서 완전한 코드 재작성을 추출합니다.

```python
from openevolve.utils.code_utils import parse_full_rewrite

response = """다음은 최적화된 코드입니다:

```python
def optimized_function(x):
    return x * 2 + 1
```

이 버전이 더 효율적입니다."""

code = parse_full_rewrite(response, "python")
print(f"추출된 코드:\n{code}")

# 출력:
# 추출된 코드:
# def optimized_function(x):
#     return x * 2 + 1
```

### calculate_edit_distance(code1, code2)

두 코드 간의 편집 거리를 계산합니다.

```python
from openevolve.utils.code_utils import calculate_edit_distance

code1 = "def add(a, b): return a + b"
code2 = "def add(x, y): return x + y"
code3 = "def multiply(a, b): return a * b"

print(f"유사한 코드 거리: {calculate_edit_distance(code1, code2)}")
print(f"다른 코드 거리: {calculate_edit_distance(code1, code3)}")

# 출력:
# 유사한 코드 거리: 4
# 다른 코드 거리: 12
```

### extract_code_language(code)

코드의 프로그래밍 언어를 감지합니다.

```python
from openevolve.utils.code_utils import extract_code_language

python_code = "def hello(): return 'world'"
java_code = "public class Hello { public static void main() {} }"

print(f"Python 코드: {extract_code_language(python_code)}")
print(f"Java 코드: {extract_code_language(java_code)}")

# 출력:
# Python 코드: python
# Java 코드: java
```

## 11. CLI (명령줄 인터페이스)

### 기본 사용법

```bash
# 기본 실행
python -m openevolve.cli my_program.py evaluate.py

# 설정 파일 사용
python -m openevolve.cli my_program.py evaluate.py --config config.yaml

# 반복 횟수 제한
python -m openevolve.cli my_program.py evaluate.py --iterations 50

# 목표 점수 설정
python -m openevolve.cli my_program.py evaluate.py --target-score 0.95

# 체크포인트에서 재시작
python -m openevolve.cli my_program.py evaluate.py --checkpoint ./results/checkpoints/checkpoint_100
```

### 출력 예시

```
OpenEvolve - Evolutionary coding agent
Using API base: https://api.openai.com/v1
Using primary model: gpt-4

Evolution complete!
Best program metrics:
  accuracy: 0.9500
  efficiency: 0.8800
  overall_score: 0.9150

Latest checkpoint saved at: ./openevolve_output/checkpoints/checkpoint_50
To resume, use: --checkpoint ./openevolve_output/checkpoints/checkpoint_50
``` 