# HypoEvolve 인터페이스 문서

## 개요
HypoEvolve는 LLM을 활용한 효율적인 코드 진화 프레임워크입니다. 동기식 처리를 통해 간단하고 직관적인 사용이 가능하며, MAP-Elites 알고리즘을 통한 다양성 보존 기능을 제공합니다.

**주요 특징:**
- LLM 기반 코드 변이 및 재작성
- MAP-Elites 기반 프로그램 데이터베이스 관리
- 동기식 평가 시스템
- 유연한 평가 함수 지원
- 복잡도와 다양성 기반 행동 기술자 (Behavior Descriptors)

## 설치

### Poetry 사용 (권장)
```bash
poetry add hypoevolve
```

### pip 사용
```bash
pip install hypoevolve
```

**의존성:**
- Python ^3.9
- openai ^1.90.0
- pyyaml ^6.0.2
- numpy <2.0

## 빠른 시작

```python
from hypoevolve import HypoEvolve
from hypoevolve.core.config import Config

# MAP-Elites 설정이 포함된 HypoEvolve 인스턴스 생성
config = Config(
    model="gpt-3.5-turbo", 
    max_iterations=10,
    population_size=50,
    elite_ratio=0.2,
    grid_size=10  # MAP-Elites 그리드 크기
)
hypo = HypoEvolve(config)

# 사용자 정의 평가 함수 설정
def custom_evaluator(program, context):
    # 간단한 평가 로직
    return {"score": 0.8, "accuracy": 0.9}

hypo.set_custom_evaluator(custom_evaluator)

# 코드 진화 실행
initial_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

best_program = hypo.evolve(
    initial_code=initial_code,
    problem_description="피보나치 함수 최적화",
    max_iterations=10
)

print(f"최고 점수: {best_program.score}")
print(f"최적화된 코드:\n{best_program.code}")

# 통계 조회
stats = hypo.database.stats()
print(f"데이터베이스 크기: {stats['size']}")
print(f"최고 점수: {stats['best_score']}")

# 최고 프로그램 조회
best = hypo.database.get_best()
print(f"최고 프로그램 점수: {best.score}")

```

---

## 클래스별 인터페이스

### 1. HypoEvolve 클래스

#### 클래스 선언
```python
from hypoevolve import HypoEvolve
from hypoevolve.core.config import Config

# 기본 설정으로 생성
hypo = HypoEvolve()

# 사용자 정의 설정으로 생성
config = Config(model="gpt-4", max_iterations=20, population_size=30)
hypo = HypoEvolve(config)
```

#### 메서드

##### `__init__(config: Optional[Config] = None)`
HypoEvolve 인스턴스를 초기화합니다.

**매개변수:**
- `config`: 설정 객체 (선택사항, 기본값: Config())

**입력 예시:**
```python
config = Config(model="gpt-3.5-turbo", max_iterations=15)
hypo = HypoEvolve(config)
```

**출력:** HypoEvolve 인스턴스

---

##### `set_evaluation_function(evaluation_file: str, context: Optional[Dict[str, Any]] = None) -> None`
파일 기반 평가 함수를 설정합니다.

**매개변수:**
- `evaluation_file`: 평가 함수 파일 경로
- `context`: 평가 컨텍스트 딕셔너리 (선택사항)

**입력 예시:**
```python
hypo.set_evaluation_function("eval_function.py", {"test_mode": True})
```

**출력:** None

---

##### `set_custom_evaluator(custom_evaluator: Callable) -> None`
사용자 정의 평가 함수를 설정합니다.

**매개변수:**
- `custom_evaluator`: 사용자 정의 평가 함수

**입력 예시:**
```python
def my_evaluator(program, context):
    return {"score": 0.85, "performance": "good"}

hypo.set_custom_evaluator(my_evaluator)
```

**출력:** None

---

##### `evolve(initial_code: str, problem_description: str = "", max_iterations: Optional[int] = None, context: Optional[Dict[str, Any]] = None) -> Optional[Program]`
코드 진화 프로세스를 실행합니다.

**매개변수:**
- `initial_code`: 초기 코드
- `problem_description`: 문제 설명 (선택사항)
- `max_iterations`: 최대 반복 횟수 (선택사항)
- `context`: 추가 컨텍스트 (선택사항)

**입력 예시:**
```python
result = hypo.evolve(
    initial_code="def add(a, b): return a + b",
    problem_description="덧셈 함수 최적화",
    max_iterations=5
)
```

**출력 예시:**
```python
Program(
    id="prog_123",
    code="def add(a, b):\n    \"\"\"최적화된 덧셈 함수\"\"\"\n    return a + b",
    score=0.92,
    language="python"
)
```

**참고**: 진화 통계 조회는 `hypo.database.stats()`, 최고 프로그램 조회는 `hypo.database.get_best()` 메서드를 사용하세요.

---

### 2. Config 클래스

#### 클래스 선언
```python
from hypoevolve.core.config import Config

# 기본 설정
config = Config()

# 사용자 정의 설정
config = Config(
    model="gpt-4",
    max_iterations=50,
    population_size=100,
    temperature=0.8
)
```

#### 메서드

##### `__init__(**kwargs)`
설정 객체를 초기화합니다.

**주요 매개변수:**
- `model`: LLM 모델명 (기본값: "gpt-3.5-turbo")
- `max_iterations`: 최대 반복 횟수 (기본값: 100)
- `population_size`: 인구 크기 (기본값: 50)
- `temperature`: 온도 설정 (기본값: 0.7)

---

##### `to_dict() -> Dict[str, Any]`
설정을 딕셔너리로 변환합니다.

**입력 예시:**
```python
config_dict = config.to_dict()
```

**출력 예시:**
```python
{
    "model": "gpt-3.5-turbo",
    "max_iterations": 100,
    "population_size": 50,
    "temperature": 0.7
}
```

---

##### `from_dict(data: Dict[str, Any]) -> Config` (클래스 메서드)
딕셔너리에서 설정 객체를 생성합니다.

**입력 예시:**
```python
data = {"model": "gpt-4", "max_iterations": 20}
config = Config.from_dict(data)
```

---

### 3. Program 클래스 (MAP-Elites 기능 포함)

#### 클래스 선언
```python
from hypoevolve.core.program import Program

# 기본 프로그램 생성
program = Program(code="def hello(): print('Hello')", language="python")

# 복잡도와 다양성 계산
program.complexity = program.calculate_complexity()
program.diversity = program.calculate_diversity(reference_codes)
```

#### 새로운 속성 (MAP-Elites)
- `complexity: float` - 코드 복잡도 (0.0-1.0 범위)
- `diversity: float` - 다양성 측정값 (0.0-1.0 범위)

#### 새로운 메서드

##### `calculate_complexity() -> float`
코드의 복잡도를 계산합니다.

**입력 예시:**
```python
program = Program(code="def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)")
complexity = program.calculate_complexity()
```

**출력 예시:**
```python
0.45  # 복잡도 점수 (0.0-1.0)
```

---

##### `calculate_diversity(reference_codes: List[str]) -> float`
다른 코드들과의 다양성을 계산합니다.

**매개변수:**
- `reference_codes`: 비교 대상 코드 리스트

**입력 예시:**
```python
reference_codes = ["def add(a, b): return a + b", "def multiply(a, b): return a * b"]
diversity = program.calculate_diversity(reference_codes)
```

**출력 예시:**
```python
0.78  # 다양성 점수 (0.0-1.0)
```

---

##### `get_behavior_descriptor() -> Tuple[int, int]`
MAP-Elites 그리드를 위한 행동 기술자를 반환합니다.

**입력 예시:**
```python
descriptor = program.get_behavior_descriptor()
```

**출력 예시:**
```python
(4, 7)  # (복잡도 빈, 다양성 빈)
```

---

### 4. ProgramDatabase 클래스 (MAP-Elites 기능 포함)

#### 클래스 선언
```python
from hypoevolve.core.program import ProgramDatabase

# MAP-Elites 설정으로 데이터베이스 생성
db = ProgramDatabase(
    population_size=50,
    elite_ratio=0.2,
    grid_size=10  # 10x10 MAP-Elites 그리드
)
```

#### 새로운 속성 (MAP-Elites)
- `grid_size: int` - MAP-Elites 그리드 크기
- `map_elites_grid: Dict[Tuple[int, int], str]` - 그리드 셀과 프로그램 ID 매핑
- `grid_stats: Dict[str, Any]` - 그리드 통계 정보

#### 주요 메서드

##### `sample_parent() -> Optional[Program]` (개선됨)
엘리트와 다양성을 고려한 부모 프로그램 샘플링입니다.

**입력 예시:**
```python
parent = db.sample_parent()
```

**출력 예시:**
```python
Program(id="prog_elite", score=0.95, complexity=0.4, diversity=0.6, ...)
```

---

##### `get_best() -> Optional[Program]`
현재 데이터베이스에서 가장 높은 점수를 가진 프로그램을 반환합니다.

**입력 예시:**
```python
best = db.get_best()
```

**출력 예시:**
```python
Program(id="prog_best", score=0.98, code="def optimized(): ...", ...)
```

---

##### `get_top_programs(n: int = 10) -> List[Program]`
상위 n개의 프로그램을 점수 순으로 반환합니다.

**매개변수:**
- `n`: 반환할 프로그램 수

**입력 예시:**
```python
top_programs = db.get_top_programs(5)
```

**출력 예시:**
```python
[
    Program(id="prog_1", score=0.98, ...),
    Program(id="prog_2", score=0.95, ...),
    Program(id="prog_3", score=0.92, ...),
    Program(id="prog_4", score=0.89, ...),
    Program(id="prog_5", score=0.87, ...)
]
```

---

##### `stats() -> Dict[str, Any]` (개선됨)
MAP-Elites 정보를 포함한 데이터베이스 통계를 반환합니다.

**입력 예시:**
```python
stats = db.stats()
```

**출력 예시:**
```python
{
    "size": 45,
    "generation": 8,
    "best_score": 0.96,
    "avg_score": 0.73,
    "min_score": 0.12,
    "grid_coverage": 0.28,
    "occupied_cells": 28,
    "total_cells": 100
}
```

---

### 5. FunctionEvaluator 클래스

#### 클래스 선언
```python
from hypoevolve.evaluation.evaluator import FunctionEvaluator

evaluator = FunctionEvaluator(
    evaluation_file="my_eval.py",
    timeout=30,
    max_retries=3
)
```

#### 메서드

##### `__init__(evaluation_file: str, timeout: int = 30, max_retries: int = 3, temp_dir: Optional[str] = None)`
파일 기반 평가자를 초기화합니다.

**매개변수:**
- `evaluation_file`: 평가 함수 파일 경로
- `timeout`: 타임아웃 (초)
- `max_retries`: 최대 재시도 횟수
- `temp_dir`: 임시 디렉토리 경로

---

##### `evaluate_program(program: Program, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`
프로그램을 평가합니다.

**입력 예시:**
```python
program = Program(code="def add(a, b): return a + b", language="python")
result = evaluator.evaluate_program(program, {"test_mode": True})
```

**출력 예시:**
```python
{
    "success": True,
    "metrics": {
        "score": 0.85,
        "accuracy": 0.9,
        "performance": 0.8
    },
    "execution_time": 0.05
}
```

---

### 6. SimpleEvaluator 클래스

#### 클래스 선언
```python
from hypoevolve.evaluation.evaluator import SimpleEvaluator

def my_evaluator(program, context):
    return {"score": 0.8}

evaluator = SimpleEvaluator(my_evaluator)
```

#### 메서드

##### `__init__(custom_evaluator: Callable)`
사용자 정의 평가자를 초기화합니다.

**매개변수:**
- `custom_evaluator`: 사용자 정의 평가 함수

---

##### `evaluate_program(program: Program, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`
프로그램을 평가합니다.

**입력 예시:**
```python
program = Program(code="def multiply(a, b): return a * b", language="python")
result = evaluator.evaluate_program(program)
```

**출력 예시:**
```python
{
    "success": True,
    "metrics": {"score": 0.8},
    "execution_time": 0.02
}
```

---

### 7. LLMClient 클래스

#### 클래스 선언
```python
from hypoevolve.llm.client import LLMClient
from hypoevolve.core.config import Config

config = Config(model="gpt-3.5-turbo")
client = LLMClient(config)
```

#### 메서드

##### `__init__(config: Config)`
LLM 클라이언트를 초기화합니다.

**매개변수:**
- `config`: 설정 객체

---

##### `generate_mutation(current_code: str, inspirations: List[str] = None, context: str = "") -> str`
코드 변이를 생성합니다.

**입력 예시:**
```python
mutation = client.generate_mutation(
    current_code="def add(a, b): return a + b",
    context="덧셈 함수 개선"
)
```

**출력 예시:**
```python
"""<<<<<<< SEARCH
def add(a, b):
    return a + b
=======
def add(a, b):
    \"\"\"두 수를 더하는 함수\"\"\"
    return a + b
>>>>>>> REPLACE"""
```

---

##### `generate_full_rewrite(current_code: str, inspirations: List[str] = None, context: str = "") -> str`
완전한 코드 재작성을 생성합니다.

**입력 예시:**
```python
rewrite = client.generate_full_rewrite(
    current_code="def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
    context="팩토리얼 함수 최적화"
)
```

**출력 예시:**
```python
"""def factorial(n):
    \"\"\"최적화된 팩토리얼 함수\"\"\"
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result"""
```

---

## 사용 예시

### 1. 파일 기반 평가 함수 사용

```python
# eval_function.py 파일 생성
"""
def evaluate(program_path, context):
    # 프로그램 평가 로직
    return {
        "score": 0.9,
        "accuracy": 0.95,
        "performance": 0.85
    }
"""

# HypoEvolve 사용
from hypoevolve import HypoEvolve

hypo = HypoEvolve()
hypo.set_evaluation_function("eval_function.py", {"test_mode": True})

result = hypo.evolve(
    initial_code="def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    problem_description="피보나치 최적화"
)
```

### 2. 사용자 정의 평가 함수 사용

```python
def performance_evaluator(program, context):
    # 성능 기반 평가
    code_length = len(program.code)
    complexity_score = 1.0 - (code_length / 1000)  # 간단한 복잡도 측정
    
    return {
        "score": max(0.1, complexity_score),
        "code_length": code_length,
        "complexity": complexity_score
    }

hypo = HypoEvolve()
hypo.set_custom_evaluator(performance_evaluator)

result = hypo.evolve(
    initial_code="def sort_list(lst): return sorted(lst)",
    problem_description="정렬 함수 최적화"
)
```

### 3. YAML 설정 파일 사용

```yaml
# config.yaml
model: "gpt-4"
max_iterations: 30
population_size: 75
temperature: 0.8
max_tokens: 2048
timeout: 60
save_interval: 5
output_dir: "./evolution_results"
```

```python
from hypoevolve.core.config import Config
from hypoevolve import HypoEvolve

config = Config.from_yaml("config.yaml")
hypo = HypoEvolve(config)

# 평가 함수 설정 및 진화 실행
hypo.set_custom_evaluator(lambda p, c: {"score": 0.8})
result = hypo.evolve("def example(): pass")
```

### 4. 고급 기능 - 데이터베이스 조작

```python
# 프로그램 데이터베이스 직접 조작
db = hypo.database

# 모든 프로그램 조회
all_programs = list(db.programs.values())

# 상위 프로그램들 조회
top_programs = db.get_top_programs(5)

# 통계 정보
stats = db.stats()
print(f"데이터베이스 크기: {stats['size']}")
print(f"최고 점수: {stats['best_score']}")
print(f"평균 점수: {stats['avg_score']}")

# 데이터베이스 저장/로드
db.save("my_database.json")
db.load("my_database.json")
```

### 5. 진화 과정 모니터링

```python
from hypoevolve import HypoEvolve
from hypoevolve.core.config import Config

config = Config(max_iterations=20, save_interval=5)
hypo = HypoEvolve(config)

def monitoring_evaluator(program, context):
    score = len(program.code) / 100  # 간단한 점수 계산
    print(f"프로그램 평가: ID={program.id}, 점수={score:.3f}")
    return {"score": score}

hypo.set_custom_evaluator(monitoring_evaluator)

# 진화 실행
result = hypo.evolve("def initial(): pass")

# 최종 통계
```

---

## 완전한 작업 예시

```python
from hypoevolve import HypoEvolve
from hypoevolve.core.config import Config

# 1. 설정 생성
config = Config(
    model="gpt-3.5-turbo",
    max_iterations=15,
    population_size=30,
    temperature=0.7,
    output_dir="./results"
)

# 2. HypoEvolve 인스턴스 생성
hypo = HypoEvolve(config)

# 3. 평가 함수 정의
def code_quality_evaluator(program, context):
    code = program.code
    
    # 간단한 코드 품질 메트릭
    has_docstring = '"""' in code or "'''" in code
    has_type_hints = ':' in code and '->' in code
    line_count = len(code.split('\n'))
    
    quality_score = 0.5  # 기본 점수
    if has_docstring:
        quality_score += 0.2
    if has_type_hints:
        quality_score += 0.2
    if line_count < 20:  # 간결함 보너스
        quality_score += 0.1
    
    return {
        "score": min(1.0, quality_score),
        "has_docstring": has_docstring,
        "has_type_hints": has_type_hints,
        "line_count": line_count
    }

# 4. 평가 함수 설정
hypo.set_custom_evaluator(code_quality_evaluator)

# 5. 초기 코드 정의
initial_code = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
"""

# 6. 진화 실행
print("코드 진화 시작...")
best_program = hypo.evolve(
    initial_code=initial_code,
    problem_description="피보나치 함수의 코드 품질 향상",
    max_iterations=15
)

# 7. 결과 출력
if best_program:
    print(f"\n=== 진화 완료 ===")
    print(f"최고 점수: {best_program.score:.3f}")
    print(f"세대: {best_program.generation}")
    print(f"\n최적화된 코드:")
    print(best_program.code)
    
    # 메트릭 정보
    if best_program.metrics:
        print(f"\n상세 메트릭:")
        for key, value in best_program.metrics.items():
            print(f"  {key}: {value}")
else:
    print("진화 실패: 결과를 생성할 수 없습니다.")
```