# HypoEvolve 진화 알고리즘 상세 분석

## 🧬 개요

HypoEvolve는 **LLM 기반 코드 진화 알고리즘**으로, **MAP-Elites 알고리즘**에 엘리트 보존 전략을 결합한 하이브리드 접근법을 사용합니다. 이 시스템은 대규모 언어 모델의 언어 이해 능력을 활용하여 코드를 의미론적으로 진화시키며, 복잡도와 다양성을 기반으로 한 행동 공간에서 프로그램을 관리합니다.

**핵심 특징:** HypoEvolve는 **실제 MAP-Elites 그리드**를 사용하여 복잡도와 다양성 차원에서 프로그램 다양성을 보존하는 **MAP-Elites 기반 진화 알고리즘**입니다.

## 🏗️ 전체 아키텍처

### 핵심 구성 요소

1. **Controller** (`hypoevolve/core/controller.py`): 진화 프로세스 제어
2. **Program** (`hypoevolve/core/program.py`): MAP-Elites 기반 프로그램 표현 및 데이터베이스
3. **LLM Client** (`hypoevolve/llm/client.py`): 언어 모델과의 통신
4. **Evaluator** (`hypoevolve/evaluation/`): 프로그램 성능 평가
5. **Config** (`hypoevolve/core/config.py`): MAP-Elites 설정 관리

## 🗺️ MAP-Elites 알고리즘 구현

### 행동 기술자 (Behavior Descriptors)

HypoEvolve는 2차원 행동 공간을 사용합니다:

1. **복잡도 (Complexity)**: 코드의 구조적 복잡성 (0.0-1.0)
   - 라인 수, 문자 수 기반 정규화된 복잡도
   - `complexity = line_complexity * 0.4 + char_complexity * 0.6`

2. **다양성 (Diversity)**: 다른 프로그램들과의 차별성 (0.0-1.0)
   - 문자 집합 기반 유사도의 역수
   - `diversity = 1.0 - max_similarity`

### MAP-Elites 그리드 구조

```python
class ProgramDatabase:
    def __init__(self, population_size: int = 50, elite_ratio: float = 0.2, grid_size: int = 10):
        self.map_elites_grid: Dict[Tuple[int, int], str] = {}  # (complexity_bin, diversity_bin) -> program_id
        self.grid_stats = {
            "total_cells": grid_size * grid_size,
            "occupied_cells": 0,
            "coverage": 0.0
        }
```

**그리드 좌표 계산:**
```python
def get_behavior_descriptor(self) -> Tuple[int, int]:
    complexity_bin = min(int(self.complexity * 10), 9)  # 0-9
    diversity_bin = min(int(self.diversity * 10), 9)    # 0-9
    return (complexity_bin, diversity_bin)
```

### MAP-Elites 업데이트 메커니즘

```python
def _update_map_elites_grid(self, program: Program) -> None:
    behavior = program.get_behavior_descriptor()
    
    # 해당 셀이 비어있거나, 새 프로그램이 더 좋은 점수를 가진 경우
    if (behavior not in self.map_elites_grid or 
        program.score > self.programs[self.map_elites_grid[behavior]].score):
        self.map_elites_grid[behavior] = program.id
```

**특징:**
- 각 그리드 셀에는 해당 행동 특성을 가진 **최고 성능 프로그램만** 보존
- 동일한 셀에 더 좋은 프로그램이 나타나면 기존 프로그램 교체
- 그리드 전체에서 **다양성과 성능의 균형** 유지

### 다양성 보존 메커니즘

1. **그리드 기반 선택**:
   ```python
   def sample_parent(self) -> Optional[Program]:
       # MAP-Elites 그리드에서 프로그램 샘플링
       if self.map_elites_grid and random.random() < 0.3:
           grid_program_id = random.choice(list(self.map_elites_grid.values()))
           return self.programs[grid_program_id]
       # 성능 기반 선택
       return self._select_by_performance()
   ```

2. **하이브리드 부모 선택**:
   ```python
   def sample_parent(self) -> Optional[Program]:
       if random.random() < 0.7:
           # 70% 확률로 엘리트 선택 (성능 중심)
           return elite_selection()
       else:
           # 30% 확률로 다양성 선택 (MAP-Elites 그리드)
           return diverse_selection()
   ```

3. **인구 정리 시 다양성 보존**:
   ```python
   def _cull_population(self) -> None:
       # 엘리트 프로그램 + MAP-Elites 그리드 프로그램 우선 보존
       elite_programs = sorted_programs[:self.elite_size]
       grid_programs = [p for p in programs if p.id in grid_program_ids]
   ```

## 🔄 진화 프로세스

### 1. 초기화 단계

```python
def _initialize_population(self, initial_code: str, problem_description: str):
```

- **단일 시드 프로그램**으로 시작 (전통적 GA와 차별점)
- 초기 코드를 정리하고 프로그래밍 언어 자동 감지
- 초기 프로그램을 평가하여 베이스라인 점수 설정
- `ProgramDatabase`에 초기 개체 저장

### 2. 세대별 진화 루프

```python
for iteration in range(max_iterations):
    # 1. 후보 생성 (변이 + 재작성)
    candidates = self._generate_candidates()
    
    # 2. 후보 평가 (성능 측정)
    self._evaluate_candidates(candidates)
    
    # 3. 개체군 업데이트 (선택 압력 적용)
    self._update_population(candidates)
    
    # 4. 진행 상황 로깅
    self._log_progress()
```

### 3. 종료 조건

- 최대 반복 횟수 도달
- 성능 향상 정체 (연속 N회 개선 없음)
- 사용자 중단

## 🧪 변이 연산자 (Mutation Operators)

### 1. 미세 변이 (Mutation) - 80% 확률

```python
def _mutate_program(self, parent: Program) -> Optional[Program]:
```

**특징:**
- **SEARCH/REPLACE** 형식으로 국소적 변화 적용
- **영감 시스템**: 상위 3개 프로그램을 참조 자료로 활용
- **구문 검증**: 변이된 코드의 문법 유효성 검사
- **유사도 검사**: 95% 이상 유사하면 거부하여 다양성 보장

**프롬프트 구조:**
```
System: "SEARCH/REPLACE 형식으로 성능 개선을 제안하세요"
User: "현재 코드: [code] + 고성능 프로그램들의 영감"
```

### 2. 전체 재작성 (Rewrite) - 20% 확률

```python
def _rewrite_program(self, parent: Program) -> Optional[Program]:
```

**특징:**
- 완전히 새로운 구현 생성
- 동일한 목표를 다른 알고리즘/접근법으로 달성
- 더 급진적인 변화를 통한 탐색 공간 확장
- 지역 최적해 탈출 메커니즘

## 🗄️ 개체군 관리 (Population Management)

### MAP-Elites 기반 데이터베이스

```python
class ProgramDatabase:
    def __init__(self, population_size: int = 50, elite_ratio: float = 0.2,
                 complexity_bins: int = 5, diversity_bins: int = 5):
        self.programs: Dict[str, Program] = {}
        self.elite_size = max(1, int(population_size * elite_ratio))
        
        # MAP-Elites 그리드 설정
        self.complexity_bins = complexity_bins
        self.diversity_bins = diversity_bins
        self.map_elites_grid: Dict[Tuple[int, int], str] = {}  # (복잡도, 다양성) -> 프로그램 ID
        
        # 그리드 통계
        self.grid_stats = {
            'filled_cells': 0,
            'total_cells': complexity_bins * diversity_bins,
            'coverage': 0.0
        }
```

**핵심 전략:**

#### 1. MAP-Elites 그리드 관리
- **2차원 그리드**: 복잡도(0-4) × 다양성(0-4) = 25개 셀
- **각 셀당 최고 성능 프로그램 하나만 보존**
- **행동 기술자 기반 자동 배치**

```python
def _get_behavior_descriptor(self, program: Program) -> Tuple[int, int]:
    """프로그램의 행동 기술자 (복잡도, 다양성) 계산"""
    complexity_bin = min(int(program.complexity * self.complexity_bins), 
                        self.complexity_bins - 1)
    diversity_bin = min(int(program.diversity * self.diversity_bins), 
                       self.diversity_bins - 1)
    return (complexity_bin, diversity_bin)

def _update_map_elites_grid(self, program: Program) -> bool:
    """MAP-Elites 그리드 업데이트"""
    behavior_descriptor = self._get_behavior_descriptor(program)
    
    if behavior_descriptor not in self.map_elites_grid:
        # 빈 셀에 새 프로그램 배치
        self.map_elites_grid[behavior_descriptor] = program.id
        self.grid_stats['filled_cells'] += 1
        return True
    else:
        # 기존 프로그램과 성능 비교
        existing_id = self.map_elites_grid[behavior_descriptor]
        if program.score > self.programs[existing_id].score:
            self.map_elites_grid[behavior_descriptor] = program.id
            return True
    return False
```

#### 2. 하이브리드 엘리트 보존
- **MAP-Elites 그리드**: 다양성 기반 보존
- **전통적 엘리트**: 상위 성능 프로그램 추가 보존
- **이중 보존 전략**으로 성능과 다양성 모두 확보

```python
def _cull_population(self) -> None:
    """개체군 크기 관리 - MAP-Elites + 엘리트 보존"""
    # 1. MAP-Elites 그리드의 모든 프로그램 보존
    protected_ids = set(self.map_elites_grid.values())
    
    # 2. 상위 성능 엘리트 프로그램 추가 보존
    sorted_programs = sorted(self.programs.values(), 
                           key=lambda p: p.score, reverse=True)
    for program in sorted_programs[:self.elite_size]:
        protected_ids.add(program.id)
    
    # 3. 보호되지 않은 프로그램 중 무작위 제거
    if len(self.programs) > self.population_size:
        # ... 초과 개체 제거 로직
```

#### 3. 다양성 기반 부모 선택

```python
def sample_parent(self, method: str = "weighted") -> Optional[Program]:
    """다양한 선택 전략"""
    if method == "grid_based":
        # MAP-Elites 그리드에서 균등 선택
        if self.map_elites_grid:
            grid_program_id = random.choice(list(self.map_elites_grid.values()))
            return self.programs[grid_program_id]
    
    elif method == "diverse":
        # 다양성 높은 프로그램 우선 선택
        programs = list(self.programs.values())
        weights = [p.diversity + 0.1 for p in programs]
        return random.choices(programs, weights=weights, k=1)[0]
    
    else:  # "weighted" - 성능 기반 가중 선택
        programs = list(self.programs.values())
        weights = [max(0.1, p.score + 1.0) for p in programs]
        return random.choices(programs, weights=weights, k=1)[0]
```

#### 4. 그리드 통계 및 모니터링

```python
def stats(self) -> Dict[str, Any]:
    """MAP-Elites 확장 통계"""
    programs = list(self.programs.values())
    scores = [p.score for p in programs]
    
    # 그리드 커버리지 계산
    coverage = len(self.map_elites_grid) / (self.complexity_bins * self.diversity_bins)
    
    return {
        # 기본 통계
        "size": len(self.programs),
        "best_score": max(scores) if scores else 0.0,
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        
        # MAP-Elites 그리드 통계
        "filled_cells": len(self.map_elites_grid),
        "total_cells": self.complexity_bins * self.diversity_bins,
        "coverage": coverage,
        "grid_diversity": len(set(self.map_elites_grid.values())),
        
        # 행동 기술자 분포
        "avg_complexity": sum(p.complexity for p in programs) / len(programs) if programs else 0.0,
        "avg_diversity": sum(p.diversity for p in programs) / len(programs) if programs else 0.0,
        
        # 그리드 셀별 성능 분포
        "grid_performance": {
            f"cell_{coord[0]}_{coord[1]}": self.programs[prog_id].score
            for coord, prog_id in self.map_elites_grid.items()
        }
    }
```

### 영감 시스템 제거

기존의 복잡한 영감 시스템은 제거되었으며, 대신 더 직접적인 부모 선택 메커니즘을 사용합니다:

```python
def sample_parent(self) -> Optional[Program]:
    """단순화된 부모 선택 전략"""
    programs = list(self.programs.values())
    if not programs:
        return None
    
    # 성능 기반 가중 선택
    weights = [max(0.1, p.score + 1.0) for p in programs]
    return random.choices(programs, weights=weights, k=1)[0]
```

## 🆚 OpenEvolve와의 차이점

### HypoEvolve (현재 시스템)
- ✅ **MAP-Elites 그리드 구현** - 복잡도 × 다양성 2D 그리드 (5×5 = 25셀)
- ✅ **행동 기술자** - 복잡도, 다양성 기반 프로그램 분류
- ✅ **하이브리드 보존** - MAP-Elites + 전통적 엘리트 보존
- ✅ **다양성 기반 선택** - 그리드, 다양성, 성능 기반 선택 전략
- ❌ **섬 모델 없음** - 단일 개체군 (병렬 처리 없음)
- ✅ **영감 시스템** - 고성능 프로그램 참조

### OpenEvolve (고급 시스템)
- ✅ **진짜 MAP-Elites** - 2D 그리드 (점수 × 복잡도)
- ✅ **섬 기반 진화** - 5개 섬 + 이주 메커니즘
- ✅ **다층적 선택** - 탐험/활용/무작위 전략
- ✅ **특성 공간** - 복잡도, 다양성, 점수 기반 구분
- ✅ **병렬 처리** - 멀티 섬 동시 진화

```python
# HypoEvolve의 MAP-Elites 그리드
map_elites_grid: Dict[Tuple[int, int], str] = {}  # (복잡도, 다양성) -> program_id
behavior_descriptor = (complexity_bin, diversity_bin)  # (3, 2)

# OpenEvolve의 MAP-Elites 그리드  
feature_map: Dict[str, str] = {}  # "3-7" -> program_id
feature_coords = [complexity_bin, score_bin]  # [3, 7]
```

### 주요 차이점

#### 1. 그리드 차원
- **HypoEvolve**: 복잡도 × 다양성 (품질 무관 다양성 보존)
- **OpenEvolve**: 복잡도 × 점수 (성능 기반 다양성)

#### 2. 병렬 처리
- **HypoEvolve**: 단일 개체군, 순차적 진화
- **OpenEvolve**: 다중 섬, 병렬 진화 + 이주

#### 3. 선택 전략
- **HypoEvolve**: 그리드/다양성/성능 기반 선택
- **OpenEvolve**: 탐험/활용/무작위 3단계 선택

#### 4. 보존 전략
- **HypoEvolve**: MAP-Elites + 엘리트 이중 보존
- **OpenEvolve**: 순수 MAP-Elites 기반 보존

## 🎯 평가 시스템

### 평가자 종류

1. **FunctionEvaluator**: 파일 기반 평가 함수
2. **SimpleEvaluator**: 커스텀 평가 함수
3. **CascadeEvaluator**: 다단계 평가

### 평가 메트릭

```python
@dataclass
class Program:
    metrics: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0
```

- **주 점수 (score)**: 전체 성능 지표
- **실행 시간**: 알고리즘 효율성
- **정확도**: 문제 해결 정확성
- **메모리 사용량**: 리소스 효율성
- **사용자 정의 메트릭**: 도메인별 특화 지표

## ⚙️ 하이퍼파라미터

```python
@dataclass
class Config:
    # 진화 파라미터
    max_iterations: int = 100      # 최대 세대 수
    population_size: int = 50      # 개체군 크기
    elite_ratio: float = 0.2       # 엘리트 보존 비율
    mutation_rate: float = 0.8     # 변이 vs 재작성 비율
    
    # MAP-Elites 파라미터
    complexity_bins: int = 5       # 복잡도 차원 빈 개수
    diversity_bins: int = 5        # 다양성 차원 빈 개수
    grid_selection_ratio: float = 0.3  # 그리드 기반 선택 비율
    
    # LLM 파라미터
    model: str = "gpt-4"          # 사용할 언어 모델
    temperature: float = 0.7       # 창의성 수준
    max_tokens: int = 4096         # 최대 응답 길이
    
    # 평가 파라미터
    timeout: int = 30             # 평가 제한 시간
    max_retries: int = 3          # 최대 재시도 횟수
```

### 파라미터 설명

#### MAP-Elites 관련 파라미터
- **complexity_bins**: 복잡도 차원을 나누는 구간 수 (기본값: 5)
- **diversity_bins**: 다양성 차원을 나누는 구간 수 (기본값: 5)
- **grid_selection_ratio**: 부모 선택 시 그리드 기반 선택 비율 (기본값: 0.3)

#### 권장 설정값
```python
# 작은 문제 (빠른 실험)
small_config = Config(
    complexity_bins=3,
    diversity_bins=3,
    population_size=20,
    max_iterations=25
)

# 중간 문제 (균형잡힌 탐색)
medium_config = Config(
    complexity_bins=5,
    diversity_bins=5,
    population_size=50,
    max_iterations=100
)

# 큰 문제 (철저한 탐색)
large_config = Config(
    complexity_bins=7,
    diversity_bins=7,
    population_size=100,
    max_iterations=200
)
```

## 🔍 알고리즘의 독특한 특징

### 1. LLM 기반 의미론적 변이
- 전통적 비트 플립이 아닌 **의미 이해 기반 변화**
- 프로그래밍 지식과 베스트 프랙티스 자동 적용
- 문맥을 고려한 지능적 코드 개선

### 2. 영감 기반 학습
- 상위 성능 프로그램들을 **학습 자료**로 활용
- 성공적인 패턴과 기법의 자연스러운 전파
- 교차 없이도 효과적인 지식 공유

### 3. 적응적 품질 관리
- **구문 검증**으로 무효한 개체 자동 제거
- **유사도 검사**로 정체 상황 방지
- **타임아웃 처리**로 무한 루프 방지

### 4. 계층적 변이 전략
- **미세 조정 (80%)**: 점진적이고 안전한 개선
- **급진적 재설계 (20%)**: 혁신적 솔루션 탐색

## 📊 진화 통계 및 모니터링

```python
def stats(self) -> Dict[str, Any]:
    """MAP-Elites 확장 통계"""
    programs = list(self.programs.values())
    scores = [p.score for p in programs]
    
    # 그리드 커버리지 계산
    coverage = len(self.map_elites_grid) / (self.complexity_bins * self.diversity_bins)
    
    return {
        # 기본 통계
        "size": len(self.programs),
        "best_score": max(scores) if scores else 0.0,
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        
        # MAP-Elites 그리드 통계
        "filled_cells": len(self.map_elites_grid),
        "total_cells": self.complexity_bins * self.diversity_bins,
        "coverage": coverage,
        "grid_diversity": len(set(self.map_elites_grid.values())),
        
        # 행동 기술자 분포
        "avg_complexity": sum(p.complexity for p in programs) / len(programs) if programs else 0.0,
        "avg_diversity": sum(p.diversity for p in programs) / len(programs) if programs else 0.0,
        
        # 그리드 셀별 성능 분포
        "grid_performance": {
            f"cell_{coord[0]}_{coord[1]}": self.programs[prog_id].score
            for coord, prog_id in self.map_elites_grid.items()
        }
    }
```

### 진행 상황 추적

#### 1. 기본 진화 메트릭
- **세대별 성능 변화** 모니터링
- **개체군 크기** 및 엘리트 비율 추적
- **변이 성공률** 모니터링

#### 2. MAP-Elites 그리드 메트릭
- **그리드 커버리지**: 채워진 셀 비율 (0.0 ~ 1.0)
- **셀별 성능 분포**: 각 (복잡도, 다양성) 조합의 최고 점수
- **다양성 지수**: 서로 다른 프로그램 수

#### 3. 행동 기술자 분석
- **복잡도 분포**: 개체군의 평균 복잡도 변화
- **다양성 분포**: 개체군의 평균 다양성 변화
- **특성 공간 밀도**: 각 차원별 분포 균형

#### 4. 시각화 예시
```python
# 그리드 히트맵 생성
import matplotlib.pyplot as plt
import numpy as np

def visualize_grid(database: ProgramDatabase):
    grid = np.zeros((database.complexity_bins, database.diversity_bins))
    
    for (complexity, diversity), prog_id in database.map_elites_grid.items():
        grid[complexity, diversity] = database.programs[prog_id].score
    
    plt.imshow(grid, cmap='viridis', aspect='auto')
    plt.xlabel('다양성 빈')
    plt.ylabel('복잡도 빈')
    plt.title('MAP-Elites 그리드 성능 분포')
    plt.colorbar(label='성능 점수')
    plt.show()
```

## 💡 알고리즘의 강점

### 1. 의미적 이해
- LLM이 코드의 **의도와 목적**을 파악
- 단순한 구문 변경을 넘어선 **논리적 개선**

### 2. 도메인 지식 활용
- 프로그래밍 **베스트 프랙티스** 자동 적용
- **알고리즘 최적화** 패턴 인식 및 적용

### 3. MAP-Elites 기반 다양성 보존
- **복잡도 × 다양성 그리드**로 체계적인 다양성 관리
- **각 특성 조합에서 최고 성능 프로그램** 자동 보존
- **조기 수렴 방지** 및 탐색 공간 확장

### 4. 하이브리드 보존 전략
- **MAP-Elites 그리드**: 다양성 기반 보존
- **엘리트 보존**: 성능 기반 추가 보존
- **이중 안전망**으로 성능과 다양성 모두 확보

### 5. 영감 기반 학습
- **고성능 프로그램들로부터 학습**
- **그리드 기반 다양한 영감** 제공
- 교차 없이도 효과적인 지식 전파

### 6. 적응적 선택 전략
- **그리드 기반**: 다양성 우선 선택
- **성능 기반**: 고성능 프로그램 우선
- **다양성 기반**: 새로운 특성 공간 탐색

### 7. 확장 가능성
- 새로운 평가자 쉽게 추가
- 다양한 프로그래밍 언어 지원
- 행동 기술자 커스터마이징 가능

## ⚠️ 현재 시스템의 한계점

### 1. 단일 개체군 한계
- **병렬 처리 없음** - 섬 모델 미구현으로 탐색 효율성 제한
- **단일 진화 경로** - 다양한 진화 전략 동시 실행 불가

### 2. 그리드 차원 제한
- **2차원 그리드만 지원** - 복잡도 × 다양성만 고려
- **점수 차원 미포함** - OpenEvolve 대비 성능 기반 분류 부족

### 3. 행동 기술자 단순함
- **복잡도**: 단순 코드 길이 기반 계산
- **다양성**: 기본 해시 기반 유사도만 고려
- **더 정교한 메트릭 필요** - AST 구조, 알고리즘 패턴 등

### 4. 선택 압력 조절 어려움
- **고정된 선택 전략** - 동적 탐험/활용 균형 조절 부족
- **그리드 셀 간 균형** - 일부 셀에 집중될 가능성

### 5. 확장성 제약
- **그리드 크기 고정** - 5×5 그리드로 제한
- **메모리 사용량** - 대규모 개체군에서 그리드 관리 오버헤드

## 🎯 적용 분야

### 1. 알고리즘 최적화
- 정렬, 검색 알고리즘 성능 개선
- 수치 계산 최적화
- 데이터 구조 효율성 향상

### 2. 코드 리팩토링
- 가독성 향상
- 성능 최적화
- 코드 간소화

### 3. 자동 버그 수정
- 논리 오류 탐지 및 수정
- 성능 병목 해결
- 예외 처리 개선

### 4. 코드 변환 및 포팅
- 언어 간 자동 변환
- 라이브러리 마이그레이션
- 플랫폼 최적화

## 🚀 사용 예시

```python
from hypoevolve import HypoEvolve

# 설정 초기화
config = Config(
    max_iterations=50,
    population_size=30,
    model="gpt-4"
)

# 진화 컨트롤러 생성
controller = HypoEvolve(config)

# 코드 진화 실행
result = controller.evolve(
    initial_code="""
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    """,
    problem_description="Optimize the Fibonacci function for better performance",
    max_iterations=50
)

print(f"Best score: {result.score}")
print(f"Optimized code:\n{result.code}")
```

## 🔬 연구 및 개발 방향

### 현재 한계점
1. LLM API 비용 및 속도
2. 단순한 개체군 관리 (진짜 MAP-Elites 없음)
3. 다양성 관리 부족

### 향후 개선 방향
1. **진짜 MAP-Elites 구현** - OpenEvolve 방식 도입
2. **섬 기반 진화** - 병렬 탐색 전략
3. **특성 공간 확장** - 복잡도, 다양성, 가독성 등 다차원 평가
4. **로컬 LLM** 통합으로 비용 절감
5. **멀티모달** 코드 이해 (주석, 문서 포함)

## 📚 참고 문헌

- Genetic Programming 기본 이론
- MAP-Elites 알고리즘 (실제 구현은 OpenEvolve 참조)
- Large Language Models for Code
- Evolutionary Computation 최신 연구

---

*이 문서는 HypoEvolve 프로젝트의 진화 알고리즘을 상세히 분석한 기술 문서입니다. HypoEvolve는 MAP-Elites 영감을 받았지만 실제로는 전통적인 엘리트 보존 유전 알고리즘을 구현하고 있습니다.* 