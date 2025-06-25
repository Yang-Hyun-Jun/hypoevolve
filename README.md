# HypoEvolve 예제

이 디렉토리에는 HypoEvolve를 사용하는 다양한 예제들이 포함되어 있습니다.

## 평가 함수 (Evaluation Functions)

HypoEvolve는 이제 OpenEvolve 스타일의 평가 함수를 지원합니다! 테스트 케이스 대신 사용자 정의 평가 함수를 사용하여 더 복잡하고 유연한 평가가 가능합니다.

### 평가 함수 사용법

```python
from hypoevolve.core.config import Config
from hypoevolve.core.controller import HypoEvolve

# HypoEvolve 인스턴스 생성
config = Config.from_yaml("configs/hypoevolve_quick.yaml")
hypoevolve = HypoEvolve(config)

# 평가 함수 설정
hypoevolve.set_evaluation_function("examples/evaluations/factorial_evaluation.py")

# 진화 실행
best_program = await hypoevolve.evolve(
    initial_code=initial_code,
    problem_description=problem_description
)
```

### 평가 함수 작성법

평가 함수는 다음과 같은 형식으로 작성해야 합니다:

```python
def evaluate(program_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    프로그램을 평가하는 함수
    
    Args:
        program_path: 평가할 프로그램 파일 경로
        context: 평가 컨텍스트 (추가 정보)
    
    Returns:
        평가 결과 딕셔너리
        - score: 0.0~1.0 사이의 점수 (필수)
        - 기타 메트릭들 (선택사항)
    """
    
    # 프로그램 실행 및 평가 로직
    # ...
    
    return {
        "score": final_score,  # 필수: 0.0~1.0 사이의 값
        "passed_tests": passed_count,
        "total_tests": total_count,
        "metrics": {
            "accuracy": accuracy_score,
            "performance": performance_score,
        }
    }
```

## 제공되는 평가 함수 예제

### 1. 팩토리얼 평가 (`evaluations/factorial_evaluation.py`)
- 팩토리얼 함수의 정확성을 평가
- 다양한 입력값에 대한 테스트
- 간단한 정확성 기반 평가

### 2. 정렬 알고리즘 평가 (`evaluations/sorting_evaluation.py`)
- 정렬 알고리즘의 정확성과 성능을 평가
- 다양한 크기의 배열 테스트
- 정확성 (70%) + 성능 (30%) 복합 평가

## 실행 예제

### 평가 함수 예제 실행
```bash
cd /path/to/openevolve
python examples/function_evaluation_example.py
```

## 기존 테스트 케이스 방식과의 차이점

| 특징 | 테스트 케이스 방식 | 평가 함수 방식 |
|------|------------------|---------------|
| **유연성** | 제한적 (입력/출력 매칭) | 높음 (사용자 정의 로직) |
| **평가 복잡도** | 단순 (정확성만) | 복합 (성능, 메모리, 스타일 등) |
| **실행 환경** | 제한적 | 실제 환경에서 실행 가능 |
| **도메인 특화** | 어려움 | 쉬움 (도메인별 평가 로직) |
| **설정 복잡도** | 간단 | 중간 (평가 함수 작성 필요) |

## 평가 함수의 장점

1. **복합 평가**: 정확성, 성능, 메모리 사용량 등을 종합적으로 평가
2. **실제 환경**: 프로그램을 실제 환경에서 실행하여 평가
3. **도메인 특화**: 특정 도메인에 맞는 평가 로직 구현 가능
4. **유연성**: 복잡한 평가 기준과 다단계 평가 지원
5. **확장성**: 새로운 평가 기준을 쉽게 추가 가능

## 평가 함수 개발 팁

1. **타임아웃 설정**: 무한 루프나 긴 실행 시간을 방지
2. **예외 처리**: 프로그램 실행 실패에 대한 적절한 처리
3. **점수 정규화**: 최종 점수를 0.0~1.0 사이로 정규화
4. **상세한 결과**: 디버깅을 위한 상세한 평가 결과 제공
5. **성능 고려**: 평가 함수 자체의 실행 시간도 고려

평가 함수를 사용하면 더욱 강력하고 유연한 코드 진화가 가능합니다! 🚀 