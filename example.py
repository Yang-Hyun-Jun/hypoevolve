"""
HypoEvolve 테스트 실행 스크립트

이 스크립트는 hypoevolve_default.yaml 설정을 사용하여
피보나치 함수 최적화 테스트를 실행합니다.
"""

import os
import tempfile

from hypoevolve import HypoEvolve
from hypoevolve.core.config import Config


def main():
    """
    HypoEvolve 테스트 실행
    """
    print("🚀 HypoEvolve 실행 테스트 시작!")
    print("=" * 50)

    # API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("   다음 명령어로 API 키를 설정하세요:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return

    # 설정 파일 로드
    config_path = "configs/hypoevolve_default.yaml"
    print(f"📋 설정 파일 로드: {config_path}")

    try:
        config = Config.from_yaml(config_path)
        # 환경 변수에서 API 키 사용
        config.api_key = api_key

        print("✅ 설정 로드 성공!")
        print(f"   - 모델: {config.model}")
        print(f"   - 최대 반복: {config.max_iterations}")
        print(f"   - 인구 크기: {config.population_size}")
        print(f"   - 출력 디렉토리: {config.output_dir}")
        print(f"   - 온도: {config.temperature}")
        print(f"   - 병렬 평가: {config.parallel_evaluations}")
    except Exception as e:
        print(f"❌ 설정 로드 실패: {e}")
        return

    # HypoEvolve 인스턴스 생성
    hypoevolve = HypoEvolve(config)

    # 초기 코드 (비효율적인 피보나치 수열)
    initial_code = """
def fibonacci(n):
    \"\"\"비효율적인 재귀 피보나치 구현\"\"\"
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 테스트 실행
if __name__ == "__main__":
    import sys
    n = int(input())
    result = fibonacci(n)
    print(result)
"""

    # 문제 설명
    problem_description = """
피보나치 수열을 계산하는 함수를 최적화하세요.

요구사항:
1. 입력: 정수 n (0 <= n <= 20)
2. 출력: n번째 피보나치 수
3. 성능: 빠른 실행 시간 (특히 큰 n에 대해)
4. 정확성: 올바른 결과 출력

피보나치 수열: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, ...

현재 코드는 비효율적인 재귀 방식으로 구현되어 있습니다.
동적 프로그래밍, 메모이제이션, 또는 반복문을 사용해서 최적화해보세요.

성능 목표:
- n=10일 때 0.01초 이내
- n=15일 때 0.1초 이내  
- n=20일 때 0.5초 이내
"""

    # 평가 함수 생성
    evaluation_code = """
import subprocess
import sys
import time
from typing import Dict, Any


def evaluate(program_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"피보나치 함수 평가\"\"\"
    
    # 테스트 케이스 (작은 값부터 큰 값까지)
    test_cases = [
        {"input": 0, "expected": 0, "weight": 1.0},
        {"input": 1, "expected": 1, "weight": 1.0},
        {"input": 2, "expected": 1, "weight": 1.0},
        {"input": 3, "expected": 2, "weight": 1.0},
        {"input": 5, "expected": 5, "weight": 1.2},
        {"input": 8, "expected": 21, "weight": 1.5},
        {"input": 10, "expected": 55, "weight": 2.0},
        {"input": 15, "expected": 610, "weight": 3.0},
        {"input": 20, "expected": 6765, "weight": 5.0},
    ]
    
    total_score = 0.0
    total_weight = sum(tc["weight"] for tc in test_cases)
    passed_tests = 0
    total_tests = len(test_cases)
    execution_times = []
    results = []
    
    for test_case in test_cases:
        try:
            start_time = time.time()
            
            # 프로그램 실행
            process = subprocess.run(
                [sys.executable, program_path],
                input=str(test_case["input"]),
                capture_output=True,
                text=True,
                timeout=5  # 5초 타임아웃
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            
            if process.returncode == 0:
                output = process.stdout.strip()
                expected = str(test_case["expected"])
                
                if output == expected:
                    passed_tests += 1
                    
                    # 기본 점수
                    base_score = test_case["weight"]
                    
                    # 성능 보너스 (빠른 실행에 보너스)
                    if execution_time < 0.1:
                        time_bonus = base_score * 0.2
                    elif execution_time < 0.5:
                        time_bonus = base_score * 0.1
                    else:
                        time_bonus = 0
                    
                    total_score += base_score + time_bonus
                    results.append({
                        "input": test_case["input"],
                        "passed": True,
                        "time": execution_time,
                        "score": base_score + time_bonus
                    })
                else:
                    results.append({
                        "input": test_case["input"],
                        "passed": False,
                        "time": execution_time,
                        "score": 0,
                        "expected": expected,
                        "actual": output
                    })
            else:
                results.append({
                    "input": test_case["input"],
                    "passed": False,
                    "time": execution_time,
                    "score": 0,
                    "error": process.stderr.strip()
                })
                    
        except subprocess.TimeoutExpired:
            execution_times.append(5.0)
            results.append({
                "input": test_case["input"],
                "passed": False,
                "time": 5.0,
                "score": 0,
                "error": "Timeout"
            })
            continue
        except Exception as e:
            execution_times.append(5.0)
            results.append({
                "input": test_case["input"],
                "passed": False,
                "time": 5.0,
                "score": 0,
                "error": str(e)
            })
            continue
    
    # 최종 점수 계산
    normalized_score = total_score / total_weight if total_weight > 0 else 0.0
    accuracy = passed_tests / total_tests if total_tests > 0 else 0.0
    avg_time = sum(execution_times) / len(execution_times) if execution_times else 5.0
    
    # 성능 점수 (빠를수록 높은 점수)
    if avg_time < 0.1:
        performance_score = 1.0
    elif avg_time < 0.5:
        performance_score = 0.8
    elif avg_time < 1.0:
        performance_score = 0.6
    elif avg_time < 2.0:
        performance_score = 0.4
    else:
        performance_score = 0.2
    
    return {
        "score": min(normalized_score, 1.0),  # 최대 1.0으로 제한
        "accuracy": accuracy,
        "performance": performance_score,
        "passed_tests": passed_tests,
        "total_tests": total_tests,
        "avg_execution_time": avg_time,
        "test_results": results,
        "metrics": {
            "correctness": accuracy,
            "speed": performance_score,
            "efficiency": performance_score,
            "weighted_score": normalized_score
        }
    }
"""

    # 임시 평가 함수 파일 생성
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(evaluation_code)
        eval_file = f.name

    try:
        print("\n📊 평가 함수 설정...")
        hypoevolve.set_evaluation_function(eval_file, {"test_mode": True})
        print("✅ 평가 함수 설정 완료!")

        print("\n🧬 진화 시작...")
        print(f"초기 코드 길이: {len(initial_code)} 문자")
        print(f"최대 반복 횟수: {config.max_iterations}")
        print(f"인구 크기: {config.population_size}")
        print(f"엘리트 비율: {config.elite_ratio}")
        print(f"변이 확률: {config.mutation_rate}")
        print()

        # 🧬 진화 실행
        best_program = hypoevolve.evolve(
            initial_code=initial_code,
            problem_description=problem_description,
            max_iterations=config.max_iterations,
        )

        print("\n🎉 진화 완료!")
        print("=" * 50)

        if best_program:
            print("🏆 최고 성능 프로그램:")
            print(f"   - 점수: {best_program.score:.4f}")
            print(f"   - 세대: {best_program.generation}")
            print(f"   - 프로그램 ID: {best_program.id}")

            # 메트릭 정보 출력
            if hasattr(best_program, "metrics") and best_program.metrics:
                print(
                    f"   - 정확도: {best_program.metrics.get('correctness', 'N/A'):.4f}"
                )
                print(f"   - 성능: {best_program.metrics.get('speed', 'N/A'):.4f}")
                print(
                    f"   - 가중 점수: {best_program.metrics.get('weighted_score', 'N/A'):.4f}"
                )

            print("\n📈 최적화된 코드:")
            print("-" * 50)
            print(best_program.code)
            print("-" * 50)

            # 데이터베이스 통계
            stats = hypoevolve.database.stats()
            print("\n📊 진화 통계:")
            print(f"   - 총 프로그램 수: {stats['size']}")
            print(f"   - 현재 세대: {stats['generation']}")
            print(f"   - 최고 점수: {stats['best_score']:.4f}")
            print(f"   - 평균 점수: {stats['avg_score']:.4f}")
            print(f"   - 최저 점수: {stats['min_score']:.4f}")

            # MAP-Elites 정보 (있는 경우)
            if "coverage" in stats:
                print(f"   - 그리드 커버리지: {stats['coverage']:.2%}")
                print(
                    f"   - 점유된 셀: {stats['occupied_cells']}/{stats['total_cells']}"
                )

            # 진화 히스토리 정보
            try:
                history = hypoevolve.database.get_evolution_history()
                if history:
                    print("\n🔍 진화 히스토리:")
                    print(f"   - 추적된 세대: {len(history.get('generations', []))}")
                    print(f"   - 개선 추세: {history.get('trend', 'N/A')}")

                    # 최근 개선 사항 출력
                    if "generations" in history and len(history["generations"]) > 0:
                        recent_gens = history["generations"][-5:]  # 최근 5세대
                        print("   - 최근 세대별 최고 점수:")
                        for gen_data in recent_gens:
                            gen_num = gen_data.get("generation", "N/A")
                            best_score = gen_data.get("best_score", "N/A")
                            print(f"     * 세대 {gen_num}: {best_score:.4f}")

            except Exception as e:
                print(f"   - 히스토리 로드 실패: {e}")

        else:
            print("❌ 진화 실패 - 최적 프로그램을 찾지 못했습니다.")

    except Exception as e:
        print(f"❌ 진화 실행 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # 임시 파일 정리
        if os.path.exists(eval_file):
            os.unlink(eval_file)
            print("\n🧹 임시 파일 정리 완료")

    print("\n✅ 테스트 완료!")


if __name__ == "__main__":
    main()
