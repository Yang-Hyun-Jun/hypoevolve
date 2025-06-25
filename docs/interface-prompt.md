# HypoEvolve 프롬프트 인터페이스 문서

## 개요

HypoEvolve는 LLM을 활용한 코드 진화 시스템에서 두 가지 주요 프롬프트 전략을 사용합니다:
1. **코드 변이 (Mutation)**: SEARCH/REPLACE 형식으로 기존 코드의 일부를 개선
2. **완전 재작성 (Full Rewrite)**: 전체 코드를 새롭게 구현

모든 프롬프트는 `hypoevolve/llm/prompts.py` 파일에서 관리되며, `PromptManager` 클래스를 통해 동적으로 생성됩니다.

---

## 1. 코드 변이 프롬프트 (Mutation Prompts)

### 1.1 시스템 프롬프트 (MUTATION_SYSTEM_PROMPT)

**원문:**
```
You are an expert programmer tasked with evolving code to improve its performance.

Your task is to analyze the current code and suggest improvements using the SEARCH/REPLACE format.

Format your response as:
<<<<<<< SEARCH
[exact code to be replaced]
=======
[improved replacement code]
>>>>>>> REPLACE

Focus on:
1. Algorithmic improvements
2. Performance optimizations
3. Code simplification
4. Bug fixes

Make meaningful changes that could improve the program's performance.
```

**한국어 번역:**
```
당신은 코드 성능 향상을 위해 코드를 진화시키는 임무를 맡은 전문 프로그래머입니다.

현재 코드를 분석하고 SEARCH/REPLACE 형식을 사용하여 개선 사항을 제안하는 것이 당신의 임무입니다.

다음 형식으로 응답하세요:
<<<<<<< SEARCH
[교체할 정확한 코드]
=======
[개선된 대체 코드]
>>>>>>> REPLACE

다음에 집중하세요:
1. 알고리즘 개선
2. 성능 최적화
3. 코드 단순화
4. 버그 수정

프로그램의 성능을 향상시킬 수 있는 의미 있는 변경을 수행하세요.
```

### 1.2 사용자 프롬프트 템플릿 (MUTATION_USER_TEMPLATE)

**원문:**
```
Current code:
{current_code}

{context}

Please suggest an improvement using the SEARCH/REPLACE format.
```

**한국어 번역:**
```
현재 코드:
{current_code}

{context}

SEARCH/REPLACE 형식을 사용하여 개선 사항을 제안해주세요.
```

---

## 2. 완전 재작성 프롬프트 (Full Rewrite Prompts)

### 2.1 시스템 프롬프트 (REWRITE_SYSTEM_PROMPT)

**원문:**
```
You are an expert programmer tasked with rewriting code to improve its performance.

Analyze the current code and create a completely new implementation that achieves the same goal but with better performance.

Return only the new code without any explanations or markdown formatting.
```

**한국어 번역:**
```
당신은 성능 향상을 위해 코드를 재작성하는 임무를 맡은 전문 프로그래머입니다.

현재 코드를 분석하고 동일한 목표를 달성하지만 더 나은 성능을 가진 완전히 새로운 구현을 만드세요.

설명이나 마크다운 형식 없이 새로운 코드만 반환하세요.
```

### 2.2 사용자 프롬프트 템플릿 (REWRITE_USER_TEMPLATE)

**원문:**
```
Current code:
{current_code}

{context}

Please provide a complete rewrite of this code with improved performance.
```

**한국어 번역:**
```
현재 코드:
{current_code}

{context}

성능이 개선된 이 코드의 완전한 재작성을 제공해주세요.
```

---

## 3. 영감 템플릿 (Inspiration Template)

### 3.1 영감 예시 템플릿 (INSPIRATION_TEMPLATE)

**원문:**
```

Here are some high-performing code examples for inspiration:
{inspiration_text}
```

**한국어 번역:**
```

영감을 위한 고성능 코드 예시들입니다:
{inspiration_text}
```

---

## 4. PromptManager 클래스

### 4.1 클래스 개요

`PromptManager` 클래스는 HypoEvolve의 모든 프롬프트 생성을 담당하는 정적 클래스입니다.

**주요 기능:**
- 코드 변이용 프롬프트 생성
- 완전 재작성용 프롬프트 생성
- 영감 코드 예시 추가
- 동적 컨텍스트 삽입

### 4.2 메서드 분석

#### 4.2.1 get_mutation_prompts()

**메서드 시그니처:**
```python
@staticmethod
def get_mutation_prompts(
    current_code: str, 
    context: str = "", 
    inspirations: list = None
) -> tuple[str, str]:
```

**기능:**
- 코드 변이를 위한 시스템 및 사용자 프롬프트 생성
- 현재 코드와 컨텍스트를 템플릿에 삽입
- 영감 코드가 제공되면 추가로 포함

**반환값:**
- `tuple[str, str]`: (시스템 프롬프트, 사용자 프롬프트)

#### 4.2.2 get_rewrite_prompts()

**메서드 시그니처:**
```python
@staticmethod
def get_rewrite_prompts(
    current_code: str, 
    context: str = "", 
    inspirations: list = None
) -> tuple[str, str]:
```

**기능:**
- 완전 재작성을 위한 시스템 및 사용자 프롬프트 생성
- 현재 코드와 컨텍스트를 템플릿에 삽입
- 영감 코드가 제공되면 추가로 포함

**반환값:**
- `tuple[str, str]`: (시스템 프롬프트, 사용자 프롬프트)

---

## 5. 프롬프트 사용 흐름

### 5.1 코드 변이 시나리오

```python
# 1. PromptManager를 통해 프롬프트 생성
system_prompt, user_prompt = PromptManager.get_mutation_prompts(
    current_code="def fibonacci(n): return 1 if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    context="피보나치 함수 최적화",
    inspirations=["def fibonacci_optimized(n): # 메모이제이션 버전"]
)

# 2. LLM에게 전송
# 3. SEARCH/REPLACE 형식의 응답 수신
# 4. 코드 패치 적용
```

### 5.2 완전 재작성 시나리오

```python
# 1. PromptManager를 통해 프롬프트 생성
system_prompt, user_prompt = PromptManager.get_rewrite_prompts(
    current_code="def sort_list(lst): return sorted(lst)",
    context="정렬 알고리즘 성능 최적화"
)

# 2. LLM에게 전송
# 3. 완전히 새로운 코드 수신
# 4. 기존 코드 완전 교체
```

---

## 6. 프롬프트 예시

### 6.1 변이 프롬프트 완전한 예시

**시스템 메시지:**
```
당신은 코드 성능 향상을 위해 코드를 진화시키는 임무를 맡은 전문 프로그래머입니다.

현재 코드를 분석하고 SEARCH/REPLACE 형식을 사용하여 개선 사항을 제안하는 것이 당신의 임무입니다.

다음 형식으로 응답하세요:
<<<<<<< SEARCH
[교체할 정확한 코드]
=======
[개선된 대체 코드]
>>>>>>> REPLACE

다음에 집중하세요:
1. 알고리즘 개선
2. 성능 최적화
3. 코드 단순화
4. 버그 수정

프로그램의 성능을 향상시킬 수 있는 의미 있는 변경을 수행하세요.
```

**사용자 메시지:**
```
현재 코드:
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

피보나치 함수의 성능을 개선해주세요. 현재 지수적 시간 복잡도를 가지고 있습니다.

SEARCH/REPLACE 형식을 사용하여 개선 사항을 제안해주세요.

영감을 위한 고성능 코드 예시들입니다:
Inspiration 1:
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]
```

### 6.2 재작성 프롬프트 완전한 예시

**시스템 메시지:**
```
당신은 성능 향상을 위해 코드를 재작성하는 임무를 맡은 전문 프로그래머입니다.

현재 코드를 분석하고 동일한 목표를 달성하지만 더 나은 성능을 가진 완전히 새로운 구현을 만드세요.

설명이나 마크다운 형식 없이 새로운 코드만 반환하세요.
```

**사용자 메시지:**
```
현재 코드:
def matrix_multiply(A, B):
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            sum_val = 0
            for k in range(len(B)):
                sum_val += A[i][k] * B[k][j]
            row.append(sum_val)
        result.append(row)
    return result

행렬 곱셈 알고리즘을 최적화해주세요.

성능이 개선된 이 코드의 완전한 재작성을 제공해주세요.
```

---

## 7. 프롬프트 설계 철학

### 7.1 HypoEvolve의 프롬프트 특징

1. **단순성**: OpenEvolve에 비해 매우 간결하고 직관적
2. **명확성**: 각 프롬프트의 목적이 명확히 구분됨
3. **유연성**: 컨텍스트와 영감 코드를 동적으로 추가 가능
4. **일관성**: 모든 프롬프트가 성능 향상에 집중

### 7.2 OpenEvolve와의 차이점

| 측면 | HypoEvolve | OpenEvolve |
|------|------------|------------|
| **복잡도** | 단순함 (4개 기본 템플릿) | 복잡함 (8개+ 템플릿) |
| **진화 히스토리** | 미포함 | 포함 (이전 시도 기록) |
| **메트릭 정보** | 미포함 | 포함 (성능 메트릭) |
| **다양성 관리** | 영감 코드만 | 다양한 프로그램 예시 |
| **템플릿 변형** | 정적 | 동적 (확률적 변형) |

### 7.3 프롬프트 효율성

**장점:**
- 빠른 응답 시간 (간결한 프롬프트)
- 명확한 지시사항으로 일관된 출력
- 적은 토큰 사용량

**단점:**
- 진화 컨텍스트 부족
- 메트릭 기반 개선 지침 없음
- 다양성 관리 제한적

---

## 8. 프롬프트 개선 제안

### 8.1 단기 개선 사항

1. **메트릭 정보 추가**
```python
MUTATION_USER_TEMPLATE_ENHANCED = """현재 코드:
{current_code}

현재 성능 메트릭:
{metrics}

{context}

SEARCH/REPLACE 형식을 사용하여 개선 사항을 제안해주세요."""
```

2. **진화 히스토리 추가**
```python
EVOLUTION_CONTEXT_TEMPLATE = """

이전 시도들:
{previous_attempts}

최고 성능 코드들:
{top_programs}"""
```

### 8.2 장기 개선 사항

1. **동적 프롬프트 생성**: 성능에 따른 프롬프트 조정
2. **메타 프롬프팅**: LLM이 프롬프트 일부를 생성
3. **컨텍스트 인식**: 문제 도메인별 특화 프롬프트
4. **다국어 지원**: 언어별 최적화된 프롬프트

---

## 9. 사용 가이드라인

### 9.1 프롬프트 선택 기준

**코드 변이 사용 시기:**
- 기존 코드가 대체로 올바른 방향일 때
- 특정 부분만 최적화가 필요할 때
- 안정적인 개선을 원할 때

**완전 재작성 사용 시기:**
- 기존 접근법이 근본적으로 비효율적일 때
- 알고리즘 자체를 바꿔야 할 때
- 혁신적인 개선이 필요할 때

### 9.2 컨텍스트 작성 팁

1. **구체적인 목표 명시**: "성능 향상" → "시간 복잡도 O(n²)에서 O(n log n)으로 개선"
2. **제약 조건 포함**: "메모리 사용량 최소화", "외부 라이브러리 사용 금지"
3. **테스트 케이스 언급**: "큰 입력에서도 빠르게 동작해야 함"

### 9.3 영감 코드 활용

```python
# 효과적인 영감 코드 예시
inspirations = [
    "# 메모이제이션 패턴\ndef func_with_cache(n, cache={}):\n    if n in cache:\n        return cache[n]\n    # 계산 로직\n    cache[n] = result\n    return result",
    "# 동적 프로그래밍 패턴\ndef dp_solution(n):\n    dp = [0] * (n + 1)\n    for i in range(1, n + 1):\n        dp[i] = # 점화식\n    return dp[n]"
]
```

---

## 10. 결론

HypoEvolve의 프롬프트 시스템은 **단순성과 효율성**에 중점을 둔 설계입니다. 복잡한 진화 컨텍스트보다는 명확한 지시사항을 통해 일관된 코드 개선을 추구합니다.

**핵심 특징:**
- 4개의 핵심 템플릿으로 구성
- SEARCH/REPLACE와 완전 재작성 두 가지 전략
- 동적 컨텍스트 및 영감 코드 지원
- 성능 향상에 특화된 지시사항

**향후 발전 방향:**
- 메트릭 기반 프롬프트 개선
- 진화 히스토리 통합
- 도메인별 특화 프롬프트
- 다국어 및 다양성 지원 강화 