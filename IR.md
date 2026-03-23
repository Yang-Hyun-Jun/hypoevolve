# Executable Logic Graph (ELG) IR Specification v3

## 1. 개요 (Overview)

본 문서는 자연어 기반 가설을 **알고리드믹하게 탐색, 변형, 평가**하기 위한
중간 표현(Intermediate Representation, IR)인 **Executable Logic Graph (ELG)**의*명세를 정의한다.

이 IR의 핵심 목적:

* 자연어 가설 → 구조적 IR 변환
* Mutation 기반 가설 탐색
* 데이터 기반 평가 가능

---

## 2. 설계 목표

### 2.1 Executability

모든 가설은 데이터 위에서 평가 가능해야 한다.

### 2.2 Composability

복잡한 가설은 단순한 구성 요소의 조합으로 표현된다.

### 2.3 Mutability

IR은 구조적으로 변형 가능해야 한다.

### 2.4 Domain Agnostic

도메인 독립적 표현

---

## 3. 핵심 설계 원칙

### 3.1 Operator Unified

모든 구성 요소는 operator node로 표현

### 3.2 Feature Agnostic

IR은 feature를 생성하지 않음

### 3.3 Edge-less Semantics

의미는 node(op)가 담당

---

## 4. Graph 정의

```
G = (V)
```

* V: node 집합
* edge는 inputs로 암묵 표현

---

## 5. Node 정의

```
v = (
  op_type,
  inputs,
  params
)
```

* op_type: 연산자 타입 (필수)
* inputs: 입력 노드 리스트 (필수)
* params: 연산에 필요한 추가 파라미터 (선택)

### 5.1 Node 타입 분류

모든 노드는 아래 세 가지 중 하나로 해석된다.

1. Operator Node
2. Relation Node
3. Atomic Node

---

## 6. Atomic Inputs 정의 (중요)

### 6.1 정의

```
A, B, C, ... ∈ AtomicNode
```

Atomic Node는 **논리식에서 더 이상 분해되지 않는 최소 단위의 명제**이다.

---

### 6.2 형식

```
AtomicNode = (
  name: string,
  type: {boolean | numeric | abstract},
  source: {primitive | semantic}
)
```

---

### 6.3 분류

#### (1) Primitive Atomic

명확한 데이터 기반 의미를 가진 명제

예:

```
CLOSE > SMA_20
FUNDING_FEE > 0
```

---

#### (2) Semantic Atomic

자연어로 표현된 추상 명제

예:

```
"변동성이 크다"
"시장 상태가 불안정하다"
```

---

### 6.4 핵심 원칙

* Atomic은 실행 가능한 함수일 필요 없음
* Atomic은 "의미 단위"로 존재
* 실제 해석 및 실행은 LLM이 담당

---

## 7. Operator 정의

### 7.1 Logical Operators

```
AND(inputs)
OR(inputs)
NOT(input)
```

#### 의미

* AND: 모든 입력 명제가 동시에 성립
* OR: 입력 명제 중 하나 이상 성립
* NOT: 입력 명제가 성립하지 않음

---

### 7.2 Relational Operators

```
RELATION(type, inputs)
```

```
type ∈ {IMPLIES, SUPPORT, CONTRADICT, CORRELATE}
```

#### 의미 정의

RELATION은 **논리적 관계를 표현하는 기호적 연산자**이며,
실행 함수가 아니라 **가설의 구조를 정의하는 역할**을 한다.

* IMPLIES(A, B)

  * 의미: "A이면 B이다"

* SUPPORT(A, B)

  * 의미: "A는 B를 지지한다"

* CONTRADICT(A, B)

  * 의미: "A는 B와 반대되는 관계이다"

* CORRELATE(A, B)

  * 의미: "A와 B는 연관되어 있다"

---

#### 입력 구조

* inputs[0]: condition (조건 명제)
* inputs[1]: target (결과 명제)

---

#### 출력

* 없음 (논리 표현이므로 출력 개념 없음)

---

## 9. Mutation

### 9.1 Operator Mutation

```
AND → OR
IMPLIES → CORRELATE
```

### 9.2 Input Mutation

```
AND(A,B) → AND(A,C)
```

### 9.3 Subtree Mutation

```
subtree(v) → subtree(v')
```

### 9.4 Structure Mutation

```
OR(AND(A,B), C) → AND(OR(A,C), B)
```

### 9.5 Arity Mutation

```
AND(A,B) → AND(A,B,C)
```

### 9.6 Negation Mutation

```
A → NOT(A)
```

---

## 10. 예시 (실제 도메인 기반)

### 10.1 Trading (Crypto / Perp)

#### Natural Language

Funding fee가 양수이고 가격이 단기 이동평균보다 높으면 이후 수익률이 양수일 가능성이 높다

#### IR

```
n1 = AND(
  FUNDING_FEE > 0,
  CLOSE > SMA_20
)

h = RELATION(IMPLIES, [n1, RETURN_5M > 0])
```

---

### 10.2 Trading (Volatility Hypothesis)

#### Natural Language

변동성이 높은 상태에서는 급격한 가격 하락이 발생할 가능성이 높다

#### IR

```
n1 = "변동성이 크다"

h = RELATION(IMPLIES, [n1, RETURN_5M < -0.01])
```

---

### 10.3 Macro / Finance

#### Natural Language

금리가 상승하면 주식 시장 수익률은 감소하는 경향이 있다

#### IR

```
n1 = INTEREST_RATE_CHANGE > 0

h = RELATION(CONTRADICT, [n1, EQUITY_RETURN > 0])
```

---

### 10.4 NLP / User Behavior

#### Natural Language

문장이 길고 부정 표현이 포함되면 사용자 만족도가 낮을 가능성이 높다

#### IR

```
n1 = AND(
  TEXT_LENGTH > 100,
  CONTAINS_NEGATION == True
)

h = RELATION(IMPLIES, [n1, USER_SATISFACTION < 0.5])
```

---

### 10.5 Recommendation System

#### Natural Language

사용자가 최근에 특정 카테고리 상품을 많이 클릭했다면 해당 카테고리 상품을 구매할 가능성이 높다

#### IR

```
n1 = CLICK_COUNT(category, window=7d) > 10

h = RELATION(SUPPORT, [n1, PURCHASE(category) == True])
```

---

### 10.6 System / Infra

#### Natural Language

CPU 사용률이 높고 메모리 사용률도 높으면 시스템 장애가 발생할 가능성이 높다

#### IR

```
n1 = AND(
  CPU_USAGE > 0.9,
  MEMORY_USAGE > 0.9
)

h = RELATION(IMPLIES, [n1, SYSTEM_FAILURE == True])
```

---

### 10.7 Hybrid (Semantic + Numeric)

#### Natural Language

시장 상태가 불안정하고 funding fee가 급격히 증가하면 가격이 급락할 가능성이 있다

#### IR

```
n1 = AND(
  "시장 상태가 불안정하다",
  FUNDING_FEE_CHANGE > 0.01
)

h = RELATION(IMPLIES, [n1, RETURN_1M < -0.02])
```

---

## 11. Natural Language 변환

```
AND(A,B) → "A이고 B"
OR(A,B) → "A 또는 B"
NOT(A) → "A가 아니다"
```

```
IMPLIES(A,B) → "A이면 B"
CORRELATE(A,B) → "A는 B와 상관관계가 있다"
```

---

## 12. Pipeline

```
Natural Language
    ↓
IR (ELG)
    ↓
Mutation / Search
    ↓
New IR
    ↓
Evaluation
    ↓
Natural Language
```

---

## 13. 핵심 속성

* Closure under mutation
* Deterministic execution
* Composability
* Interpretability

---

## 14. 결론

모든 것은 operator node로 표현되며,
가설은 구조이고 mutation은 탐색이다.
