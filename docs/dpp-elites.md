# 🧲 DPP-Elites — 페르미온적 배타 원리 기반 가설 아카이빙

본 방법론은 **Determinantal Point Process(DPP)**를 가설 진화 알고리즘의 아카이빙·샘플링 엔진으로 도입한 아키텍처이다. DPP는 원래 양자역학에서 **페르미온(fermion)의 파울리 배타 원리**를 기술하기 위해 정립된 확률 분포로, "서로 가까운 개체는 동시에 존재하기 어렵다"는 물리 법칙을 확률로 정확히 표현한다. 이 물리적 반발(anti-clustering)을 그대로 가설 공간에 적용하여, **다양성(diversity)을 수학적 법칙으로 강제하는** 아카이빙 메커니즘을 구축한다.

---

## 💡 1. 핵심 철학: "다양성은 통계량이 아니라 확률 법칙이다"

기존 접근들은 다양성을 **사후 통계**로만 다루었다.

- **MAP-Elites**: `(coverage, complexity)`처럼 사람이 임의로 지정한 축으로 격자를 만들고 셀마다 top-k 유지. → 축이 다양성을 정말 포착하는지 불분명, 격자 밖 다양성은 잡히지 않음.
- **EvoDiverse (Parallel Tempering)**: 서로 다른 온도의 풀들 사이 MH 스왑으로 선택 압력을 이완. → 공간 개념이 없어 "어디가 비어 있는가"를 저격하지 못함(spatial blindness).

DPP-Elites의 시각은 다르다.

> **"아카이브는 부분집합 위의 확률 분포다. 다양성은 이 분포의 정의역에서부터 물리 법칙처럼 강제된다."**

DPP가 정의하는 분포는 부분집합의 **행렬식(determinant)**에 비례하며, 이 값은 부분집합의 "부피"에 해당한다. 서로 유사한 원소들이 뭉치면 행렬식이 급격히 작아지므로, 다양성이 확률의 정의 자체에서 자연스럽게 튀어나온다.

---

## 🔬 2. 배경: DPP는 어디서 왔는가

### 물리학적 기원

양자역학에서 페르미온은 파울리 배타 원리에 따라 **동일한 양자 상태에 두 입자가 동시에 존재할 수 없다**. 이 원리를 여러 페르미온 위치의 결합 확률로 표현하면, 그 확률은 파동함수의 행렬식(Slater determinant)의 절댓값 제곱으로 주어진다. 두 페르미온이 같은 자리에 있으면 행렬식이 0이 되어 확률이 사라진다. 자연이 다양성을 강제하는 것이다.

### 머신러닝 학계로의 이식

Kulesza & Taskar (JMLR 2012)가 DPP를 요약 문서 추출, 다양성 있는 추천, 능동 학습으로 이식했다. 그 이후 DPP는 다음과 같은 목적으로 활발히 쓰인다.

- 다양성 있는 부분집합 샘플링 (diverse subset selection)
- Batch active learning의 배치 구성
- Bayesian optimization에서 candidate 다양화
- 뉴럴넷 pruning의 필터 선택

**핵심**: DPP는 "quality × diversity" 트레이드오프를 하나의 커널 행렬 L에 우아하게 분해해 담는 유일한 확률 분포다.

---

## 🏗️ 3. 아키텍처: L-Ensemble on ELG Trees

### 3.1 L-Ensemble 커널의 정의

임의의 부분집합 $S \subseteq \{h_1, \dots, h_N\}$에 대해 DPP는 다음 확률을 부여한다.

$$P(S) \propto \det(L_S)$$

여기서 $L$은 대칭 양의 준정부호 커널 행렬이고, $L_S$는 $S$에 해당하는 부분 행렬이다.

우리는 $L$을 다음과 같이 **품질(quality) × 유사도(similarity) × 품질** 로 분해한다.

$$L_{ij} = q(h_i) \cdot K(h_i, h_j) \cdot q(h_j)$$

- $q(h_i) = \exp(\beta \cdot \text{combined\_score}(h_i))$ — 품질 항. β는 exploration/exploitation 조절 다이얼.
- $K(h_i, h_j) \in [0, 1]$ — 두 ELG 사이의 유사도 커널. 대각 성분은 1.

이 분해의 위력은 명시성이다. 품질 다이얼 β를 키우면 고품질에 쏠리고, 커널 K가 유사한 가설끼리 반발시킨다. 두 압력이 하나의 행렬 안에서 자연스럽게 균형을 이룬다.

### 3.2 유사도 커널 K: Tree Kernel

**중요한 설계 결정: ELG 트리를 벡터로 압축하지 않는다.**

Tree-LSTM이나 Hyperbolic embedding 같은 학습·수치 압축 없이, 두 트리의 유사도를 직접 계산하는 **Tree Kernel**(Collins & Duffy 2001, Moschitti 2006)을 사용한다.

$$K_{\text{tree}}(T_1, T_2) = \sum_{n_1 \in T_1} \sum_{n_2 \in T_2} \Delta(n_1, n_2)$$

여기서 $\Delta(n_1, n_2)$는 두 노드를 뿌리로 하는 공통 부분트리(subtree)의 개수를 재귀적으로 세는 함수다. HypoEvolve의 ELG에 맞추어 다음과 같이 정의한다.

- **AtomicNode**: 두 atomic 노드의 `name` 텍스트를 Sentence-BERT 코사인 유사도로 매칭. 0.9 이상이면 동일 노드로 간주.
- **LogicalNode (AND/OR)**: 자식들의 **집합 매칭(bag-of-children)**으로 계산 → 교환법칙 반영.
- **LogicalNode (NOT)**: 자식 하나에 대한 재귀. 극성 정보 유지.
- **RelationNode (IMPLIES/CONTRADICT/...)**: 자식 두 개를 **순서에 민감하게(condition-first)** 매칭 → 비교환성 반영.

이 설계는 다음 두 가지를 자동으로 만족한다.

- `AND(X, Y)`와 `AND(Y, X)`는 동일한 커널 값을 갖는다.
- `IMPLIES(A, B)`와 `IMPLIES(B, A)`는 완전히 다른 커널 값을 갖는다.

즉 Gemini 대화에서 문제 삼았던 **인과 역전 붕괴**는 커널 정의에서 원천 차단된다. Tree-LSTM을 학습시킬 필요가 전혀 없다.

### 3.3 왜 트리 커널이 HypoEvolve와 궁합이 좋은가

- **ELG는 이미 검증된 스키마의 트리** — arity 제약(relation=2, NOT=1, AND/OR≥2)과 depth가 얕은 편이라 tree kernel 연산이 저렴(수백 마이크로초).
- **파라미터 슬롯이 있는 atomic 텍스트**(`ENTITY_A_ZSCORE_W{LOOKBACK_WINDOW}@t < {NEG_Z_THRESHOLD}`)도 Sentence-BERT 매칭으로 자연스럽게 처리.
- **학습 데이터 불필요** — 매 실행마다 다른 도메인·데이터셋에 대해 튜닝 없이 즉시 작동.

---

## 🔄 4. 알고리즘 파이프라인

DPP-Elites는 매 세대 다음 4단계를 반복한다.

```
[1. Kernel 계산] → [2. 아카이브 유지: k-DPP 부분집합] → [3. 부모 선택: DPP 샘플링] → [4. 자식 통합: marginal gain]
```

### ① Step 1: Kernel 행렬 갱신

새 자식 가설 $h^*$가 평가되어 들어오면, 후보 풀 $\mathcal{C}$ (아카이브 + 최근 후보들)에 대해 L 행렬의 새 행/열만 계산한다.

- $K(h^*, h_i)$를 모든 $h_i \in \mathcal{C}$에 대해 tree kernel로 계산
- $q(h^*) = \exp(\beta \cdot \text{combined\_score}(h^*))$

전체 재계산이 아닌 rank-1 업데이트이므로 O(|𝒞|·tree_kernel_cost)로 저렴.

### ② Step 2: 아카이브 유지 — k-DPP로 재샘플링

아카이브 크기 상한 $k$가 설정되어 있을 때, 전체 후보 풀에서 정확히 $k$개를 뽑는 **k-DPP**(Kulesza & Taskar 2011)로 아카이브를 재구성한다.

$$P(A) \propto \det(L_A), \quad |A| = k$$

이렇게 하면 아카이브는 **"고품질이면서 서로 최대한 다른"** k개의 조합으로 자동 수렴한다. MAP-Elites의 셀별 top-k 규칙이나 EvoDiverse의 온도 스왑 없이도 다양성이 유지된다.

효율화: k-DPP의 정확 샘플링은 $O(N k^2)$인데, greedy MAP 근사(Nemhauser 형식)를 쓰면 $O(N k)$로 떨어지고 실전 품질도 거의 동일하다.

### ③ Step 3: 부모 선택 — DPP-Weighted Sampling

다음 세대 부모를 뽑을 때, 아카이브 내부에서 다시 DPP로 서로 다른 영역의 대표를 소수(예: 3~5개) 샘플링한다. 이후 각 대표 안에서 score-weighted로 최종 하나를 뽑는다.

이 이중 구조는 다음을 보장한다.

- 부모 후보군이 서로 상이한 논리 구조·개념 영역을 대표
- 최종 부모는 그 대표들 중 고득점 → **exploration이 exploitation을 잠식하지 않음**

### ④ Step 4: 자식 통합 — Marginal Gain

새 자식이 들어올 때 아카이브에 편입할지 결정한다. 기준은 **log-determinant marginal gain**.

$$\Delta_i = \log \det(L_{A \cup \{i\}}) - \log \det(L_A)$$

이 값은 정확히 **"이 자식이 아카이브에 얼마나 새로운 정보를 추가하는가"**를 측정한다. Marginal gain이 임계값을 넘으면 편입, 아니면 폐기. MAP-Elites의 fingerprint 기반 중복 필터가 여기서 자연스럽게 일반화된다.

---

## 🛡️ 5. 두 가지 붕괴 시나리오 방어

### 붕괴 1. 인과 방향성 역전 (Causal Reversal)

- **문제**: `IMPLIES(A, B)`와 `IMPLIES(B, A)`가 동일 벡터로 매핑되면 하나가 중복으로 폐기됨.
- **방어**: Tree kernel의 relation 노드에서 자식 순서에 민감한 매칭 규칙을 강제. 두 트리의 $K$ 값이 완전히 달라져 DPP는 둘을 별개 개체로 인지.

### 붕괴 2. 조건절 비대화 (Hypothesis Clutter)

- **문제**: `AND(X, IF_1, IF_2, ..., IF_n)`처럼 조건절이 누더기로 붙어도 핵심 단어가 겹쳐 유사 가설로 뭉침.
- **방어**: Tree kernel은 부분트리 매칭 개수를 세므로, **트리가 커질수록 자기 자신과의 매칭 수가 급격히 커진다(K(h,h) 증가)**. 정규화된 커널 $\tilde{K}(h_i, h_j) = K(h_i,h_j)/\sqrt{K(h_i,h_i) K(h_j,h_j)}$을 쓰면 지저분한 가설은 깔끔한 가설과 낮은 유사도를 갖게 되어 다른 영역으로 분리된다.
- **추가 방어**: quality 항 $q$에 depth penalty를 곱할 수도 있다. $q(h) = \exp(\beta \cdot \text{score}(h) - \gamma \cdot \text{depth}(h))$.

---

## ⚔️ 6. EvoDiverse / MAP-Elites와의 비교

| 기준 | MAP-Elites | EvoDiverse (Parallel Tempering) | **DPP-Elites (Ours)** |
|---|---|---|---|
| 물리학 기원 | — (기하학) | 통계역학 (온도) | **양자역학 (페르미온 배타)** |
| 다양성 정의 | 격자 셀 점유 | 온도별 풀 분리 | **확률 분포의 정의역** |
| 축·좌표 필요? | Yes (handcrafted) | No | **No** (kernel만 필요) |
| 공간 인지 | Yes (기하학적 격자) | No (spatial blindness) | **Yes (커널 기반)** |
| 학습 필요? | No | No | **No** (tree kernel) |
| 다양성-품질 분해 | 암묵적 | 온도 스케줄 | **명시적** (L = qKq) |
| ELG 구조 활용 | 노드 수만(complexity 축) | 전혀 활용 안 함 | **직접 활용** (tree kernel) |
| 중복 자식 필터 | Fingerprint 해시 | 없음 | **Marginal gain** (연속적 판정) |

---

## ⚙️ 7. 하이퍼파라미터와 조작 다이얼

| 파라미터 | 역할 | 실용 범위 |
|---|---|---|
| $\beta$ | quality 강도 (exploitation) | 1.0~10.0 |
| $\gamma$ | depth penalty | 0.0~0.5 |
| $k$ | 아카이브 크기 | 50~200 |
| $m$ | 부모 후보군 크기 | 3~5 |
| 커널 종류 | Subtree / SST / Partial-tree | 실험적 결정 |
| 유사도 임계 | atomic 매칭 컷오프 | 0.85~0.95 |

$\beta$ 하나만 조절해도 EvoDiverse의 온도 스케줄과 유사한 exploration-exploitation 제어가 가능하다. 하지만 축(kernel) 자체가 다양성을 정의하므로 EvoDiverse보다 훨씬 정교하다.

---

## ⚠️ 8. 리스크와 대응

1. **k-DPP 샘플링 복잡도** — Naive는 $O(Nk^2)$. 실전에서는 greedy MAP 근사($O(Nk)$)로 충분히 근사 가능. 아카이브 크기 100~200 규모에서는 문제되지 않음.
2. **콜드 스타트** — 아카이브가 매우 작을 때 DPP는 무의미. 초기 20~30개까지는 uniform 또는 단순 UCB로 운영, 이후 DPP 전환.
3. **Tree Kernel 선택 민감도** — Subtree kernel(엄격) vs Subset-tree kernel(유연) vs Partial-tree kernel(가장 유연). Ablation study 필수.
4. **커널의 양의 정부호성(PSD)** — Sentence-BERT 유사도를 그대로 넣으면 PSD가 깨질 수 있음. Gram matrix로 변환하거나 RBF 커널로 감싸는 후처리 필요.

---

## 🔗 9. 관련 연구

- **Kulesza & Taskar (2012)** — *Determinantal Point Processes for Machine Learning*. Foundational monograph.
- **Kulesza & Taskar (2011)** — *k-DPPs: Fixed-Size Determinantal Point Processes*. ICML.
- **Collins & Duffy (2001)** — *Convolution Kernels for Natural Language*. NeurIPS. Tree kernel 원조.
- **Moschitti (2006)** — *Efficient Convolution Kernels for Dependency and Constituent Syntactic Trees*. ECML. Partial tree kernel.
- **Chen et al. (2018)** — *Fast Greedy MAP Inference for Determinantal Point Process*. NeurIPS. 실전 근사 알고리즘.
- **Mouret & Clune (2015)** — *Illuminating Search Spaces by Mapping Elites*. MAP-Elites 원조.
- **EvoDiverse (ICML 2026)** — Parallel tempering 기반 가설 탐색. 본 방법론의 직접 비교 대상.

---

## 🎯 10. 최종 요약

> **EvoDiverse는 열역학의 온도로 선택 압력을 이완했고, MAP-Elites는 사람이 그은 격자로 다양성을 강제했다. DPP-Elites는 페르미온적 배타를 통해 다양성 자체를 확률 법칙으로 승격시킨다.**

- ELG 트리를 벡터로 압축하지 않고 tree kernel로 직접 활용 → 학습 코스트 0.
- Quality × Similarity 커널 분해로 exploration-exploitation을 명시적으로 튜닝.
- 인과 역전, 조건절 비대화 붕괴를 커널 정의만으로 원천 차단.
- MAP-Elites의 handcrafted 축 문제와 EvoDiverse의 spatial blindness를 동시에 해결.

물리학적 우아함, ELG 구조와의 자연스러운 결합, 리뷰어에게 익숙한 이론적 기반이 세 축을 이룬다.
