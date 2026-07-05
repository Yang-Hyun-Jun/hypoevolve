# ⚡ Coulomb Archive — 전기적 반발장 기반 가설 아카이빙

본 방법론은 아카이브의 각 가설을 **점전하(point charge)** 로 취급하고, 전자기학의 **Coulomb 반발**을 그대로 아카이빙과 부모 샘플링에 적용하는 초경량 아키텍처이다. 필요한 수학은 `1/r²` 하나이고, 하이퍼파라미터는 반발 강도 `γ` 하나이다.

---

## 💡 1. 핵심 철학: "붐비는 곳은 밀어낸다"

기존 접근들은 다양성을 확보하기 위해 무거운 장치를 도입했다.

- **MAP-Elites**: 사람이 그은 격자 축(coverage × complexity 등)
- **EvoDiverse (Parallel Tempering)**: 온도가 다른 두 pool + MH 스왑
- **DPP 계열**: 부분집합 위의 행렬식 확률 분포 + PSD 커널

Coulomb Archive의 관점은 훨씬 단순하다.

> **"고품질이더라도 이미 붐빈 곳에 있으면 손해를 봐야 한다. 그리고 그 손해는 pairwise 반발장으로 정의된다."**

물리적 직관: 같은 부호의 점전하는 거리 제곱에 반비례하는 힘으로 서로 밀친다. 아카이브를 이 반발장 위에 얹으면, 밀도 높은 클러스터는 자연스럽게 감쇄되고 빈 골짜기가 상대적으로 유리해진다.

---

## 🔬 2. 물리학적 배경

전자기학에서 점전하 `q_i` 가 위치 `r` 에 만드는 potential 은

```
φ_i(r) = q_i / |r - r_i|
```

이다. 여러 전하가 있을 때 총 potential 은 각 전하의 기여를 그대로 더한 합. 이 **선형 중첩 원리(linear superposition)** 덕에 pairwise 합만으로 전체 field 를 알 수 있다.

가설 아카이브에 그대로 대응시키면:

- **점전하** = 아카이브에 저장된 가설 `h_i`
- **전하량** `q_i` = 그 가설의 quality score
- **거리** `|r - r_i|` = 두 가설 사이의 논리적 거리 (tree kernel 기반 `1 - K`)
- **Potential** `φ(h)` = 이 지점의 "혼잡도"

Boltzmann 스타일의 확률 가중치를 얹으면 아카이브·샘플링 규칙이 자동으로 유도된다.

---

## 🏗️ 3. 딱 두 개의 규칙

Coulomb Archive 는 전체 알고리즘이 **두 개의 수식**으로 끝난다.

### 3.1 거리 정의

두 ELG 트리 `h_i, h_j` 사이의 논리적 거리:

```
d(h_i, h_j) = 1 - K_tree(h_i, h_j)
```

`K_tree` 는 학습 없이 ELG 구조 위에서 직접 계산되는 자체 정규화된 tree kernel (값이 이미 `[0, 1]`). Collins-Duffy 계열이지만 HypoEvolve 의 ELG semantics 에 맞춰 두 가지로 확장:

**(a) Soft Jaccard for AND/OR children.** 자식을 단순 sum 하지 않고 자카드로 결합. 남는 자식이 명시적으로 합집합에 포함되어 유사도를 낮춤:

```
                        intersection(c1, c2)
K_AND/OR(c1, c2) = ─────────────────────────────────────────
                    |c1| + |c2| - intersection(c1, c2)
```

여기서 `intersection` 은 자식들 사이 greedy best-match sum.

예:
- `AND(A, B)` vs `AND(A, B, C)` → `inter=2`, `union=3`, `K = 0.67`. 추가된 자식 `C` 가 union 을 명시적으로 키워 유사도를 자연스럽게 낮춤.
- `AND(A, B)` vs `AND(A, B, if_1, if_2, if_3)` → `K = 0.4` 로 클러터가 훨씬 강하게 감지됨.

**(b) Wrapper Descent for kind mismatch.** Kind 가 다를 때 통째로 0 이 아니라, wrapper 를 벗겨 내부 자식과 비교하되 decay 를 곱:

- `atomic` vs `AND/OR(children)`: wrapper 의 자식들 중 best match 를 찾아
  ```
  K = λ_wrap · max_c K(atomic, c),   λ_wrap = 0.5
  ```
- `NOT(child)` vs anything:
  ```
  K = λ_neg · K(child, other),   λ_neg = 0.3
  ```

이 조합으로 depth 차이(예: `atomic(X)` vs `AND(atomic(X), atomic(Y))`)가 부드럽게 처리되면서도 극성 반전(NOT 삽입)은 강하게 페널티 받음.

**(c) Relation nodes**: `IMPLIES/CONTRADICT/CORRELATE` 는 자식 두 개(condition, target)의 유사도 평균. Position-aware 이므로 `IMPLIES(A, B)` 와 `IMPLIES(B, A)` 는 자연스럽게 다르게 매핑됨.

```
                        K(cond_1, cond_2) + K(tgt_1, tgt_2)
K_relation(h_1, h_2) = ─────────────────────────────────────
                                     2
```

결과 `K_tree ∈ [0, 1]` 이 자체 정규화되어 있어 외부 정규화 나눗셈이 불필요. Sentence-BERT 나 Tree-LSTM 같은 무거운 인코더 필요 없음.

### 3.2 국소 반발 potential

가설 `h` 가 아카이브 `A` 안에서 느끼는 반발 potential:

```
              ┌─          score(h')
U(h; A)  =    │       ─────────────────       for h' ∈ A, h' ≠ h
              └─      d(h, h')² + ε
```

`ε` 은 자기 자신이나 매우 가까운 이웃에서의 발산을 막는 작은 상수 (예: `10⁻³`).

### 3.3 규칙 1 — 부모 샘플링

```
P(parent = h)  ∝  score(h) · exp(-γ · U(h; A))
```

- score 가 높으면 뽑히기 쉬움 (exploitation)
- `U` 가 높으면 (붐빈 지역) 뽑히기 어려움 (exploration)
- `γ` 하나로 두 압력의 균형 조절

### 3.4 규칙 2 — 아카이브 편입/축출

새 자식 `h*` 가 평가된 후, 그 자식의 **net contribution**:

```
Δ(h*) = score(h*) - γ · U(h*; A)
```

- `Δ(h*) > τ` 이면 아카이브에 편입
- 아카이브가 상한 `K` 에 도달했다면 **가장 낮은 Δ 를 가진 기존 멤버를 축출**

즉 "붐빈 곳에 있으면서 score 도 낮은 놈"이 자연스럽게 밀려나간다.

---

## 🧬 4. ELG 트리와의 궁합

Coulomb Archive 는 거리 `d` 만 잘 정의되면 어떤 데이터 형태에도 적용되지만, HypoEvolve 의 ELG 트리에 특히 잘 맞는다.

- **Tree kernel 로 거리 계산 → 학습 코스트 0**: `hypoevolve/elg/ir.py` 의 `AtomicNode / LogicalNode / RelationNode` 를 그대로 재귀 순회.
- **자식 순서 민감성**: `RelationNode` (IMPLIES/CONTRADICT/CORRELATE/SUPPORT) 는 position-aware, `LogicalNode` (AND/OR) 는 order-invariant 로 자연스럽게 분리 처리 가능.
- **파라미터 슬롯 무시 가능**: `{LOOKBACK_WINDOW}` 같은 placeholder 는 atomic 텍스트 유사도가 알아서 매칭.

---

## 🛡️ 5. 두 가지 붕괴 시나리오에 대한 방어

### 붕괴 1. 인과 방향성 역전 (Causal Reversal)

`IMPLIES(A, B)` 와 `IMPLIES(B, A)` 는 같은 어휘를 사용하지만 논리 방향이 반대. Tree kernel 이 relation 자식을 position-aware 로 매칭하므로 `K_tree` 값이 낮아지고, 따라서 `d` 가 커진다. 두 가설이 서로 강하게 반발하지 않으므로 둘 다 아카이브에 살아남을 수 있다 — 즉 시스템이 "중복이라고 오판해서 하나를 폐기"하지 않는다.

### 붕괴 2. 조건절 비대화 (Hypothesis Clutter)

`IMPLIES(A, B)` 와 `IMPLIES(AND(A, IF_1, IF_2, IF_3), B)` 는 핵심은 같지만 후자가 트리 크기·깊이가 훨씬 크다. Soft Jaccard 가 자식 union 을 키워 두 가설 사이 유사도를 낮추고, 여전히 유사도가 높은 경우에는 Coulomb potential 의 `1/d²` 폭발이 강하게 반발시킴. 결과적으로 quality 항 `Δ` 에서 지저분한 쪽이 축출된다.

**보강**: 필요하다면 quality 를 depth-aware 로 감쇄:

```
score_eff(h) = score(h) - λ · depth(h)
```

Coulomb 수식은 그대로 두고 score 정의만 살짝 조정하면 clutter 방어가 확실해진다.

---

## ⚖️ 6. 다른 접근과의 비교

| 기준 | MAP-Elites | EvoDiverse (PT) | DPP-Elites | **Coulomb Archive** |
|---|---|---|---|---|
| 아카이브 구조 | 격자 셀 top-k | 온도별 pool | k-DPP 부분집합 | **연속 반발장** |
| 필요한 수학 | 셀 인덱싱 | MH 스왑, 온도 스케줄 | 행렬식, PSD 커널 | **`1/r²` 합** |
| 축·좌표 필요? | Yes (handcrafted) | No | No | **No** |
| 공간 인지 | Yes (격자) | No | Yes (커널) | **Yes (potential)** |
| 배치 vs 스트리밍 | 스트리밍 | 배치 (swap) | 배치 (k-DPP) | **스트리밍** |
| HypoEvolve iteration (1개씩) 과 궁합 | 좋음 | 어색함 | 어색함 | **자연스러움** |
| 하이퍼파라미터 | bin 경계, k | 온도, swap 주기 | β, k, 커널 종류 | **`γ` 하나** |
| 리뷰어 인지 부하 | 낮음 | 중간 | 높음 | **낮음** |

---

## ⚙️ 7. 하이퍼파라미터

| 이름 | 역할 | 실용 범위 |
|---|---|---|
| `γ` | 반발 강도 (exploration 조절) | `0.1 ~ 10` |
| `τ` | 아카이브 편입 임계 | `0` 근방 (또는 매 세대 quantile) |
| `ε` | 근접 발산 방지 | `10⁻³ ~ 10⁻²` |
| `K` | 아카이브 상한 | `50 ~ 200` |
| `λ` (선택) | depth penalty | `0 ~ 0.5` |

**`γ` 만 하나 튜닝하면** 사실상 모든 exploration-exploitation 트레이드오프를 조절할 수 있다. EvoDiverse 의 온도 스케줄이나 DPP 의 `β` 와 개념적으로 동일한 역할.

---

## ⚠️ 8. 리스크와 대응

1. **`γ` 감도** — 너무 작으면 pure exploitation, 너무 크면 무작위. 초기값 `γ = 1` 부터 시작해 sweep 권장.
2. **거리 `d` 의 스케일 의존성** — Tree kernel 정규화가 잘 되어 있으면 문제 없지만, atomic 유사도 지표를 바꿀 때 재보정 필요.
3. **아카이브 크기 상한** — `U` 계산은 `O(|A|)` 이므로 `|A| = 200` 규모까지는 실시간. 그 이상이면 KD-tree 나 sparse 근사 검토.
4. **거리가 0 인 진짜 중복** — fingerprint 해시로 사전 필터 (HypoEvolve 에 이미 있는 메커니즘) 후 Coulomb 통과.

---

## 🔗 9. 관련 연구와 지적 계보

이 방법론은 완전히 새롭게 등장한 것은 아니다. 학술적으로 다음 계열의 아이디어에 뿌리를 두고 있으며, 그 재조합·재해석·재적용이 contribution 이다.

- **Goldberg & Richardson (1987)** — *Genetic Algorithms with Sharing for Multimodal Function Optimization*. Fitness sharing 의 원조. Coulomb Archive 의 규칙 1 은 이 fitness sharing 을 물리학 언어로 재정식화한 것.
- **Lehman & Stanley (2008)** — *Novelty Search*. Score 를 신경 쓰지 않고 novelty 만으로 아카이빙. Coulomb 은 score × novelty 를 하나의 potential 수식으로 통합.
- **Coulomb interaction / Boltzmann distribution** — 통계역학의 기본. EvoDiverse 가 온도를 pool 에 부여했다면, 우리는 potential 을 아카이브 field 로 부여.
- **Repulsive point processes / Matérn hard-core** — 공간 통계학. DPP 의 조상.
- **Collins & Duffy (2001), Moschitti (2006)** — Tree kernel. 거리 정의의 기반.

**Novelty claim**: 이 아이디어들의 재조합이 (a) LLM 기반 가설 진화에, (b) ELG 트리 구조를 활용해, (c) archiving과 sampling 규칙을 단일 potential 로 통합해 적용된 사례는 없다.

---

## 🎯 10. 최종 요약

> **EvoDiverse 는 통계역학의 온도로 pool 을 계층화했지만 공간을 보지 못한다. MAP-Elites 는 공간을 보지만 축을 손으로 그어야 한다. Coulomb Archive 는 각 가설을 점전하로 두고 pairwise 반발만으로 archiving 과 sampling 을 단일 원리로 통합한다.**

- 수식: `1/r²` 합, Boltzmann 가중치 — **끝**.
- 하이퍼파라미터: `γ` 하나.
- ELG 트리 구조를 tree kernel 로 학습 없이 활용.
- HypoEvolve 의 iteration-당-1개 샘플링 모델과 자연스럽게 합치.
- 리뷰어에게 그림 하나로 설명 가능: "점전하가 반발하는 field."

물리학적 우아함, 극도의 단순성, 실전 iteration 과의 적합성이 세 축을 이룬다.
