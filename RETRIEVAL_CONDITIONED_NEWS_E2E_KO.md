# LLM 기억 위반 기반 금융 뉴스 필터: 설계부터 최종 검증까지

> 문서 목적: 이 연구에서 실제로 무엇을 했고, 왜 그렇게 설계했으며, 각 단계의 출력이 다음 단계로 어떻게 이어졌는지를 실제 prompt, 기사, activation, 점수와 최종 금융 결과를 사용해 처음부터 끝까지 설명한다.
>
> 기준 시점: 2026-07-23에 동결된 최종 artifact.
>
> 최종 감사 상태: `RETRIEVAL_CONDITIONED_COMPLETION_AUDIT_PASSED`
>
> 최종 결과 상태: `LIMITED_POSITIVE_JULY_POINT_ESTIMATE`

---

## 0. 가장 먼저 볼 결론

우리가 최종적으로 측정한 것은 다음이다.

> LLM이 어떤 기업에 관해 뉴스 이전부터 가지고 있던 기업별 기억을 먼저 꺼낸 뒤, 새 기사가 그 기억과 반대되는 정도를 계산하면, 일반적인 텍스트 novelty나 embedding을 넘어 중요한 금융 뉴스를 거르는 데 도움이 되는가?

이 질문에서 가장 중요한 설계 선택은 다음 세 가지다.

1. **주가, residual, volatility로 activation 방향을 학습하지 않았다.**
2. **기사를 보기 전에 기업별 기억을 먼저 추출했다.**
3. **“모델이 모르는 기업”과 “알고 있던 내용에 반하는 기사”를 분리했다.**

전체 흐름은 다음과 같다.

```text
기업 이름/CIK/ticker universe
    ↓
뉴스 이전 기업 기억 회수
    ├─ A/B/C/D 방향 기억
    └─ 8개 관계의 자유회상 프로필
    ↓
기억의 존재·안정성·기업 특이성 측정
    ├─ memory strength M
    └─ coverage gap U = 1 - M
    ↓
새 기사에서 하나의 완료 사건과 방향 추출
    ↓
기억과 기사의 관계를 세 경로로 측정
    ├─ behavioral violation
    ├─ activation-geometry violation
    └─ free-profile semantic violation
    ↓
기사별 memory feature 동결
    ↓
그 이후에만 과거 가격을 열어 개발 모델 선택
    ↓
모델까지 동결한 후 July confirmation 가격을 처음 열어 검증
```

최종 July confirmation은 383개 기사, 343개 ticker, 9개 event date로 구성됐다.

- `all_memory - strong_text`의 log-target test `ΔR² = +0.001283`
- event-date block bootstrap 95% CI: `[-0.014568, +0.019820]`
- `P(ΔR² ≤ 0) = 0.466`
- 중요 뉴스 분류 AUC 증분: `+0.003117`

점추정치는 양수지만 신뢰구간이 0을 포함한다. 따라서 결과는 **제한적 양의 증거**이지, “기억 위반 필터가 확실히 시장 반응을 예측한다”는 결론이 아니다.

### 단계 목적 recap

이 연구의 목적은 변동성을 설명하는 activation을 뒤에서 맞춰 찾는 것이 아니라, **시장 label 없이 먼저 LLM의 기업 기억과 기억 위반을 정의하고, 그 고정된 신호의 금융적 유용성을 나중에 검증하는 것**이다.

---

## 1. 무엇을 측정했고 무엇을 측정하지 않았는가

### 1.1 핵심 단위는 ticker 문자열이 아니라 `기업 × 관계 × 기사`다

모델에게 단순히 `MSFT`라는 ticker token을 주고 activation을 뽑은 것이 아니다.

- 기억 prompt에는 원칙적으로 canonical company name을 사용했다.
- 예: `MICROSOFT CORP`, `Walmart Inc`, `RIO TINTO PLC`
- ticker와 CIK는 모델 기억의 입력이라기보다, 기억을 실제 시장 event와 정확히 연결하는 식별자다.
- 기사 하나가 여러 관계를 포함하면 관계별 점수를 만든 뒤 기사 수준에서 최대값으로 집계했다.

관계는 다음 8개로 고정했다.

| 관계 | 의미 |
|---|---|
| `earnings` | 실적 |
| `guidance` | 가이던스·전망 |
| `contract_mna` | 계약·파트너십·M&A |
| `regulatory_legal` | 규제·법률 |
| `product_operations` | 제품·임상·프로젝트·운영 |
| `capital_policy` | 배당·자사주·부채·자금조달 |
| `leadership_workforce` | 경영진·인력 |
| `cyber_safety` | 사이버·장애·안전 |

따라서 질문은 다음처럼 구체적이다.

```text
“Microsoft를 아는가?”가 아니라
“모델이 Microsoft의 leadership/workforce에 관해
안정적이고 기업 특이적인 기대를 가지고 있는가?”
```

그 다음에야 다음을 묻는다.

```text
“Microsoft가 4,800명을 감원한다는 기사가
그 leadership/workforce 기억을 위반하는가?”
```

### 1.2 coverage gap과 memory violation은 다르다

기업 `f`, 관계 `r`, 기사 `n`에 대해 다음을 분리했다.

```text
M_fr       = 기업 f의 관계 r에 대한 안정적·기업 특이적 기억 강도
U_fr       = 1 - M_fr
C_frn      = 새 기사 n이 회수된 기억과 모순되는 정도
V_frn      = M_fr × C_frn
```

- `U`가 크다: 모델이 그 기업·관계를 잘 모른다.
- `V`가 크다: 모델이 기억을 가지고 있었고, 기사가 그 기억을 거슬렀다.

이 분리가 없으면 tail company의 모든 기사가 “놀라운 기사”로 잘못 올라온다.

### 1.3 novelty와 memory violation도 다르다

- 텍스트 novelty: 기존 기사 corpus와 문장/embedding이 얼마나 다른가.
- memory violation: **해당 기업에 대해 모델이 회수한 사전 기대와 얼마나 다른가.**

같은 “CEO 교체” 기사라도 다음 두 경우는 다르다.

```text
기업 A: 모델이 “10년간 안정적 경영진”을 반복 회상
       → CEO 교체는 기억 위반 후보

기업 B: 모델이 UNKNOWN만 출력
       → CEO 교체는 coverage gap이지 기억 위반이 아님
```

### 단계 목적 recap

이 단계의 목적은 결과 변수를 만들기 전에 construct를 정확히 분리하는 것이다. 이후 모든 계산은 `unknown`, `generic novelty`, `firm familiarity`, `true memory violation`을 서로 대신 사용하지 못하도록 설계됐다.

---

## 2. 선행 단계: recall representation을 처음 어떻게 찾았는가

현재 뉴스 연구에 앞서, 4,000개 SEC CIK 기업을 대상으로 **회사명 → 미국 ticker 자유회상** 실험을 수행했다. 이 단계가 “LLM을 거대한 retrieval system으로 본다”는 현재 설계의 출발점이다.

### 2.1 자유회상 task

실제 prompt 예시는 다음과 같다.

```text
Recall the U.S. trading symbol of Walmart Inc.
Write only the ticker.
```

또는 별도 template:

```text
Which ticker identifies ADVANCED MICRO DEVICES INC on U.S. markets?
Answer using the ticker alone.
```

모델은 최대 8 token을 greedy decoding했다. 정규화한 생성 문자열이 정답 ticker와 정확히 같으면 correct로 판정했다.

### 2.2 arXiv:2510.09033에서 가져온 핵심: FA/AH/UH 분리

[Do LLMs Really Know What They Don't Know?](https://arxiv.org/abs/2510.09033)의 핵심 아이디어를 다음처럼 금융 free recall에 적용했다.

- 마지막 query token이 company subject token에 attention하지 못하도록 모든 layer에서 block한다.
- 원래 출력분포와 block 출력분포의 full-vocabulary Jensen–Shannon divergence를 계산한다.
- 답은 틀렸지만 subject 접근에 강하게 의존했다면 단순 무기억 오류와 구분한다.

클래스 정의는 다음과 같다.

```text
FA: free recall 정답

AH: free recall 오답이지만
    subject-attention block JS가 validation FA 평균보다 큼

UH: free recall 오답이고
    subject-attention block JS가 위 threshold 이하
```

Qwen2.5-7B의 threshold는 validation FA 217개의 평균인 `0.557937`로 동결했다. test는 threshold 설정에 사용하지 않았다.

### 2.3 실제 FA/AH/UH 예시

| 클래스 | 회사 | 정답 | 모델 생성 | subject-block JS | 해석 |
|---|---|---:|---:|---:|---|
| FA | Walmart Inc | WMT | WMT | 0.000339 | 정확한 자유회상 |
| AH | Sumitomo Mitsui Financial Group | SMFG | SMT | 0.615514 | 틀렸지만 회사 subject association에 강하게 의존 |
| UH | Banco Santander | SAN | BND | 0.027131 | 틀렸고 subject association 의존도도 낮음 |

여기서 중요한 점은 AH다. `SMT`는 틀린 답이지만 attention block에 민감하다. 즉 내부적으로는 “아무것도 모르는 상태”와 다를 수 있다. 이 때문에 단순 `correct/incorrect` 방향을 memorization 방향이라고 부르면 안 된다.

### 2.4 recall direction 계산

각 layer에서 company-name token span의 activation을 평균해 subject state `h_i`를 만들었다.

```text
h_i^(l) = layer l에서 company-name token들의 평균 activation
z_i^(l) = h_i^(l) / ||h_i^(l)||_2
```

Primary direction은 FA와 UH만으로 difference-in-means를 계산했다.

```text
w_l =
    mean(z_i^(l) | FA)
  - mean(z_i^(l) | UH)

u_l = w_l / ||w_l||_2

score_i = z_i^(l) · u_l
```

AH는 fitting과 layer selection에서 완전히 제외하고, 동결된 방향 위에서 어디에 놓이는지만 진단했다.

선택 절차는 다음과 같다.

1. train에서 각 layer의 FA-minus-UH 방향을 계산한다.
2. validation FA-vs-UH AUC가 가장 좋은 layer를 고른다.
3. validation에서 decision threshold를 고정한다.
4. test를 한 번 열어 AUC와 bootstrap CI를 계산한다.
5. seed/fold 방향 cosine으로 stability를 확인한다.

### 2.5 실제 projection 예시

Qwen2.5의 선택 layer는 21, validation decision threshold는 `0.018640`이었다.

| 회사 | 클래스 | recall direction projection | threshold 대비 |
|---|---|---:|---|
| Walmart | FA | `+0.144983` | 기억 방향 쪽 |
| Sumitomo Mitsui FG | AH | `+0.020231` | 약하게 기억 방향 쪽 |
| Banco Santander | UH | `-0.040418` | 무기억 방향 쪽 |

이 표는 직관적인 한 사례일 뿐이다. test 전체 평균은 Qwen2.5에서 FA `+0.03168`, AH `-0.02419`, UH `-0.02983`이었다. 즉 특정 AH는 FA처럼 보이지만, Qwen2.5 전체 AH는 대체로 UH 쪽에 가까웠다. Qwen3에서는 AH가 상대적으로 FA에 더 가까웠다. 그래서 이 방향은 모델마다 의미가 같다고 가정하지 않았다.

### 2.6 초기 recall 결과와 한계

| 모델 | 선택 layer | test FA-vs-UH AUC | bootstrap 95% CI | direction stability |
|---|---:|---:|---:|---:|
| Qwen2.5-7B | 21 | 0.724 | [0.673, 0.764] | 0.941 |
| Qwen3-4B | 28 | 0.751 | [0.705, 0.792] | 0.875 |

275개 금융 cohort CIK를 축 fitting 전에 완전히 제외한 transfer-safe 분석에서도 free recall decoding은 유지됐다.

그러나 activation intervention은 두 모델에서 재현되지 않았다.

- FA direction 제거가 두 모델에서 선택적으로 recall을 낮추지 못했다.
- Qwen2.5의 일부 UH add 효과는 Qwen3에서 반전됐다.
- Qwen3의 큰 변화는 semantic recall보다 A/B answer-token 편향으로 해석됐다.

따라서 최종 명칭은 다음처럼 제한했다.

```text
가능한 주장:
stable successful-recall-correlated direction

불가능한 주장:
cross-model causal memorization axis
```

### 2.7 이 선행 단계가 현재 뉴스 설계에 준 영향

초기 recall axis를 그대로 뉴스에 투영해 “surprise”라고 부르지 않았다. 대신 다음 원칙만 가져왔다.

- recall 성공과 subject association reliance를 분리한다.
- 틀린 연상 AH를 무기억 UH와 분리한다.
- 기억의 존재를 먼저 식별한 뒤 새 정보와의 충돌을 본다.
- 축 하나가 실패할 수 있으므로 행동, activation geometry, 자유회상 semantic 경로를 병렬로 둔다.

초기 ticker recall correctness와 subject-block JS는 현재 연구에서 **identity/familiarity diagnostic**으로만 사용했다.

### 단계 목적 recap

이 선행 단계의 목적은 “기억 activation”을 주가로 학습하는 것이 아니라, 실제 자유회상 성공과 subject-dependent association에서 찾는 것이었다. 동시에 개입 실패를 통해 한 개의 보편적 causal axis를 강제하면 안 된다는 경계도 얻었다.

---

## 3. 현재 연구의 사전 가설과 성공 규칙

### 3.1 사전 가설

```text
LLM이 특정 기업·관계에 대해 안정적이고 기업 특이적인 패턴을 회수한다면,
그 패턴과 반대되는 실제 뉴스는 일반 novelty와 다른 정보일 수 있다.

이 memory-violation score가 큰 기사는
나중의 시장 반응 크기와 관련될 가능성이 있다.
```

### 3.2 완전 성공 규칙

July market 단계의 full support는 다음 세 조건을 모두 요구했다.

1. `all_memory - strong_text` test `ΔR² > 0`
2. event-date bootstrap 95% CI lower bound `> 0`
3. 독립적으로 정의한 memory-violation score 중 최소 3개의 July Spearman CI lower bound `> 0`

실제 결과는 1번만 충족했다.

- 양수 점추정치: 충족
- `ΔR²` CI lower > 0: 실패
- 양의 CI lower를 가진 violation score 수: 2개

따라서 미리 정한 명칭대로 `LIMITED_POSITIVE_JULY_POINT_ESTIMATE`가 됐다.

### 단계 목적 recap

이 단계의 목적은 결과를 본 후 성공 기준을 낮추지 못하게 하는 것이다. “점추정치가 양수”와 “재현 가능한 금융 필터”를 명확히 구분했다.

---

## 4. Stage A — label-blind 기사 cohort 만들기

### 4.1 목적

기억 위반을 측정하려면 기사가 실제로 해당 기업의 완료된 사건을 말해야 한다. earnings call 초대, analyst target, 주가가 이미 올랐다는 recap이 섞이면 모델이 기억 위반이 아니라 시장 반응 문구를 읽게 된다.

### 4.2 원천 데이터

- 기업 universe: SEC 기반 4,000 CIK company-to-ticker benchmark
- 뉴스: Yahoo `stock_news.parquet`
- event session 정렬: Yahoo `stock_prices.parquet`의 `symbol`, `report_date`만 사용

이 시점에는 OHLC, return, volume, outcome 값을 읽지 않았다.

### 4.3 1차 선택 규칙

원천 date-range news 490,896행에서 다음 규칙을 적용했다.

1. Yahoo type이 `STORY`
2. UUID-symbol row가 유일
3. Yahoo URL에서 정확한 timestamp 복구
4. 제목에 target ticker/company alias가 직접 등장
5. UUID가 전체에서 정확히 하나의 symbol에 매핑
6. 본문 첫 1,500자에도 target company가 직접 등장
7. 본문 길이 300자 이상
8. 제목이 완료된 corporate event
9. preview, schedule, recommendation, valuation, rating, price recap 제외
10. ticker-event session당 기사 하나만 label-blind ranking으로 선택

한 session 내 ranking에는 다음만 사용했다.

- presswire provenance
- relation specificity
- body quality
- recency
- UUID tie-break

sentiment, volatility, return은 ranking에 사용하지 않았다.

### 4.4 event session 규칙

- 장 시작 전 기사: 해당 거래일
- 09:30 New York 이후 기사: 다음 full trading session
- 개발/검증 경계에는 5 trading-session purge/embargo 적용

초기 13,735개 representation super-cohort split은 다음과 같다.

| split | 기사 수 | session 범위 |
|---|---:|---|
| development train | 5,621 | 2026-01-09 ~ 2026-03-24 |
| boundary excluded | 1,344 | 경계 purge/embargo 구간 |
| development test | 6,110 | 2026-04-09 ~ 2026-06-23 |
| July confirmation | 660 | 2026-07-09 ~ 2026-07-21 |
| 합계 | 13,735 |  |

### 4.5 기사 본문 leakage 제거

raw body는 감사용으로 보존하되, 모든 모델·semantic judge·TF-IDF·embedding 입력에서는 다음 문장을 제거했다.

- 이미 관측된 share-price movement
- analyst price target

총 1,694개 기사에서 3,673개 문장을 제거했다.

### 4.6 v1부터 v9까지 반복한 label-blind audit

기사 규칙은 한 번에 완성되지 않았다. market value gate가 닫힌 상태에서 제목과 본문만 보고 systematic false positive를 제거했다.

| 버전 | 새로 발견한 문제 | 수정 |
|---|---|---|
| v1 | earnings schedule, “offering value”, loan facility 오분류 | schedule/preview 제외, loan을 capital policy로 이동 |
| v2 | date announcement, analyst PT recap, bought-deal | analyst/recap 제외, securities offering을 capital policy로 이동 |
| v3 | dividend가 earnings로 오인, conference notice, price recap | earnings noun 조건 강화 |
| v4 | earnings release date, peer roundup, 비재무 “results” | realized financial/operating result 조건 강화 |
| v5 | call invitation, “reports next week” | anticipatory form 제외 |
| v6 | “shares rallied/slid after”, analyst forecast, executive award | market-reaction/title clue 제외 |
| v7 | 본문 속 주가 반응·target | analysis body sentence redaction |
| v8 | retrospective sector comparison | atomic new issuer event가 아닌 template 제외 |
| v9 | terminal headline rule | 이후 representation super-cohort 고정 |

v9 extraction 이후에도 가격을 열기 전 atomic-claim audit을 수행했다. 기존 13,735개 representation은 폐기하지 않고 보존했으며, 별도의 `analysis_keep` mask만 동결했다.

최종적으로 다음 4,359개 기사를 분석에서 제외했다.

| 대표 제외 이유 | 제외 수 |
|---|---:|
| earnings/results call | 2,953 |
| investment-commentary publisher | 648 |
| post-event deep dive/peer ranking | 214 |
| analyst preview | 193 |
| retrospective peer earnings | 171 |
| earnings/results webcast | 70 |
| 기타 analyst recap·target·rating·시장반응·speculation | 110 |
| 합계 | 4,359 |

최종 분석 cohort는 9,376개 기사다.

### 단계 출력

- `data/retrieval_conditioned_news/documents.parquet`
- 13,735개 원본 representation super-cohort
- raw body와 redacted `analysis_body` 동시 보존
- `observation_id` 하나가 독립 article-ticker-session 하나를 의미

### 단계 목적 recap

Stage A의 목적은 “주가가 많이 움직인 뉴스”를 먼저 고르는 것이 아니라, **해당 기업의 완료된 사건을 말하는 기사만 label-blind하게 고르고 시장반응 문장을 제거하는 것**이었다.

---

## 5. Stage B — 기사별 atomic claim과 관계 고정

### 5.1 목적

긴 기사 전체를 semantic contradiction judge에 넣으면 unrelated background와 recap 문장이 판단을 지배할 수 있다. 그래서 각 `기사 × 관계`마다 하나의 완료 사건 문장을 고정했다.

### 5.2 방법

deterministic selector가 다음 중 하나를 선택했다.

- headline
- market-reaction-redacted body의 한 문장

선택에는 relation pattern, 구체성 cue, boilerplate 제외 규칙만 사용했다. 모델 label과 시장값은 사용하지 않았다.

### 5.3 결과

- super-cohort relation rows: 14,107
- `analysis_keep=True` relation rows: 9,714
- eligible articles: 9,376
- headline claim: 8,158
- body-sentence claim: 5,949
- 최대 claim 길이: 600자

최종 relation 분포는 다음과 같다.

| 관계 | eligible relation rows |
|---|---:|
| earnings | 5,084 |
| capital policy | 2,351 |
| contract/M&A | 1,060 |
| leadership/workforce | 422 |
| product/operations | 348 |
| guidance | 318 |
| regulatory/legal | 130 |
| cyber/safety | 1 |
| 합계 | 9,714 |

### 5.4 실제 atomic claim 예시

Microsoft 감원:

```text
Relation: leadership_workforce
Atomic claim: Microsoft Corporation (MSFT) to Cut 4,800 Jobs
Source: headline
```

Del Monte credit facility:

```text
Relation: capital_policy
Atomic claim:
Del Monte Corporation ... successfully amended its existing senior unsecured
revolving credit facility, increasing total commitments by $150 million to $900 million.
Source: body sentence
```

Rio Tinto 기사는 두 relation으로 나뉘었다.

```text
earnings:
LABRADOR IRON ORE ROYALTY CORPORATION ... RIO TINTO RELEASES
IOC PRODUCTION AND SALES INFORMATION

product_operations:
Rio Tinto released its operations review for the second quarter ...
```

기사 하나가 두 관계를 갖더라도 서로 다른 claim과 점수를 유지한 뒤, 마지막에 각 construct별 최대값을 기사 점수로 사용했다.

### 단계 출력

- `data/retrieval_conditioned_atomic_claims/atomic_claims.parquet`
- 14,107 relation rows
- `analysis_keep`가 behavioral, activation, semantic 세 arm에 동일하게 적용

### 단계 목적 recap

Stage B의 목적은 모든 arm이 정확히 같은 완료 사건을 평가하게 하고, 긴 기사 속 background나 market recap이 기억 위반 판정을 대신하지 못하게 하는 것이다.

---

## 6. Stage C — 뉴스 이전 firm-relation 기억 prompt 만들기

### 6.1 목적

기사를 보고 난 뒤 모델에게 “이 뉴스가 놀라운가?”라고 묻지 않았다. 먼저 기사와 독립적으로 회사별 사전 기대를 회수해야 hindsight contamination을 막을 수 있다.

### 6.2 A/B/C/D 의미

각 관계에서 A/B/C는 relation-specific 방향이고 D는 no memory다.

예를 들어 `leadership_workforce`는 다음과 같다.

```text
A. favorable: leadership or workforce change strengthened execution
B. continuity: an orderly or routine transition occurred
C. adverse: a disruptive departure, layoff, or workforce deterioration occurred
D. no firm-specific memory, unrelated, or insufficient information
```

### 6.3 두 개의 독립 paraphrase

같은 `기업 × 관계`에 대해 두 prompt를 사용했다.

Template v0:

```text
Use only information already stored in your parameters about MICROSOFT CORP.
Do not use or imagine a current article.
If a new leadership or workforce announcement arrived,
which direction is most characteristic relative to the firm's remembered profile?

A. ...
B. ...
C. ...
D. ...
Return only A, B, C, or D.
```

Template v1:

```text
Before seeing any recent news, retrieve what you know about MICROSOFT CORP.
For a hypothetical leadership or workforce update,
select the direction most consistent with that company-specific memory.

A. ...
B. ...
C. ...
D. ...
Return only A, B, C, or D.
```

두 paraphrase가 서로 다른 답을 내면 기억이 안정적이지 않다고 본다.

### 6.4 세 가지 기업 특이성 control

각 real-firm prompt와 같은 option semantics로 다음 control을 만들었다.

1. `Unknown Corporation`
2. 같은 popularity bin의 deterministic shuffled company
3. 마지막 query가 company subject token에 attention하지 못하는 blocked run

예를 들어 Microsoft의 shuffled company가 Broadcom이라면 다음을 비교한다.

```text
P(A,B,C,D | MICROSOFT CORP)
P(A,B,C,D | Unknown Corporation)
P(A,B,C,D | Broadcom)
P(A,B,C,D | MICROSOFT CORP, subject attention blocked)
```

이 control이 필요한 이유는 “대기업이면 늘 B를 고른다” 또는 “모든 adverse event가 희귀하다” 같은 generic pattern을 기업 기억으로 오인하지 않기 위해서다.

### 6.5 prompt package 크기

| prompt package | 행 수 | 의미 |
|---|---:|---|
| expectations | 23,380 | 11,690 firm-relation pair × 2 paraphrase |
| controlled | 21,504 | 896 firm × 8 relation × A/B/C statement |
| news relations | 14,107 | 실제 article-relation post-news 질문 |
| profiles | 6,642 | 3,321 firm × 2 free-profile prompt |

controlled firm은 recall split별로 train 512, validation 192, test 192로 고정했다.

### 단계 출력

- `data/retrieval_conditioned_memory/expectations.parquet`
- `data/retrieval_conditioned_memory/controlled.parquet`
- `data/retrieval_conditioned_memory/news_relations.parquet`
- `data/retrieval_conditioned_memory/profiles.parquet`

### 단계 목적 recap

Stage C의 목적은 실제 기사를 보기 전에 기업별 기억을 회수하고, 그 기억이 단순한 option prior나 대기업 familiarity가 아니라 회사 이름에 특이적인지를 비교할 control을 동시에 만드는 것이다.

---

## 7. Stage D — 모델 inference와 activation 추출

### 7.1 목적

동일한 prompt package에서 다음을 함께 저장했다.

- A/B/C/D logit
- 모든 layer의 last pre-answer hidden state
- Unknown/shuffled/attention-block control
- 자유회상 profile 생성

이렇게 해야 행동 점수와 activation 점수가 서로 다른 입력을 보지 않는다.

### 7.2 사용 모델

| 모델 | frozen snapshot | layer 수 | hidden size |
|---|---|---:|---:|
| Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` | 28 | 3,584 |
| Qwen3-4B | `1cfa9a7208912126459214e8b04321603b3df60c` | 36 | 2,560 |

모든 main prompt의 `max_length=1024`, profile generation의 `max_new_tokens=320`이었다.

### 7.3 activation 위치

각 prompt를 chat template로 rendering한 뒤, 모델이 A/B/C/D 답을 생성하기 직전 마지막 token의 residual hidden state를 저장했다.

```text
h_pre[f,r,t,l] =
  firm f, relation r, paraphrase t의
  answer 직전 layer l hidden state

h_post[f,r,n,l] =
  article n을 본 뒤 같은 A/B/C/D 질문의
  answer 직전 layer l hidden state
```

controlled report도 동일 위치를 저장했다.

### 7.4 실제 array

Qwen2.5:

```text
expectation pre_hidden:  [23,380, 28, 3,584]
controlled hidden:       [21,504, 28, 3,584]
real-news hidden:        [14,107, 28, 3,584]
pre/unknown/shuffled/
blocked logits:          [23,380, 4]
news logits:             [14,107, 4]
```

Qwen3:

```text
expectation pre_hidden:  [23,380, 36, 2,560]
controlled hidden:       [21,504, 36, 2,560]
real-news hidden:        [14,107, 36, 2,560]
```

최종 auditor는 main, semantic judge, embedding extraction의 총 34개 array가 모두 finite인지 확인했다.

### 7.5 실행 안정성

GPU inference 전에 다음 bootstrap을 실행했다.

```bash
bash scripts/bootstrap_tmp_runtime.sh
```

`RUNTIME_BOOTSTRAP_OK`를 확인한 뒤 `/tmp`의 검증된 model copy를 사용했다. main GPU lane은 named tmux session으로 실행하고 checkpoint를 통해 재시작 가능하게 했다.

```text
retrieval_qwen25 → GPU 0
retrieval_qwen3  → GPU 1
```

완료된 checkpoint/manifest가 있으면 같은 row를 다시 계산하지 않았다.

### 단계 출력

- `outputs/retrieval_conditioned_memory/qwen25_7b/`
- `outputs/retrieval_conditioned_memory/qwen3_4b/`
- task별 `manifest.json`, `metadata.parquet`, logit·hidden array

### 단계 목적 recap

Stage D의 목적은 기억, control, controlled report, 실제 기사를 같은 모델·같은 answer 위치에서 측정해 이후 차이가 입력 위치나 extraction 방식 때문이 아니게 만드는 것이다.

---

## 8. Stage E — firm-relation memory strength 계산

### 8.1 목적

D 선택 확률이 낮다는 것만으로 “기억이 있다”고 볼 수 없다. 두 paraphrase가 일치하고, Unknown과 shuffled firm보다 다른 분포를 보여야 한다.

### 8.2 기호

```text
p_f^(0), p_f^(1): real firm의 두 paraphrase A/B/C/D 분포
p_f:              두 분포의 평균
p_u:              Unknown Corporation 분포 평균
p_s:              shuffled firm 분포 평균
JS:               Jensen–Shannon divergence
```

### 8.3 memory strength

```text
known_mass
  = 1 - p_f(D)

paraphrase_consistency
  = 1 - JS(p_f^(0), p_f^(1)) / ln(2)

company_specificity
  = [JS(p_f,p_u) + JS(p_f,p_s)] / [2 ln(2)]

M
  = known_mass
    × paraphrase_consistency
    × sqrt(company_specificity)
```

각 항의 의미는 다음과 같다.

| 항 | 막는 오류 |
|---|---|
| `1-p(D)` | 모델이 스스로 no memory라고 하는 경우 |
| paraphrase consistency | 질문 표현만 바뀌어도 답이 뒤집히는 불안정 기억 |
| firm-vs-Unknown JS | option의 generic prior |
| firm-vs-shuffled JS | popularity나 업종의 generic firm pattern |

coverage gap은 다음처럼 별도로 둔다.

```text
coverage_gap = 1 - M
```

### 8.4 directional memory strength

activation confirmation/contradiction을 정의하려면 “회사에 익숙하다”를 넘어 A/B/C 중 어느 방향을 기대하는지가 선명해야 한다.

먼저 D를 제외하고 A/B/C를 다시 정규화한다.

```text
p_f_ABC = p_f(A:B:C) / sum p_f(A:B:C)
```

그 다음:

```text
expectation_margin
  = max(p_f_ABC) - min(p_f_ABC)

directional_specificity
  = [JS(p_f_ABC,p_u_ABC) + JS(p_f_ABC,p_s_ABC)] / [2 ln(2)]

directional_memory_strength
  = M
    × expectation_margin
    × sqrt(directional_specificity)
```

이 점수의 recall-train 75 percentile 이상인 firm-relation만 controlled activation geometry를 정의하는 데 사용했다.

### 8.5 Microsoft forced-choice 기억의 실제 계산

Qwen2.5, `MICROSOFT CORP × leadership_workforce`의 두 paraphrase 분포는 다음과 같았다.

```text
v0 [A, B, C, D]
   [0.000016, 0.985920, 0.0000003, 0.014063]

v1 [A, B, C, D]
   [0.000704, 0.008571, 0.0000002, 0.990725]

평균
   [0.000360, 0.497246, 0.0000002, 0.502394]
```

한 prompt에서는 B/continuity를 강하게 고르지만 다른 prompt에서는 D/no memory를 강하게 골랐다.

계산된 component:

```text
known_mass              = 0.497606
paraphrase JS           = 0.631467
paraphrase consistency  = 0.088986
option specificity JS   = 0.206816

memory strength M       = 0.024187
coverage gap            = 0.975813

ABC expectation margin  = 0.999276
directional specificity = 0.000458
directional strength    = 0.000621
```

즉 모델은 조건부로 B를 매우 선명하게 택하지만, 두 paraphrase 중 하나가 거의 D이므로 **안정적인 forced-choice memory는 약하다**고 판정했다.

이것이 단순 argmax만 사용하지 않은 이유다.

### 8.6 familiarity gradient sanity check

SEC company rank를 head/mid/tail로 고정했을 때 다음 gradient가 나타났다.

| 모델 | bin | ticker free-recall accuracy | 평균 memory strength | 평균 directional strength |
|---|---|---:|---:|---:|
| Qwen2.5 | head | 0.8313 | 0.0254 | 0.0092 |
| Qwen2.5 | mid | 0.5431 | 0.0135 | 0.0043 |
| Qwen2.5 | tail | 0.2958 | 0.0126 | 0.0031 |
| Qwen3 | head | 0.4940 | 0.0407 | 0.0033 |
| Qwen3 | mid | 0.2857 | 0.0303 | 0.0028 |
| Qwen3 | tail | 0.1499 | 0.0172 | 0.0019 |

큰 기업일수록 recall과 memory score가 높아지는 방향은 맞았다. 그러나 downstream `strong_text`에도 company rank와 popularity bin을 넣었으므로, memory model의 gain을 단순 대기업 효과로 설명할 수 없게 했다.

### 단계 출력

- 모델별 `memory_strength.parquet`
- firm-relation별 A/B/C/D prior
- memory strength, coverage gap, directional strength
- ticker recall과 subject-block diagnostic

### 단계 목적 recap

Stage E의 목적은 모델이 특정 option을 골랐다는 사실이 아니라, **두 표현에서 반복되고 Unknown 및 비슷한 다른 기업과 구분되는 기업별 기억**만 다음 단계의 기반으로 허용하는 것이다.

---

## 9. Stage F — behavioral memory violation

### 9.1 목적

새 기사가 실제로 어느 방향을 말하는지 post-news A/B/C/D logit으로 읽고, 그 방향이 firm-specific pre-news expectation에서 얼마나 의외인지 계산한다.

### 9.2 post-news 방향

예를 들어 Microsoft 감원 기사를 넣은 prompt는 다음 구조다.

```text
Target company: MICROSOFT CORP
A new leadership or workforce report says:

Headline: Microsoft Corporation (MSFT) to Cut 4,800 Jobs
Lead: ...

Which direction does this report establish for the target company?

A. favorable ...
B. continuity ...
C. adverse ...
D. unrelated or insufficient
Return only A, B, C, or D.
```

post-news `A:B:C` argmax를 뉴스 방향 `y_n`으로 사용한다. D는 relevance gate로 별도 처리한다.

### 9.3 generic-adjusted directional surprisal

D가 큰 기업이 자동으로 surprise가 되지 않도록 A/B/C conditional probability를 사용한다.

```text
firm surprisal
  = -log p_f_ABC(y_n)

unknown surprisal
  = -log p_u_ABC(y_n)

shuffled surprisal
  = -log p_s_ABC(y_n)

generic-adjusted surprisal
  = firm surprisal
    - mean(unknown surprisal, shuffled surprisal)
```

최종 behavioral violation:

```text
V_behavior
  = M
    × (1 - p_post(D))
    × max(0, generic-adjusted surprisal)
```

### 9.4 Microsoft 감원 실제 계산

Qwen2.5의 post-news 분포:

```text
post [A, B, C, D]
     [0.00000000003,
      0.00000000003,
      1.00000000000,
      0.00000000025]
```

즉 기사는 C/adverse로 거의 확정됐다.

```text
firm-specific adverse surprisal = 14.556310
control-adjusted excess         =  7.750025
memory strength M               =  0.024187
post relevance                  ≈  1.000000

behavioral violation
  = 0.024187 × 1 × 7.750025
  = 0.187452
```

중요한 해석:

- 기사는 분명 adverse다.
- Microsoft prior에서는 adverse가 매우 낮아 directional surprisal은 크다.
- 그러나 forced-choice 기억의 paraphrase 안정성이 낮아 `M=0.024`뿐이다.
- 따라서 raw surprisal 14.56을 그대로 쓰지 않고 최종 점수는 0.187로 크게 줄었다.

### 9.5 Del Monte에서 모델별 불일치 예시

기사:

```text
Del Monte Corporation Announces $150 Million Upsizing
of Revolving Credit Facility to $900 Million
```

Qwen3:

```text
pre prior:
  B continuity = 0.7121
  D no memory  = 0.2879

news:
  A favorable  = 0.9981

memory strength             = 0.4173
generic-adjusted surprisal  = 4.0988
behavioral violation        = 1.7104
```

Qwen2.5:

```text
pre B ≈ 0.5000, D ≈ 0.5000
memory strength             ≈ 0.000007
news B continuity           ≈ 0.9999
generic-adjusted surprisal  < 0
behavioral violation        = 0
```

같은 기사라도 모델마다 저장된 기억과 post-news 방향 해석이 달랐다. 이 불일치를 숨기지 않고 모델별 feature로 보존했다.

### 단계 출력

- relation별 `behavioral_memory_violation`
- `firm_direction_surprisal`
- `generic_adjusted_surprisal`
- `coverage_gap`
- 기사별 relation maximum

### 단계 목적 recap

Stage F의 목적은 adverse나 rare event 자체를 surprise로 부르지 않고, **기업 기억에서만 추가로 의외인 방향**을 memory strength와 news relevance로 gate해 측정하는 것이다.

---

## 10. Stage G — activation geometry를 찾되 axis를 강제하지 않기

### 10.1 목적

activation이 확인과 위반을 구분하는지 보되, residual norm이나 변동성 label로 방향을 만들지 않는다. label은 controlled firm memory와 matched A/B/C report에서만 정의한다.

### 10.2 controlled benchmark

896개 기업 각각에 8개 관계, A/B/C 세 statement를 만들었다.

예를 들어 `product_operations`의 A report는 모든 기업에서 동일하다.

```text
favorable: a launch, trial, project, or operation succeeded or expanded
```

firm의 pre-news conditional A/B/C prior에서:

```text
expected outcome = argmax p_f_ABC
contrary outcome = argmin p_f_ABC

controlled report가 expected outcome이면 confirm
controlled report가 contrary outcome이면 contradict
```

post-news direction classifier가 report의 A/B/C를 정확히 읽은 row만 유지했다.

### 10.3 왜 relation × realized outcome centering을 했는가

그냥 A report와 C report activation을 비교하면 “favorable 대 adverse” axis가 나올 수 있다. 우리가 원하는 것은 같은 outcome에서도 회사 기억에 따라 confirm/contradict가 달라지는가다.

따라서 train에서 다음 center를 계산했다.

```text
delta_i = h_post_i - h_pre_i

center[r,y]
  = mean(delta_i | relation=r, realized outcome=y, train)

centered_delta_i
  = delta_i - center[relation_i, outcome_i]
```

AUC도 같은 `relation × realized outcome` stratum 안에서 confirm과 contradict를 비교한 뒤 cross-class pair 수로 가중했다.

### 10.4 실제 matched activation 예시

Qwen2.5의 `product_operations × realized A` test stratum:

동일한 A/favorable report를 읽었지만 회사 prior가 다르다.

```text
McDonald's
  pre conditional expectation: A가 가장 높음
  realized report: A
  label: confirm
  selected activation score: -3.1235

Okeanis Eco Tankers
  pre conditional expectation: B가 가장 높음
  A는 가장 낮은 방향
  realized report: A
  label: contradict
  selected activation score: +5.0672
```

둘 다 “좋은 product/operation news”를 읽었다. activation score가 다른 것은 최소한 이 예시에서는 A와 C의 valence 차이만으로 설명되지 않는다.

### 10.5 비교한 geometry

각 layer에서 다음 candidate를 비교했다.

1. Difference-in-means direction
2. pre/post delta L2 norm
3. pre/post cosine distance
4. PCA diagonal-density likelihood ratio
5. confirm-update subspace 밖의 residual distance
6. PCA-space kNN confirm/contradict density ratio

cheap method인 DiM, L2, cosine를 모든 layer에 적용하고, validation AUC 상위 3개 layer에서 density/subspace method를 추가했다.

### 10.6 null arm 규칙

best cheap validation AUC가 0.55 미만이면:

```text
selected_method = null
activation score = 0
```

즉 약한 geometry에서 억지로 PCA나 kNN을 돌려 신호를 만들지 않도록 했다.

### 10.7 strong-memory gate

activation confirm/contradict benchmark에는 recall-train에서 동결한 directional-memory-strength 75 percentile 이상만 들어갔다.

| 모델 | threshold |
|---|---:|
| Qwen2.5 | 0.003716 |
| Qwen3 | 0.001374 |

### 10.8 선택 결과

| 모델 | 방법 | 저장 layer index | validation AUC | controlled test AUC | firm-bootstrap test CI | val/test eligible strata |
|---|---|---:|---:|---:|---|---|
| Qwen2.5 | DiM | 18 | 1.000 | 1.000 | [1.000, 1.000] | 1 / 1 |
| Qwen3 | DiM | 35 | 0.730 | 0.800 | [0.000, 1.000] | 4 / 2 |

겉보기에는 Qwen2.5가 완벽하지만, confirm과 contradict가 모두 있는 `relation × outcome` stratum이 validation과 test에서 각각 1개뿐이다. 사전 성공 규칙은 validation 최소 3개, test 최소 2개 stratum을 요구했다.

Qwen3는 stratum 수는 통과했지만 test CI lower가 0.5를 넘지 못했다.

따라서 broadly supported controlled representation:

```text
0 / 2 models
```

### 10.9 random-label control

선택 cheap layer에서 train label을 `relation × realized outcome` 안에서 200회 permutation해 DiM을 다시 fit했다.

- Qwen2.5 empirical diagnostic p: `0.0299`
- Qwen3 empirical diagnostic p: `0.3035`

이 permutation은 선택된 layer의 진단이지, 모든 layer/method search 전체에 대한 family-wise correction이라고 주장하지 않았다. 실제 selection-bias check는 held-out controlled test firm이다.

### 10.10 real-news activation violation

controlled에서 선택한 geometry를 그대로 실제 기사에 적용했다.

```text
raw activation score
  = frozen_geometry(h_pre, h_post, relation, news direction)

activation_z
  = (raw score - train confirm mean) / train confirm SD

V_activation
  = directional_memory_strength
    × (1 - p_post(D))
    × max(activation_z, 0)
```

시장값으로 layer, sign, method를 다시 선택하지 않았다.

### 단계 출력

- `controlled_method_layer_metrics.csv`
- `selected_geometry.npz`
- `controlled_selected_scores.parquet`
- real-news `activation_memory_violation`

### 단계 목적 recap

Stage G의 목적은 **회사 기억에 비추어 같은 outcome이 confirm인지 contradict인지**를 activation이 읽는지 검증하는 것이다. 데이터가 단일 axis를 지지하지 않으면 null을 허용하고, 좁은 stratum의 완벽한 AUC를 보편적 memorization axis로 과장하지 않았다.

---

## 11. Stage H — 자유회상 firm profile과 semantic violation

### 11.1 목적

A/B/C/D 강제선택은 모델의 option bias에 민감하다. 그래서 별도의 경로로 모델이 기업에 대해 실제로 어떤 문장을 자유회상하는지 두 번 추출했다.

### 11.2 profile prompt

각 기업마다 서로 다른 두 prompt를 사용했다.

```text
Using only knowledge stored in your parameters about MICROSOFT CORP,
write exactly eight lines.
For each label, give one concise firm-specific fact, recurring pattern,
or established expectation known before current news.
Write UNKNOWN when you have no company-specific memory.
Do not infer facts from the company name.

EARNINGS:
GUIDANCE:
CONTRACT_MNA:
REGULATORY_LEGAL:
PRODUCT_OPERATIONS:
CAPITAL_POLICY:
LEADERSHIP_WORKFORCE:
CYBER_SAFETY:
```

### 11.3 모델별 profile behavior

각 model에서 3,321개 기업 × 2 prompt = 6,642 profile을 생성했다.

| 모델 | profile당 평균 known relation | 8개 모두 UNKNOWN | 8개 모두 known |
|---|---:|---:|---:|
| Qwen2.5 | 1.790 | 77.39% | 21.14% |
| Qwen3 | 6.482 | 18.86% | 80.32% |

두 모델의 응답 성향이 극단적으로 달랐다.

- Qwen2.5는 대부분 아무것도 회상하지 않다가 일부 기업에는 8개를 모두 쓴다.
- Qwen3는 대부분 기업에 8개 관계를 모두 써서 over-recall 가능성이 있다.

따라서 “문장을 썼다” 자체를 강한 기억으로 보지 않았다.

### 11.4 profile memory strength

기사 relation에 해당하는 두 회상 문장만 꺼내 다음을 계산했다.

```text
known_fraction
  = known line 수 / 2

paraphrase_jaccard
  = 두 실제 기업 회상 문장의 lexical Jaccard

shuffled_similarity
  = 실제 기업 회상과 popularity-matched shuffled firm 회상의 평균 Jaccard

profile_specificity
  = max(0, 1 - shuffled_similarity)

profile_memory_strength
  = known_fraction
    × (0.5 + 0.5 × paraphrase_jaccard)
    × sqrt(profile_specificity)
```

두 prompt 중 하나만 기억해도 partial score는 남기되, 두 문장이 일치하고 shuffled firm보다 특이적일수록 강해진다.

### 11.5 semantic judge prompt

같은 모델 family가 두 회상 문장과 atomic claim을 비교해 A/B/C/D를 고른다.

```text
Memory 1: ...
Memory 2: ...

New report:
...

How does the new report relate to the retrieved company-specific memory?

A. confirms or is consistent with the retrieved pattern
B. contradicts, reverses, or strongly violates the retrieved pattern
C. adds genuinely new information without contradicting the retrieved pattern
D. the memory is missing, or the report is unrelated/insufficient

Return only A, B, C, or D.
```

같은 atomic claim에 다음 세 judge run을 만들었다.

1. actual company profile
2. `UNKNOWN, UNKNOWN`
3. popularity-matched shuffled firm's profile

### 11.6 semantic violation 공식

actual profile의 contradiction log-odds가 두 control보다 얼마나 높은지 계산한다.

```text
excess
  = logit[p_actual(B)]
    - mean(
        logit[p_UNKNOWN(B)],
        logit[p_shuffled(B)]
      )

V_semantic
  = profile_memory_strength
    × (1 - p_actual(D))
    × p_actual(B)
    × max(excess, 0)
```

`p_actual(B)`를 별도로 곱한 이유가 중요하다.

초기 label-free audit에서 actual judge가 C/new information을 거의 확신해도, control의 B 확률이 더 작다는 이유만으로 relative log-odds가 커지는 문제가 발견됐다. 시장 gate를 열기 전에 absolute contradiction probability gate를 추가하고 이전 score는 audit 폴더에 보존했다.

### 11.7 Microsoft 감원 실제 semantic 계산

Qwen2.5이 뉴스 전에 자유회상한 `leadership_workforce`:

```text
Memory 1:
The company is led by a stable executive team, including CEO Satya Nadella,
who has been in charge since 2014, fostering consistent leadership.

Memory 2:
Has a stable leadership team with a focus on succession planning,
though there have been some key executive changes in recent years.
```

Atomic claim:

```text
Microsoft Corporation (MSFT) to Cut 4,800 Jobs
```

계산값:

```text
profile known count            = 2 / 2
profile lexical consistency    = 0.193548
profile specificity            = 0.958333
profile memory strength        = 0.584209

p(confirm)                     ≈ 0.000000006
p(contradict)                  = 0.999924
p(new)                         = 0.000075
p(missing/unrelated)           = 0.000001
control-adjusted log-odds      = 20.299053

semantic memory violation      = 11.857979
```

Qwen3도 이 기사에 `12.917774`의 높은 semantic violation을 부여했다.

### 11.8 forced-choice와 semantic 경로가 왜 다를 수 있는가

같은 Microsoft 예시에서:

```text
Qwen2.5 forced-choice memory strength = 0.024187
Qwen2.5 profile memory strength       = 0.584209
```

forced-choice paraphrase 하나는 B, 다른 하나는 D여서 불안정했다. 반면 자유회상 문장 두 개는 “stable leadership”이라는 유사한 내용을 냈다.

따라서:

```text
behavioral violation = 0.187452
activation violation = 0
semantic violation   = 11.857979
coverage gap         = 0.975813
```

하나의 scalar로 모든 현상을 합치기 전에 arm별로 보존한 이유가 이 사례에서 보인다.

### 11.9 coverage gap 사례

USA Rare Earth의 CEO 교체 기사에서 Qwen2.5은 해당 relation의 두 profile을 모두 `UNKNOWN`으로 출력했다.

```text
profile memory strength = 0
semantic violation      = 0
forced memory strength  ≈ 0.0000002
coverage gap            ≈ 1.0
```

기사는 중요할 수 있지만 Qwen2.5의 “기억과 반대되는 기사”라고 부르지는 않는다.

Qwen3은 한 profile에서 “stable executive team”을 회상해 semantic violation `6.846864`를 부여했지만 forced-choice coverage gap은 거의 1이었다. 이처럼 model·elicitation별 불일치 자체가 중요한 limitation이다.

### 단계 출력

- `semantic_relation_features.parquet`
- `semantic_article_features.parquet`
- profile memory strength
- confirm/contradict/new/missing probability
- semantic memory violation

### 단계 목적 recap

Stage H의 목적은 option 선택에만 의존하지 않고 모델이 실제로 회상한 기업별 문장과 기사의 의미 관계를 직접 비교하는 것이다. 동시에 UNKNOWN·shuffled control과 absolute contradiction gate로 generic prose와 judge bias를 줄였다.

---

## 12. Stage I — 관계 점수를 기사 점수로 집계

### 12.1 목적

기사 하나가 둘 이상의 관계를 포함할 수 있다. 예를 들어 earnings와 product operation이 함께 있을 수 있다. 어떤 관계에서 기억 위반이 강한지를 버리지 않되, 관계 수가 많은 기사가 단순 합산으로 유리해지는 것도 막아야 한다.

### 12.2 집계 규칙

각 construct를 독립적으로 최대 집계했다.

```text
article behavioral violation
  = max_r behavioral_violation(article, r)

article activation violation
  = max_r activation_violation(article, r)

article semantic violation
  = max_r semantic_violation(article, r)

article coverage gap
  = max_r coverage_gap(article, r)
```

주의할 점:

```text
article coverage gap ≠ 1 - article memory violation
```

relation별로 다른 maximum을 가질 수 있으므로 각각 직접 집계했다.

### 12.3 Rio Tinto 예시

Rio Tinto production/sales 기사는 `earnings`와 `product_operations` 두 관계를 가졌다.

Qwen3:

```text
earnings activation violation          = 0.244798
product_operations activation violation = 0.004580

article activation violation
  = max(0.244798, 0.004580)
  = 0.244798
```

이 기사는 최종 qualitative table에서 high activation example로 올라왔다. 그러나 Qwen3 coverage gap도 약 `0.999`였고 semantic judge는 대부분 new information으로 봤다. 따라서 이 사례만으로 strong-memory contradiction이라고 해석하면 안 된다.

### 단계 출력

- 모델별 `article_features.parquet`
- 9,376개 eligible article과 one-to-one 정렬

### 단계 목적 recap

Stage I의 목적은 여러 관계 중 가장 강한 기억 위반을 기사 점수로 보존하면서, 관계 개수 자체가 점수를 키우지 못하게 하는 것이다.

---

## 13. Stage J — 강한 text·embedding baseline 만들기

### 13.1 목적

memory score가 단순 sentiment, 긴 기사, unusual wording, firm popularity, embedding novelty를 재발견했을 가능성을 통제한다.

### 13.2 label-blind text control

다음 feature를 만들었다.

- body length와 token 수
- positive/negative/uncertainty word rate
- net/absolute sentiment
- digit와 uppercase character rate
- publisher
- presswire 여부
- relation과 relation count
- 제거한 market-reaction/price-target sentence 수
- SEC company rank
- head/mid/tail popularity bin

### 13.3 TF-IDF novelty

TF-IDF vocabulary와 reference set은 analysis-eligible development-train 3,545개 기사만으로 fit했다.

```text
tfidf max train similarity
tfidf nearest-train novelty
tfidf train-centroid similarity
tfidf train-centroid novelty
```

vocabulary size는 35,624였다.

### 13.4 embedding control

instruction 없는 plain article text에 다음 embedding model을 사용했다.

- Qwen3-Embedding-0.6B
- Qwen3-Embedding-8B

각 embedding에서:

- development-train centroid distance
- nearest development-train distance
- development-train만으로 fit한 64-dimensional PCA

를 만들었다.

시장 task, return horizon, volatility instruction은 embedding 입력에 넣지 않았다.

### 13.5 post-news direction control

두 generator model의 post-news A/B/C/D 확률도 `strong_text`에 포함했다.

즉 memory model은 단순히 “adverse 확률이 높은 기사”보다 더 나아야 했다.

### 단계 출력

- `outputs/retrieval_conditioned_text_controls/`
- `outputs/retrieval_conditioned_embeddings/`
- `outputs/retrieval_conditioned_embedding_controls/`

### 단계 목적 recap

Stage J의 목적은 memory violation이 뉴스 문체, sentiment, 일반 embedding novelty, 회사 규모, 또는 post-news adverse classification의 다른 이름이 아닌지를 시험할 강한 baseline을 만드는 것이다.

---

## 14. Stage K — feature lock

### 14.1 목적

market outcome을 본 뒤 어떤 memory score, layer, embedding, article mask를 바꾸지 못하게 모든 representation을 먼저 봉인한다.

### 14.2 동결 내용

최종 feature table:

```text
rows                  = 9,376
columns               = 226
July rows             = 383
July feature finite   = true
market values read    = false
market label gate     = CLOSED
```

포함된 큰 feature family:

```text
price placeholder/identity alignment fields
text controls
TF-IDF novelty
Embedding-0.6B PCA/novelty
Embedding-8B PCA/novelty
Qwen2.5 post-news direction
Qwen3 post-news direction
identity/coverage diagnostics
behavioral violation
activation violation
semantic violation
```

각 원천과 output의 SHA-256을 `feature_lock.json`에 기록했다.

### 14.3 실제 gate chronology

```text
2026-07-23 06:14:43 UTC  feature lock frozen
2026-07-23 06:14:44 UTC  development price gate open
2026-07-23 06:15:24 UTC  downstream models frozen
2026-07-23 06:15:25 UTC  July confirmation gate open
2026-07-23 06:18:56 UTC  July evaluation complete
```

feature lock과 downstream model freeze 사이 순서가 1초 단위로 감사됐다.

### 단계 출력

- `outputs/retrieval_conditioned_feature_lock/features.parquet`
- `outputs/retrieval_conditioned_feature_lock/feature_lock.json`

### 단계 목적 recap

Stage K의 목적은 금융 label이 memory construct나 activation geometry에 역으로 스며들 수 없게 하는 것이다. 이 시점까지 symbol/date는 정렬에 사용했지만 OHLC와 outcome 값은 읽지 않았다.

---

## 15. Stage L — development price gate와 금융 모델 동결

### 15.1 목적

July 확인 구간을 보기 전에 과거 구간만으로 downstream model, Ridge penalty, 전처리, 중요 뉴스 threshold를 모두 고정한다.

### 15.2 market outcome

primary outcome은 mapped event session의 절대 SPY-adjusted close-to-close log return이다.

```text
firm_return_t
  = log(close_t / close_(t-1))

market_adjusted_return_t
  = firm_return_t - SPY_return_t

y_t
  = |market_adjusted_return_t|

model target
  = log(1 + 10,000 × y_t)
```

secondary outcome:

- 2-session absolute market-adjusted return
- same-session Parkinson range expansion
- abnormal volume

### 15.3 pre-event price control의 시점

`prevol20`과 직전 absolute market-adjusted return을 baseline에 포함했다.

장중 발행 기사는 publication-day close를 아직 관측하지 못했으므로 그 close를 pre-event control에 쓰지 않고 한 session 더 이전으로 이동했다.

### 15.4 raw OHLC split filter

Yahoo 가격은 adjusted가 아닌 raw OHLC다. 다음 조건을 만족하는 mechanical split-like session을 outcome 형성 전에 missing 처리했다.

- previous-close/current-open 비율이 1.5, 2, 3, 4, 5, 10 factor 근처
- overnight gap 절댓값 > 25%
- open-to-close move 절댓값 < 25%

development event 5개가 제외됐고 July에서는 0개였다. ordinary dividend 문제까지 해결했다고 주장하지 않았다.

### 15.5 feature set

| feature set | 구성 |
|---|---|
| `price_only` | prevol20, prior absolute return |
| `strong_text` | price + metadata + lexical + TF-IDF + 두 embedding + post-news A/B/C/D + firm rank/popularity |
| `text_plus_identity` | strong text + ticker recall/memory/coverage |
| `text_plus_behavior` | strong text + identity + behavioral |
| `text_plus_activation` | strong text + identity + activation |
| `text_plus_semantic` | strong text + identity + semantic |
| `all_memory` | strong text + identity + behavioral + activation + semantic |

categorical feature는 train에서 fit한 category만 사용했다.

### 15.6 train/test와 model selection

finite price control과 outcome을 가진:

- development train: 3,522
- development test: 4,479

각 feature set에서 Ridge alpha 후보를 비교했다.

```text
alpha ∈ {0.01, 0.1, 1, 10, 100, 1000}
```

April–June development test의 `R²_log`가 가장 높은 alpha를 고른 뒤, 전체 development data로 refit하고 model package를 hash했다.

선택 결과:

```text
price_only alpha = 100
나머지 feature set alpha = 1000
```

development-test 결과:

| feature set | R² log | raw R² | Spearman |
|---|---:|---:|---:|
| price only | 0.1242 | -0.1058 | 0.4004 |
| strong text | 0.1537 | 0.0184 | 0.4252 |
| text + identity | 0.1536 | 0.0188 | 0.4249 |
| text + behavior | 0.1536 | 0.0188 | 0.4252 |
| text + activation | 0.1535 | 0.0211 | 0.4246 |
| text + semantic | 0.1534 | 0.0186 | 0.4243 |
| all memory | 0.1528 | 0.0207 | 0.4238 |

개발 구간에서는 memory feature가 strong text보다 개선하지 않았다. 그래도 사전 설계된 July confirmation을 그대로 진행했다.

importance classification threshold는 전체 development target의 80 percentile인 `0.0673741`로 동결했다.

### 단계 출력

- `development_outcomes.parquet`
- `development_model_selection.csv`
- 7개 frozen Ridge package
- `downstream_selection.json`

### 단계 목적 recap

Stage L의 목적은 July 값 없이 금융 prediction pipeline 전체를 선택하고 고정하는 것이다. 개발 결과가 기대보다 약하더라도 July 규칙을 바꾸거나 유리한 family만 primary로 바꾸지 않았다.

---

## 16. Stage M — July confirmation을 한 번 열어 검증

### 16.1 목적

모든 feature와 model이 동결된 뒤에만 2026-07-09부터 2026-07-21까지의 가격을 읽어 true temporal confirmation을 수행한다.

### 16.2 confirmation data

```text
articles        = 383
tickers         = 343
event dates     = 9
first session   = 2026-07-09
last session    = 2026-07-21
```

July 22 가격은 July 21 event의 사전등록된 2-session secondary outcome을 정의하는 용도로만 허용했다.

### 16.3 primary 비교

```text
primary = R²_log(all_memory) - R²_log(strong_text)
```

`strong_text`가 이미 다음을 포함하므로 비교 기준이 약하지 않다.

- pre-event price history
- publisher/relation/presswire
- lexical sentiment/uncertainty
- TF-IDF novelty
- 두 plain embedding의 PCA와 novelty
- 두 LLM의 post-news A/B/C/D
- company rank와 popularity

### 16.4 July model 결과

| feature set | R² log | Spearman | importance AUC |
|---|---:|---:|---:|
| price only | -0.7302 | 0.1677 | 0.7376 |
| strong text | -0.1298 | 0.3313 | 0.8572 |
| text + identity | -0.1325 | 0.3305 | 0.8612 |
| text + behavior | -0.1319 | 0.3308 | 0.8569 |
| text + activation | -0.1244 | 0.3312 | 0.8615 |
| text + semantic | -0.1366 | 0.3281 | 0.8640 |
| all memory | -0.1285 | 0.3287 | 0.8603 |

절대 R²가 음수라는 것은 July의 짧은 regime에서 평균 예측보다 squared error가 컸다는 뜻이다. `ΔR²`가 양수여도 강한 절대 예측 모델이라는 뜻은 아니다.

### 16.5 incremental family ablation

| 추가 family | strong text 대비 ΔR² log | event-date 95% CI | P(Δ≤0) |
|---|---:|---|---:|
| identity | -0.00266 | [-0.00910, 0.00328] | 0.7855 |
| behavioral | -0.00208 | [-0.01077, 0.00575] | 0.6965 |
| activation | +0.00542 | [-0.00493, 0.01803] | 0.1730 |
| semantic | -0.00682 | [-0.01803, 0.00284] | 0.9035 |
| all memory | +0.00128 | [-0.01457, 0.01982] | 0.4660 |

activation family의 point estimate가 가장 컸지만 CI가 0을 포함한다. 이 결과를 “activation axis 성공”이라고 부를 수 없는 이유는 다음과 같다.

- controlled broad-support rule을 0/2 모델만 충족했다.
- univariate activation score는 두 모델 모두 음의 상관이었다.
- family Ridge gain은 여러 activation/identity feature의 조건부 조합이며 단일 axis 검정이 아니다.
- July event date가 9개뿐이라 uncertainty가 크다.

### 16.6 univariate memory score

| score | July Spearman | event-date 95% CI |
|---|---:|---|
| Qwen2.5 behavioral violation | +0.1289 | [+0.0053, +0.2419] |
| Qwen3 behavioral violation | -0.0691 | [-0.1843, +0.0581] |
| Qwen2.5 activation violation | -0.0615 | [-0.2016, +0.0688] |
| Qwen3 activation violation | -0.0090 | [-0.1197, +0.1034] |
| Qwen2.5 semantic violation | +0.0286 | [-0.0538, +0.1225] |
| Qwen3 semantic violation | +0.0588 | [+0.0227, +0.0864] |
| Qwen2.5 coverage gap | +0.0059 | [-0.0981, +0.1264] |
| Qwen3 coverage gap | +0.1393 | [+0.0370, +0.2477] |
| TF-IDF novelty | +0.0371 | [-0.0778, +0.1512] |
| Embedding-8B novelty | +0.0425 | [-0.0570, +0.1455] |

CI lower가 0보다 큰 memory-violation score는 2개다.

1. Qwen2.5 behavioral violation
2. Qwen3 semantic violation

Qwen3 coverage gap도 양수지만 이것은 violation이 아니라 unknown/coverage construct이므로 성공 규칙의 세 violation score에 포함하지 않았다.

### 16.7 bootstrap

동일 날짜의 market shock을 보존하기 위해 2,000회 event-date block bootstrap을 primary로 사용했다.

```text
all_memory - strong_text ΔR²
point estimate       = +0.001283
date-block 95% CI    = [-0.014568, +0.019820]
P(nonpositive)       = 0.4660
```

같은 issuer가 다른 날짜에 반복될 수 있어 2,000회 issuer-block sensitivity도 추가했다.

```text
issuer-block 95% CI  = [-0.013765, +0.019491]
P(nonpositive)       = 0.4985
```

### 16.8 secondary outcomes

primary target에 fit한 동일 frozen prediction을 refit 없이 다음 outcome에 rank-test했다.

| outcome | strong text ρ | all memory ρ |
|---|---:|---:|
| 2-session absolute return | 0.3259 | 0.3214 |
| Parkinson expansion | 0.1106 | 0.1072 |
| abnormal volume | 0.2922 | 0.2882 |

secondary outcome에서는 all-memory ranking이 strong-text보다 좋아지지 않았다.

### 단계 출력

- `july_confirmation_outcomes.parquet`
- `july_predictions.parquet`
- `july_model_comparison.csv`
- `july_incremental_ablation.csv`
- `july_univariate_rank_results.csv`
- `confirmation_manifest.json`

### 단계 목적 recap

Stage M의 목적은 이미 고정된 기억 위반 신호가 시간적으로 뒤의 시장 반응에서 incremental utility를 갖는지 검증하는 것이다. 결과는 양의 point estimate를 보였지만 불확실성이 커 full support에는 실패했다.

---

## 17. 한 기사 E2E walkthrough — Microsoft 감원

이 절은 하나의 기사가 pipeline을 통과하면서 어떤 값으로 변하는지를 순서대로 보여준다.

### Step 1 — 기사 선택

```text
Ticker: MSFT
Company: MICROSOFT CORP
Title: Microsoft Corporation (MSFT) to Cut 4,800 Jobs
Event session: 2026-07-14
Split: July confirmation
Observation ID: 9c38d3a26744f4ae82896bd5
```

목적 recap: Microsoft를 직접 말하는 완료 사건이며 event time이 확인되는 기사 하나를 고른다.

### Step 2 — leakage-safe text와 relation

```text
Relation: leadership_workforce
Atomic claim: Microsoft Corporation (MSFT) to Cut 4,800 Jobs
```

목적 recap: 투자 의견이나 사후 주가 문구가 아니라 감원이라는 하나의 issuer event를 고정한다.

### Step 3 — forced pre-news memory

Qwen2.5 두 paraphrase:

```text
v0: B continuity 0.9859, D no-memory 0.0141
v1: B continuity 0.0086, D no-memory 0.9907
```

```text
M = 0.024187
coverage gap = 0.975813
```

목적 recap: conditional direction은 B로 보이지만 paraphrase에 불안정하므로 강한 기억으로 인정하지 않는다.

### Step 4 — free-profile pre-news memory

```text
Memory 1: stable executive team, Satya Nadella since 2014
Memory 2: stable leadership team, succession planning

profile strength = 0.584209
```

목적 recap: 강제선택과 독립적으로 모델이 실제로 회상하는 회사 패턴을 확인한다.

### Step 5 — post-news direction

```text
p(C=adverse | article) ≈ 1.0
```

목적 recap: 기사 자체가 leadership/workforce 악화를 말하는지 읽는다.

### Step 6 — behavioral violation

```text
firm adverse surprisal       = 14.556310
generic-adjusted excess      = 7.750025
behavioral violation         = 0.187452
```

목적 recap: adverse가 일반적으로 희귀해서가 아니라 Microsoft pre-memory에서 더 희귀한 부분만 남긴다.

### Step 7 — activation violation

```text
activation z                 = -5.247173
max(z, 0)                    = 0
activation memory violation  = 0
```

목적 recap: frozen controlled activation geometry는 이 update를 contradiction-like로 읽지 않았다. semantic score가 높다는 이유로 activation도 높다고 바꾸지 않는다.

### Step 8 — semantic violation

```text
p(contradict)                = 0.999924
control-adjusted log odds    = 20.299053
semantic violation           = 11.857979
```

목적 recap: “stable leadership”이라는 실제 회상 문장과 감원 claim의 의미적 충돌을 측정한다.

### Step 9 — article memory card

```text
coverage gap       0.975813  forced-choice memory는 약함
behavioral         0.187452  약하게 양수
activation         0.000000  controlled geometry는 비위반
semantic          11.857979  자유회상 프로필과 강한 모순
```

이것이 최종적으로 원하는 scoring 형태에 가깝다.

```text
한 줄짜리 “surprise = 11.86”가 아니라

1. 모델이 무엇을 기억했는지
2. 기억이 얼마나 안정적인지
3. 기사가 어떤 방향인지
4. 어떤 측정 경로가 위반이라고 보는지
5. 서로 불일치하는 경로는 무엇인지

를 함께 보여주는 memory card
```

### Step 10 — market 확인

이 기사의 one-session absolute market-adjusted return은 `0.019165`였다. 이 값은 위 모든 feature와 downstream model이 동결된 후에만 열렸다.

목적 recap: 시장 반응은 기억 위반을 정의하는 label이 아니라, 고정된 memory card가 실제 news importance와 관련되는지 보는 외부 검증값이다.

---

## 18. 세 가지 대표 사례를 비교하면 보이는 것

| 사례 | 핵심 상태 | 올바른 해석 |
|---|---|---|
| Microsoft 감원 | semantic 매우 높음, forced memory 약함, activation 0 | 자유회상 패턴과는 강한 충돌이나 모든 arm이 동의하지 않음 |
| Del Monte credit expansion | Qwen3 behavioral 높음, Qwen2.5 semantic 높음, 모델 간 불일치 | model-specific retrieval fingerprint |
| USA Rare Earth CEO 교체 | Qwen2.5 profile UNKNOWN, coverage gap ≈ 1 | Qwen2.5가 모르는 사건이지 기억 위반이 아님 |

이 비교의 핵심은 다음이다.

```text
high coverage gap
≠ high memory violation
≠ high activation contradiction
≠ high semantic contradiction
```

각 construct가 다르므로 연구 결과도 한 축으로 강제로 합치지 않았다.

---

## 19. 모델 간 agreement와 무엇을 의미하는가

9,376개 기사 전체에서 Qwen2.5와 Qwen3의 Spearman:

| construct | cross-model ρ |
|---|---:|
| memory strength | -0.1801 |
| directional memory strength | -0.0359 |
| directional specificity | +0.5518 |
| behavioral violation | +0.0797 |
| activation violation | +0.0156 |
| semantic violation | +0.0070 |

두 모델은 “기업별 기억이 얼마나 강한가”와 “어느 기사가 위반인가”에 거의 동의하지 않았다.

가능한 해석:

- model pretraining corpus와 instruction behavior가 다르다.
- Qwen2.5는 profile에서 매우 보수적이고 Qwen3는 과도하게 많은 내용을 회상한다.
- A/B/C/D option prior도 다르다.
- 기억 위반은 universal semantic property라기보다 model-specific retrieval fingerprint일 수 있다.

따라서 한 모델에서 유효한 축을 다른 모델 activation basis에 직접 이식하지 않았다.

### 단계 목적 recap

cross-model 분석의 목적은 좋은 수치만 복제라고 부르지 않고, memory construct 자체의 model dependence를 정량적으로 드러내는 것이다.

---

## 20. 최종적으로 주장할 수 있는 것과 없는 것

### 20.1 주장 가능한 것

1. 기업 뉴스보다 먼저 firm-relation parametric memory를 회수하는 E2E protocol을 구현했다.
2. coverage gap과 memory violation을 분리했다.
3. Unknown, popularity-matched shuffled firm, paraphrase, subject-attention block control을 적용했다.
4. 행동, activation geometry, 자유회상 semantic violation을 독립적으로 계산했다.
5. activation을 return, residual, volatility label로 학습하지 않았다.
6. 모든 feature를 동결한 후에만 시장값을 열었다.
7. July에서 all-memory의 incremental point estimate는 작지만 양수였다.
8. Qwen2.5 behavioral과 Qwen3 semantic violation은 July absolute return과 양의 univariate rank association을 보였다.

### 20.2 주장 불가능한 것

1. universal surprise axis
2. cross-model causal memorization axis
3. 안정적으로 유의한 incremental return prediction
4. 두 Qwen model의 model-invariant memory signal
5. memory violation이 시장가격을 인과적으로 움직였다는 주장
6. coverage gap과 memory violation이 같은 construct라는 주장
7. Qwen2.5 controlled AUC 1.0만으로 broad representation이 발견됐다는 주장

### 20.3 가장 방어 가능한 한 문장

> 두 open-weight LLM에서 기업별 사전 기억을 뉴스 이전에 회수하고, 그 기억의 부재와 새 기사의 기억 위반을 분리하는 label-blind 금융 뉴스 scoring pipeline을 구축했다. July confirmation에서 강한 text baseline 대비 작은 양의 incremental point estimate와 두 개의 양의 univariate memory-violation 신호를 얻었지만, event-date bootstrap과 cross-model consistency는 완전한 외부 타당성을 지지하지 않았다.

---

## 21. 이 연구에서 가장 중요한 설계 교훈

### 21.1 “기억”은 ticker 이름만 넣은 activation이 아니다

현재 연구에서 ticker는 market identity anchor다. 실제 기억 query는 canonical company name과 관계별 질문을 사용한다.

### 21.2 “반대되는 뉴스”에는 먼저 사전 기억이 필요하다

사전 기억 없이 기사만 보고 activation delta가 크다고 해서 memory violation이 아니다.

### 21.3 D/no memory를 contradiction으로 바꾸면 안 된다

모르는 기업의 rare event는 coverage gap이다. 이것이 core hypothesis와 가장 가까운 설계 분기다.

### 21.4 한 axis에 집착할 필요가 없다

controlled geometry가 충분하지 않으면 null arm을 허용했다. 실제 결과도 행동·semantic 일부는 양수였지만 activation univariate는 null이었다.

### 21.5 좋은 qualitative example은 disagreement도 보여줘야 한다

Microsoft 사례처럼 semantic은 높지만 forced memory와 activation은 약한 경우가 연구의 약점이면서 동시에 가장 정직한 설명이다.

### 21.6 금융시장은 discovery label이 아니라 external validation이다

이 원칙을 지켜야 연구가 “volatility axis를 찾고 surprise라고 이름 붙이는 것”으로 되돌아가지 않는다.

---

## 22. 다음 논문화 단계에서 가장 유망한 확장

현재 결과를 가장 자연스럽게 확장하는 방법은 score 하나를 더 복잡하게 만드는 것이 아니라 **memory-conditioned retrieval/router**로 평가하는 것이다.

### 22.1 추천 primary task

```text
입력:
  많은 기업 뉴스 stream

출력:
  제한된 review budget에서 사람이 먼저 읽어야 할 기사 ranking

비교:
  text/embedding novelty
  sentiment/uncertainty
  post-news direction
  coverage gap
  behavioral violation
  semantic violation
  memory-card ensemble

평가:
  top-k capture rate
  NDCG
  precision@budget
  large absolute market reaction/event label recall
```

이 framing은 짧은 July sample에서 Ridge `R²` 하나에 모든 주장을 걸지 않고, 사용자가 처음 말한 “더 필터링하거나 scoring할 수 있는 기대”와 직접 맞는다.

### 22.2 다음 실험에서 고정해야 할 것

- 더 긴 confirmation horizon
- 사람 또는 독립 model의 atomic-claim validation sample
- relation-balanced controlled benchmark
- Qwen3 profile over-recall calibration
- model-specific memory card와 cross-model consensus를 분리
- high-confidence retrieval만 쓰는 abstention threshold
- `coverage gap high`와 `violation high`에 서로 다른 routing action

예:

```text
violation high:
  기존 모델 기억에 반하는 중요 후보
  → analyst review 우선

coverage gap high:
  모델이 잘 모르는 기업/관계
  → 외부 retrieval 보강 우선

both low:
  기억과 일치하거나 평범한 업데이트
  → 낮은 우선순위
```

### 22.3 arXiv와 Neuron Agreement를 다음에 어떻게 연결할지

- arXiv:2510.09033의 subject-dependence는 recall/reliance diagnostic으로 유지한다.
- AH처럼 “틀렸지만 association-dependent”인 profile을 별도 표시한다.
- [Neuron Agreement 논문](https://openreview.net/forum?id=ZVRgcQq4D2)의 ensemble 관점은 단일 axis 대신 여러 trajectory/feature의 agreement를 보는 보조 방법으로 사용할 수 있다.
- 그러나 현재 workspace의 same-prompt N=8 NAD는 Qwen2.5 AUC 0.557, Qwen3 0.522로 약하거나 null이었고 lexical consensus를 이기지 못했다. 따라서 positive mechanism처럼 재사용하면 안 된다.

### 단계 목적 recap

다음 단계의 목적은 더 그럴듯한 axis를 만드는 것이 아니라, **기억 위반과 coverage gap을 서로 다른 routing signal로 사용했을 때 제한된 금융 뉴스 검토 예산에서 실제 효율이 개선되는지**를 직접 시험하는 것이다.

---

## 23. 재현 순서와 script map

> 아래는 실제 pipeline의 논리적 실행 순서다. 현재 artifact는 이미 완료됐으므로 재실행 전 각 manifest와 tmux session을 먼저 확인해야 한다.

### 23.1 cohort와 prompt

```bash
python scripts/build_retrieval_conditioned_news.py
python scripts/build_retrieval_memory_prompts.py
python scripts/build_retrieval_atomic_claims.py
python scripts/build_retrieval_text_controls.py
```

### 23.2 GPU main extraction

GPU job 전:

```bash
bash scripts/bootstrap_tmp_runtime.sh
```

named tmux lane:

```bash
tmux new-session -d -s retrieval_qwen25 \
  'bash scripts/run_retrieval_conditioned_memory_lane.sh qwen25_7b 0'

tmux new-session -d -s retrieval_qwen3 \
  'bash scripts/run_retrieval_conditioned_memory_lane.sh qwen3_4b 1'
```

### 23.3 semantic·embedding·controlled selection follow-up

```bash
bash scripts/run_retrieval_conditioned_followup.sh \
  qwen25_7b 0 qwen3_embedding_06b

bash scripts/run_retrieval_conditioned_followup.sh \
  qwen3_4b 1 qwen3_embedding_8b
```

내부에서 수행되는 핵심 순서:

```text
build profile judge prompts
→ extract semantic judge
→ extract text embedding
→ controlled activation select
→ real-news activation/behavior score
→ semantic violation score
→ embedding PCA/novelty controls
```

### 23.4 feature lock과 market gates

```bash
python scripts/freeze_retrieval_conditioned_features.py
python scripts/retrieval_conditioned_downstream.py develop
python scripts/retrieval_conditioned_downstream.py confirm
```

### 23.5 finalization과 terminal audit

```bash
python scripts/finalize_retrieval_conditioned_study.py
python scripts/audit_retrieval_conditioned_study.py
```

성공의 terminal proof:

```text
outputs/retrieval_conditioned_completion_audit/completion_audit.json

status = RETRIEVAL_CONDITIONED_COMPLETION_AUDIT_PASSED
checks = 39 / 39
finite extraction arrays = 34
```

---

## 24. 핵심 artifact 지도

| 내용 | 경로 |
|---|---|
| 동결 연구 protocol | `RETRIEVAL_CONDITIONED_NEWS_PROTOCOL.md` |
| 기사 super-cohort | `data/retrieval_conditioned_news/documents.parquet` |
| atomic claims와 eligibility | `data/retrieval_conditioned_atomic_claims/atomic_claims.parquet` |
| memory prompts | `data/retrieval_conditioned_memory/` |
| Qwen2.5/Qwen3 extraction | `outputs/retrieval_conditioned_memory/` |
| 행동·activation 분석 | `outputs/retrieval_conditioned_analysis/` |
| semantic judge | `data/retrieval_profile_judge/`, `outputs/retrieval_profile_judge/` |
| text controls | `outputs/retrieval_conditioned_text_controls/` |
| embedding controls | `outputs/retrieval_conditioned_embedding_controls/` |
| feature lock | `outputs/retrieval_conditioned_feature_lock/feature_lock.json` |
| downstream 결과 | `outputs/retrieval_conditioned_downstream/` |
| 최종 보고서 | `outputs/retrieval_conditioned_final/RETRIEVAL_CONDITIONED_NEWS_REPORT.md` |
| machine-readable 핵심 결과 | `outputs/retrieval_conditioned_final/key_results.json` |
| qualitative examples | `outputs/retrieval_conditioned_final/qualitative_examples.csv` |
| 최종 artifact manifest | `outputs/retrieval_conditioned_final/artifact_manifest.json` |
| terminal audit | `outputs/retrieval_conditioned_completion_audit/completion_audit.json` |
| 이전 recall 결과 카드 | `outputs/financial_recall_final_summary/FINAL_RESULT_CARD.md` |
| 이전 causal audit | `outputs/financial_recall_revision_intervention_audit/CAUSAL_AUDIT.md` |

---

## 25. 최종 상태 recap

### 데이터와 실행

```text
representation super-cohort    13,735 articles
analysis cohort                 9,376 articles
eligible relation rows          9,714
firms                           3,321
expectation prompts            23,380
controlled reports             21,504
news-relation prompts          14,107
free profiles                   6,642 per model
July confirmation                 383 articles
July event dates                    9
terminal checks                  39 / 39 passed
```

### construct 결과

```text
초기 ticker free-recall direction:
  두 모델에서 안정적 decoding
  causal intervention cross-model 실패

controlled news-violation geometry:
  Qwen2.5 narrow perfect stratum
  Qwen3 uncertain
  broad support 0 / 2

July univariate:
  Qwen2.5 behavioral positive CI
  Qwen3 semantic positive CI
  activation 두 모델 모두 null

July incremental:
  all-memory ΔR² point estimate +0.001283
  CI crosses zero
  full support 실패
```

### 한 줄 최종 recap

이 연구는 **LLM의 기업 기억을 먼저 회수하고 그 기억의 부재와 기억 위반을 분리한 뒤, 그 신호를 시장 label과 독립적으로 동결해 실제 금융 뉴스 필터로 검증하는 전체 pipeline**을 완성했다. 결과는 방법론과 일부 memory score에는 긍정적이지만, 보편적 activation axis나 확정적인 시장 예측 효용을 주장할 정도로 강하지는 않다.
