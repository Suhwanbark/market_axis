# Market Axis 실험 E2E 정리

## 0. 기존 피드백과 이번 POC의 수정 방향

### 0.1 LLM activation axis 자체의 유효성과 차별성을 먼저 보여줄 필요

비교 방법은 다음과 같다.

- LLM에게 직접 중요도를 물어보는 방법
- 상용 embedding 모델을 사용하는 방법(0.5B, 7B급)
- LLM 중간 activation을 사용하는 방법

### 0.2 기존 activation 모델의 knowledge-cutoff 문제 수정

기존 activation 모델은 학습 데이터 기간과 평가 기간이 겹칠 수 있어, test 기간보다 cutoff가 앞선 모델로 교체했다.

이번 POC에서는 이를 수정하기 위해 다음과 같이 설계했다.

- Activation 모델을 pretraining cutoff가 2022년 9월인 `Llama-2-7B Base`로 교체
- Train: 2019–2021
- Validation: 2022
- Test: 2023
- Layer와 Ridge alpha는 2022 validation에서만 선택하고 2023 test에는 고정 적용

## 1. 연구 질문

> LLM 내부 activation은 금융 문서가 시장에 미칠 영향을 측정하는 데 기존 방법보다 더 유용하고 차별적인 정보를 제공하는가?

이를 두 단계에서 검증한다.

1. **Impact identification:** 문서 score가 뉴스 이후의 순수한 기업 변동성 충격(기업의 평상시 변동성과 시장 공통 변동성을 제외한 부분)과 얼마나 연결되는지 correlation으로 평가한다.
2. **Downstream task:** 추출한 score가 실제 응용에서도 유용한지 평가한다. 현재 주축은 기존 시계열 변동성 모형에 score를 추가했을 때 QLIKE/R²/MSE가 개선되는지를 확인하는 forecasting이며, 정성적인 사례 분석과 해석도 포함할 수 있다.

---

## 2. 데이터

뉴스 데이터는 `EXAONE-BI/FinTexTS targetCompany_category1-3`이며, 기업별 OHLC와 결합했다.

| Split | 기간 | 뉴스 수 | 거래일 | Ticker |
|---|---|---:|---:|---:|
| Train | 2019-01-31~2021-12-31 | 25,767 | 737 | 100 |
| Validation | 2022-01-03~2022-12-30 | 12,051 | 251 | 100 |
| Test | 2023-01-03~2023-12-18 | 12,478 | 242 | 100 |

## 3. 주 target: document impact

$$
y_{i,t}=\frac12\left[
\log\frac{firm\_post5}{firm\_pre20}
-
\log\frac{market\_post5}{market\_pre20}
\right]
$$

예를 들어 어떤 기업의 변동성이 평소 대비 50% 증가했지만 같은 기간 시장 전체 변동성도 10% 증가했다면, impact label은 시장 공통 상승분을 제외한 기업 고유의 추가 변동성 충격을 나타낸다. 값이 클수록 해당 뉴스가 기업에 더 큰 고유 변동성 충격을 준 것으로 해석한다.

## 4. 비교 모델과 문서 표현

| 조건 | 모델 | 크기·차원 | 레이어 | 레이어별 pooling |
|---|---|---|---:|---|
| BGE | BAAI/bge-m3 | 0.57B, 1,024d | 24 | CLS |
| Qwen | Qwen3-Embedding-8B | 8B, 4,096d | 36 | last token |
| Activation | Llama-2-7B Base | 7B, 4,096d | 32 | body-token mean |
| Direct LM | 동일 Llama-2-7B Base | 7B | zero-shot | 1–9 next-token 기대값 |

## 5. Impact axis 학습

레이어 $l$에서 문서별 activation 또는 embedding을 쌓으면 다음과 같다.

$$
X^{(l)}=
\begin{bmatrix}
(x_{1}^{(l)})^\top\\
(x_{2}^{(l)})^\top\\
\vdots\\
(x_{N}^{(l)})^\top
\end{bmatrix}
\in\mathbb{R}^{N\times p},
\qquad
y=
\begin{bmatrix}
y_1\\y_2\\\vdots\\y_N
\end{bmatrix}
\in\mathbb{R}^{N}
$$

Ridge alpha grid:

$$
\{0.01,0.1,1,10,100,1000\}
$$

선택 절차:

1. 각 레이어와 alpha 조합을 train에서 fitting
2. Validation에서 activation/embedding layer와 Ridge alpha 선택
3. 선택 결과 freeze
4. Test 기간의 문서에도 같은 과정 적용: 문서별 embedding/activation 추출 → impact score 계산

---

## 6. Impact correlation 결과

**핵심 RQ:** LLM에게 직접 묻는 방법과 상용 embedding(0.5B, 7B)에 비해, LLM 중간 activation이 뉴스가 유발한 순수한 변동성 확대(기업의 평상시 변동성과 시장 전체 변동성 제외)를 더 잘 식별하는가?

| 방법 | Test Spearman ρ | Test Pearson |
|---|---:|---:|
| Direct LM | -0.0015 | -0.0006 |
| Embedding (0.5B) | 0.0829 | 0.1045 |
| Embedding (7B) | 0.0893 | 0.1189 |
| LLM activation (L16) | **0.1511** | **0.1989** |

Embedding은 통상적인 사용 방식인 최종 output 기준이다.

Embedding 모델에서도 모든 layer를 탐색하고 validation에서 가장 좋은 layer를 선택했을 때:

| 방법 | 최종 output Test Spearman | 선택 layer | 선택 후 Test Spearman | 선택 후 Test Pearson |
|---|---:|---:|---:|---:|
| Embedding (0.5B) | 0.0829 | L19/24 | 0.0896 | 0.1176 |
| Embedding (7B) | 0.0893 | L28/36 | 0.1032 | 0.1247 |
| LLM activation | — | L16/32 | **0.1511** | **0.1989** |

작은 MLP head로 impact score를 학습했을 때:

| 방법 | Test Spearman | Test Pearson |
|---|---:|---:|
| Embedding (0.5B) + MLP | 0.0877 | 0.1170 |
| Embedding (7B) + MLP | 0.0910 | 0.1242 |
| LLM activation (L16) + MLP | **0.1462** | **0.1895** |

현재 MLP는 layer마다 학습하지 않았다. Embedding은 최종 output을 사용하고, LLM activation은 Ridge validation에서 선택된 L16을 사용했다. MLP에서는 각 representation별로 validation Spearman이 가장 높은 epoch를 선택하고 5개 seed의 예측을 평균했다. 따라서 MLP가 선택한 것은 layer가 아니라 epoch이며, 전 레이어 MLP 탐색은 아직 수행하지 않았다.

---

## 7. Direct LM 실험

Activation과 동일한 Llama Base checkpoint에 문서가 기업의 평상시 수준 대비 유발할 변동성 크기를 1–9로 평가하도록 했다.

Score 정의:

1. 출력 후보를 숫자 token `1,...,9`로 제한
2. 9개 token logit에 softmax 적용
3. 숫자값의 확률 기대값 계산

$$
score_{direct}=\sum_{k=1}^{9} k\,P(k\mid document,prompt)
$$

Label은 전혀 사용하지 않는 zero-shot baseline이다.

추출 설정:

- 동일 Llama-2-7B Base checkpoint
- 최대 document 길이 512
- 두 개의 단일-GPU vLLM replica
- GPU memory utilization 0.85
- Train/Validation/Test 전 split 추출

News test score 분포:

- 평균: 5.219
- 표준편차: 0.347
- 범위: 3.326–7.215

출력이 collapse한 것은 아니지만 impact test Spearman은 `-0.0015`로 사실상 0이다. Base 모델은 explicit instruction following이 약하기 때문에 hidden activation에는 정보가 있어도 직접 1–9 rating에는 실패한 것으로 해석한다.

결과 파일:

- `outputs/direct_llama2_rating/all_scores.parquet`
- `outputs/direct_llama2_rating/manifest.json`

---

## 8. Ticker-centering 민감도

초기 실험에서는 train ticker별 representation centroid와 label 평균을 제거했다.

| 조건 | BGE test ρ | Llama test ρ | Llama 선택층 |
|---|---:|---:|---:|
| Ticker-centered 초기 실험 | 0.0904 | 0.1628 | 17 |
| 최종 uncentered 실험 | 0.0829 | 0.1511 | 16 |

Centering하면 correlation이 조금 올라가지만 최종 사용자 요청에 따라 uncentered 결과를 본 결과로 사용한다.

---

## 9. MLP impact head 실험

단순 Ridge 대신 각 frozen representation 위에 작은 MLP head를 학습했다.

Architecture:

```text
Linear(input, 64)
→ GELU
→ Dropout(0.1)
→ Linear(64, 1)
```

학습 설정:

- Optimizer: AdamW
- Learning rate: 0.001
- Weight decay: 0.001
- Batch size: 512
- Max epochs: 200
- Early stopping: validation Spearman, patience 15
- Seeds: 20260721, 20260722, 20260723, 20260724, 20260725
- 최종 score: 5개 seed prediction의 산술평균
- Target: train global mean/std로 표준화
- 학습 중 test representation/label 미사용

Impact correlation:

| 표현 | Ridge test ρ | MLP test ρ | MLP 변화 |
|---|---:|---:|---:|
| BGE final | 0.0829 | 0.0877 | +0.0048 |
| Qwen final | 0.0893 | 0.0910 | +0.0017 |
| Llama L16 | **0.1511** | 0.1462 | -0.0049 |

이전 OLS forecasting 8개 설정의 평균 QLIKE 개선:

| Score | 평균 QLIKE 개선 |
|---|---:|
| Llama Ridge | **4.40%** |
| Llama MLP | 3.93% |
| Qwen MLP | 2.70% |
| Qwen Ridge | 2.36% |
| BGE MLP | 2.19% |
| BGE Ridge | 1.48% |

해석:

- MLP는 BGE/Qwen embedding에는 소폭 도움
- Llama activation에서는 단순 Ridge가 MLP보다 강함
- 비선형 head가 activation의 우위를 설명하지 않음

주의: 이 MLP 실험의 BGE/Qwen 입력은 당시의 최종층 embedding이다. 이후 선택한 BGE L19/Qwen L28에 MLP를 다시 붙이는 실험은 아직 하지 않았다.

결과 파일:

- `outputs/news_uncentered_mlp_control/test_metrics.csv`
- `outputs/news_uncentered_mlp_control/selection.json`
- `outputs/news_uncentered_mlp_forecast/forecast_results.csv`

---

## 10. Target 민감도

### 10.1 시장 조정 제거

기존 score를 고정하고 market term만 제거한 `firm_log_expansion`과 비교했다.

| Score | 기존 market-adjusted ρ | Market 제거 ρ |
|---|---:|---:|
| BGE | 0.0829 | 0.0859 |
| Qwen | 0.0893 | 0.0902 |
| Llama | 0.1511 | 0.1507 |

거의 변하지 않는다. Activation 우위가 market adjustment 항 때문에 생긴 것은 아니다.

### 10.2 기업 평상시 변동성 정규화 제거: 기존 axis 고정

| Target | BGE ρ | Qwen ρ | Llama ρ |
|---|---:|---:|---:|
| `log(firm post5)` | 0.0616 | 0.0675 | **0.0859** |
| Market-adjusted post level | 0.0549 | 0.0566 | **0.0717** |

기존 impact axis는 평상시 변동성 정규화를 제거한 absolute post level과는 correlation이 낮다.

### 10.3 기업 평상시 변동성 정규화 제거: axis 재학습

Absolute post-level target으로 axis를 train/validation에서 다시 학습했다.

| 재학습 target | BGE test ρ | Qwen test ρ | Llama test ρ |
|---|---:|---:|---:|
| `log(firm post5)` | 0.4979 | **0.5766** | 0.5731 |
| `0.5[log firm post5 − log market post5]` | 0.5708 | 0.6283 | **0.6308** |

수치는 0.6까지 올라가지만, 이 target은 기업의 정상 변동성 수준과 ticker identity를 학습하기 쉽다. 따라서 document impact보다 “원래 변동성이 큰 기업 식별”에 가까우며 본 결과와 분리한다.

결과 파일:

- `outputs/news_no_firm_vol_normalization_frozen_score.csv`
- `outputs/no_firm_vol_normalization_refit/news_test_metrics.csv`

---

## 11. 최종 forecasting 설계

### 11.1 Forecast target

- H1: 뉴스일 다음 실제 거래일의 Parkinson variance
- H5: 뉴스일 다음 5개 실제 거래일 Parkinson variance 평균
- Forecast model은 log variance를 학습하고 예측값을 raw variance로 변환
- 모든 score variant는 동일한 test row에서 비교

### 11.2 Forecast baseline

#### AR(5)

- 직전 5개 일별 log Parkinson variance
- Forecast regression: OLS

#### HAR

- Daily log volatility
- Weekly 5일 평균 log volatility
- Monthly 22일 평균 log volatility
- Forecast regression: OLS, Lasso, Ridge

#### HAR-X

HAR feature에 다음 변수를 추가한다.

- 절대수익률
- 음의 수익률
- 20일 momentum
- 동일가중 시장 HAR daily/weekly/monthly
- Forecast regression: OLS, Lasso, Ridge

#### MIDAS

- Lag 후보: 30, 50, 80
- Beta weight theta 후보: 1, 1.5, 2, 3, 5, 10
- Validation QLIKE로 lag/theta 선택
- H1 선택: `k=50, theta=10`
- H5 선택: `k=80, theta=10`
- Forecast regression: OLS

#### LSTM

- 입력: 최근 7일 log Parkinson sequence
- 3-layer LSTM
- Hidden size: 16
- Dropout: 0.1
- Ticker embedding: 4d
- Document score: 1d
- Head: `Linear(21,16) → ReLU → Linear(16,1)`
- Batch size: 2,048
- Adam, lr 0.001, weight decay 1e-5
- Max epochs: 50
- Validation QLIKE early stopping, patience 7
- Seeds: 11, 22, 33, 44, 55
- Validation-selected epoch만큼 train+validation에서 refit
- 5개 seed prediction 평균

### 11.3 Classical forecast fitting

- Numeric feature: median imputation + standardization
- Ticker: one-hot fixed effect
- Target: log future variance
- OLS: train+validation fitting
- Lasso/Ridge:
  1. train fitting
  2. validation QLIKE로 alpha 선택
  3. train+validation refit

Lasso grid:

```text
1e-5, 1e-4, 1e-3, 1e-2, 1e-1
```

Ridge grid:

```text
0.01, 0.1, 1, 10, 100
```

### 11.4 평가 지표

- QLIKE: 낮을수록 좋음
- Raw R²: 높을수록 좋음
- Raw MSE: 낮을수록 좋음

QLIKE 정의:

$$
QLIKE(y,\hat y)=\frac{y}{\hat y}-\log\left(\frac{y}{\hat y}\right)-1
$$

---

## 12. 최종 forecasting 결과

아래 값은 각 forecast family의 baseline 대비 QLIKE 감소율이다. 높을수록 좋다.

| Model | H | BGE | Qwen | Llama | Direct LM |
|---|---:|---:|---:|---:|---:|
| AR/OLS | 1 | 2.72% | 2.93% | **3.97%** | 0.18% |
| HAR/OLS | 1 | 2.89% | 3.18% | **4.27%** | 0.16% |
| HAR/Lasso | 1 | 3.67% | 3.99% | **5.19%** | 0.14% |
| HAR/Ridge | 1 | 3.12% | 3.42% | **4.54%** | 0.16% |
| HAR-X/OLS | 1 | 2.86% | 3.14% | **4.20%** | 0.27% |
| HAR-X/Lasso | 1 | 1.50% | 0.95% | **2.09%** | 0.27% |
| HAR-X/Ridge | 1 | 3.13% | 3.41% | **4.53%** | 0.25% |
| MIDAS/OLS | 1 | 3.03% | 3.25% | **4.36%** | 0.22% |
| LSTM | 1 | 3.82% | 3.38% | **5.45%** | 0.44% |
| AR/OLS | 5 | 1.34% | 0.25% | **4.16%** | ~0% |
| HAR/OLS | 5 | 1.66% | 0.65% | **4.84%** | ~0% |
| HAR/Lasso | 5 | 2.35% | ~0% | **5.16%** | ~0% |
| HAR/Ridge | 5 | 1.85% | 0.81% | **4.94%** | ~0% |
| HAR-X/OLS | 5 | 1.57% | 0.49% | **4.66%** | 0.02% |
| HAR-X/Lasso | 5 | 2.27% | 3.11% | **5.09%** | 0.01% |
| HAR-X/Ridge | 5 | 1.74% | 0.41% | **4.76%** | 0.02% |
| MIDAS/OLS | 5 | 1.62% | 0.41% | **4.73%** | ~0% |
| LSTM | 5 | 2.35% | 0.52% | **4.79%** | 0.11% |

Llama activation은 총 18개 설정 모두 QLIKE point estimate 기준 1위다.

전체 설정 평균:

| Score | 평균 QLIKE 개선 | 평균 Δ raw R² | 평균 raw MSE 개선 |
|---|---:|---:|---:|
| Llama activation | **4.54%** | **+0.00853** | **1.44%** |
| BGE best layer | 2.42% | +0.00343 | 0.53% |
| Qwen best layer | 1.91% | -0.00325 | -0.67% |
| Direct LM | 0.12% | +0.00006 | 0.01% |

주의:

- QLIKE에서는 Llama가 매우 일관적이다.
- H1 HAR-X/Lasso에서는 Llama가 QLIKE를 개선하지만 raw R²/MSE는 악화된다.
- 따라서 모든 지표의 모든 조합에서 무조건 개선된다고 표현하면 안 된다.
- 여기서의 비교는 각 family의 `baseline`과 해당 family의 `baseline + score` 비교다.

결과 파일:

- `outputs/full_forecast_method_table/full_results_long.csv`
- `outputs/full_forecast_method_table/comparison_table.csv`
- `outputs/full_forecast_method_table/table_news_h1.csv`
- `outputs/full_forecast_method_table/table_news_h5.csv`

---

## 13. 현재 해석

현재 증거가 지지하는 주장은 다음과 같다.

> Cutoff-safe Llama-2 Base의 중간 hidden activation에는 뉴스가 기업의 비정상 변동성 확대에 미치는 정보를 포착하는 구조가 있고, 단순한 train-only Ridge axis만으로도 같은 크기의 Qwen embedding, BGE embedding, 명시적 Direct LM rating보다 더 강한 forecasting signal을 얻는다.

근거:

- 동일 Llama checkpoint의 Direct rating은 실패했다.
- 작은 MLP도 Llama Ridge를 이기지 못했다.
- BGE/Qwen의 모든 레이어를 validation으로 선택해도 Llama activation이 높았다.
- Market adjustment를 제거해도 activation correlation 우위가 유지됐다.
- AR/HAR/HAR-X/MIDAS/LSTM과 H1/H5에서 QLIKE 우위가 유지됐다.

따라서 gain은 LLM에게 숫자를 직접 생성하게 해서 얻은 것이 아니라 **중간 activation에 포함된 정보를 supervised impact axis로 읽어낸 것**에 가깝다.

---

## 14. 해석상의 한계

1. BGE/Qwen은 cutoff-safe 모델로 주장하지 않고 embedding control로만 사용한다.
2. Qwen3-Embedding-8B는 2023 이후 모델이므로 엄격한 temporal comparison에서 Llama만 cutoff-safe하다.
3. 전 레이어 탐색은 validation에서 수행했으므로 test leakage는 없지만, 레이어 수가 많아 validation multiple-selection 효과가 있을 수 있다.
4. 최종 90행 forecasting 표는 point estimate이며 전체 family에 대한 paired bootstrap은 아직 붙이지 않았다.
5. 이전 OLS-only MLP forecasting에는 2,000회 date-block bootstrap이 있으며 Llama gain은 유의했다.
6. BGE L19/Qwen L28에 MLP를 다시 학습하는 조합은 아직 수행하지 않았다.
7. 최신 full forecast run은 metric CSV 저장까지 완료됐지만 row-level prediction Parquet 저장이 ticker dtype 문제로 실패했다. 코드는 수정했으며 metric 결과에는 영향이 없다.
8. Forecast classical model에는 ticker fixed effect가 있다. ticker effect까지 완전히 제거하지 않는 forecast가 필요하면 별도 control이 필요하다.
9. Absolute post-volatility target의 높은 correlation은 document impact보다 firm identity를 반영할 가능성이 높다.

---

## 15. 현재 범위에서 제외한 항목

### 15.1 8-K

8-K representation, impact axis, MLP, forecasting 관련 파일은 남아 있지만 사용자 최종 지시에 따라 다음에서 제외한다.

- 최종 표
- 본문 결론
- News activation 대 embedding 비교 주장

삭제하지 않고 archive로만 유지한다.

### 15.2 Llama-2 Chat/IT

Llama-2 Chat/IT는 full 2023 News test에 cutoff-safe하지 않아 본 실험에서 사용하지 않았다. 현재 activation과 Direct LM은 모두 Llama-2-7B Base다.

---

## 16. 다음 실험 후보

아직 실행하지 않은 후보를 우선순위 없이 기록한다.

- [ ] BGE L19와 Qwen L28에 MLP impact head 재학습
- [ ] 최종 18개 forecast 설정에 date-block bootstrap/Diebold-Mariano 계열 inference 추가
- [ ] 최신 full forecast의 row-level prediction Parquet 재생성
- [ ] Forecast model의 ticker fixed effect 제거 control
- [ ] Activation layer 주변부(L14–L18) stability 및 seed/bootstrap 분석
- [ ] Score를 연속값 하나가 아니라 여러 activation axis/subspace로 확장
- [ ] 동일 cutoff-safe 세대의 다른 Base LLM으로 activation 결과 재현

---

## 17. 주요 코드와 결과 경로

### 코드

- `scripts/news_cutoff_safe_llama2.py`
- `scripts/news_uncentered_linear_control.py`
- `scripts/news_large_embedding_control.py`
- `scripts/embedding_intermediate_layer_control.py`
- `scripts/news_uncentered_mlp_control.py`
- `scripts/news_mlp_forecast.py`
- `scripts/direct_llama2_rating.py`
- `scripts/direct_llama2_rating_vllm.py`
- `scripts/no_firm_vol_normalization_refit.py`
- `scripts/full_forecast_method_table.py`

### 핵심 결과

- `outputs/news_uncentered_mlp_control/test_metrics.csv`
- `outputs/news_uncentered_mlp_forecast/forecast_results.csv`
- `outputs/direct_llama2_rating/all_scores.parquet`
- `outputs/no_firm_vol_normalization_refit/news_test_metrics.csv`
- `outputs/full_forecast_method_table/comparison_table.csv`

---

## 18. 변경 기록

### 2026-07-22

- 현재까지의 News 데이터, target, representation, layer selection, MLP, Direct LM, target sensitivity, forecasting 설정과 결과를 최초 E2E 문서화
- 8-K를 최종 범위에서 제외
- 이후 실험은 이 문서를 기준으로 계속 누적 수정
