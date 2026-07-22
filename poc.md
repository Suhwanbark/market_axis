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

### 6.1 Ridge axis와 Direct LM

| 방법 | Test Spearman ρ | Test Pearson |
|---|---:|---:|
| Direct LM | -0.0015 | -0.0006 |
| Embedding (0.5B) | 0.0829 | 0.1045 |
| Embedding (7B) | 0.0893 | 0.1189 |
| LLM activation (L16) | **0.1511** | **0.1989** |

Embedding은 통상적인 사용 방식인 최종 output 기준이다.

### 6.2 Ridge axis: 전 레이어 탐색

Embedding 모델에서도 모든 layer를 탐색하고 validation에서 가장 좋은 layer를 선택했을 때:

| 방법 | 최종 output Test Spearman | 선택 layer | 선택 후 Test Spearman | 선택 후 Test Pearson |
|---|---:|---:|---:|---:|
| Embedding (0.5B) | 0.0829 | L19/24 | 0.0896 | 0.1176 |
| Embedding (7B) | 0.0893 | L28/36 | 0.1032 | 0.1247 |
| LLM activation | — | L16/32 | **0.1511** | **0.1989** |

### 6.3 간단한 forecasting MLP

작은 MLP head로 impact score를 학습했을 때:

| 방법 | Test Spearman | Test Pearson |
|---|---:|---:|
| Embedding (0.5B) + MLP | 0.0877 | 0.1170 |
| Embedding (7B) + MLP | 0.0910 | 0.1242 |
| LLM activation (L16) + MLP | **0.1462** | **0.1895** |

---

## 7. 최종 forecasting 설계

- Forecast target: 뉴스일 이후 Parkinson variance의 H1 및 H5
- Baseline: AR, HAR, HAR-X, MIDAS, LSTM
- Regression: AR·MIDAS는 OLS, HAR·HAR-X는 OLS/Lasso/Ridge
- 비교: 각 baseline과 동일 baseline에 Direct LM, Embedding(0.5B), Embedding(7B), LLM activation score를 추가한 모형
- 선택 및 학습: validation에서 hyperparameter를 선택한 뒤 train+validation으로 최종 refit하고 test를 한 번 평가
- 평가 지표: QLIKE, raw R², raw MSE

---

## 8. 최종 forecasting 결과

QLIKE와 raw MSE는 각 baseline 대비 감소율(%)이며, raw R²는 baseline 대비 변화량(Δ)이다. 모두 높을수록 좋고 음수는 성능 악화를 뜻한다.

### 8.1 H1: 1일 변동성

| Model | Metric | Direct LM | Embedding (0.5B) | Embedding (7B) | LLM activation |
|---|---|---:|---:|---:|---:|
| AR | QLIKE 개선률 | 0.18% | 2.72% | 2.93% | **3.97%** |
|  | Δ raw R² | +0.00011 | +0.00326 | +0.00204 | **+0.00658** |
|  | Raw MSE 개선률 | 0.01% | 0.41% | 0.26% | **0.83%** |
| HAR/OLS | QLIKE 개선률 | 0.16% | 2.89% | 3.18% | **4.27%** |
|  | Δ raw R² | +0.00003 | +0.00231 | +0.00140 | **+0.00465** |
|  | Raw MSE 개선률 | 0.00% | 0.29% | 0.18% | **0.59%** |
| HAR/Lasso | QLIKE 개선률 | 0.14% | 3.67% | 3.99% | **5.19%** |
|  | Δ raw R² | +0.00010 | +0.00409 | +0.00329 | **+0.00533** |
|  | Raw MSE 개선률 | 0.01% | 0.52% | 0.41% | **0.67%** |
| HAR/Ridge | QLIKE 개선률 | 0.16% | 3.12% | 3.42% | **4.54%** |
|  | Δ raw R² | +0.00009 | +0.00273 | +0.00202 | **+0.00467** |
|  | Raw MSE 개선률 | 0.01% | 0.35% | 0.26% | **0.59%** |
| HAR-X/OLS | QLIKE 개선률 | 0.27% | 2.86% | 3.14% | **4.20%** |
|  | Δ raw R² | -0.00028 | +0.00335 | +0.00430 | **+0.00632** |
|  | Raw MSE 개선률 | -0.04% | 0.43% | 0.55% | **0.80%** |
| HAR-X/Lasso | QLIKE 개선률 | 0.27% | 1.50% | 0.95% | **2.09%** |
|  | Δ raw R² | -0.00027 | **+0.00251** | -0.00574 | -0.00529 |
|  | Raw MSE 개선률 | -0.03% | **0.32%** | -0.73% | -0.67% |
| HAR-X/Ridge | QLIKE 개선률 | 0.25% | 3.13% | 3.41% | **4.53%** |
|  | Δ raw R² | -0.00035 | +0.00322 | +0.00410 | **+0.00550** |
|  | Raw MSE 개선률 | -0.04% | 0.41% | 0.52% | **0.70%** |
| MIDAS/OLS | QLIKE 개선률 | 0.22% | 3.03% | 3.25% | **4.36%** |
|  | Δ raw R² | +0.00025 | +0.00348 | +0.00217 | **+0.00626** |
|  | Raw MSE 개선률 | 0.03% | 0.44% | 0.27% | **0.79%** |
| LSTM | QLIKE 개선률 | 0.44% | 3.82% | 3.38% | **5.45%** |
|  | Δ raw R² | +0.00050 | +0.00905 | +0.00538 | **+0.01175** |
|  | Raw MSE 개선률 | 0.06% | 1.11% | 0.66% | **1.44%** |

<details>
<summary>H1 raw-scale 절대 성능 보기</summary>

| Model | Metric | Baseline | Direct LM | Embedding (0.5B) | Embedding (7B) | LLM activation |
|---|---|---:|---:|---:|---:|---:|
| AR | QLIKE | 0.37054 | 0.36987 | 0.36047 | 0.35968 | **0.35581** |
|  | Raw R² | 0.20918 | 0.20929 | 0.21244 | 0.21121 | **0.21575** |
|  | Raw MSE | 9.499e-08 | 9.497e-08 | 9.459e-08 | 9.474e-08 | **9.420e-08** |
| HAR/OLS | QLIKE | 0.37504 | 0.37445 | 0.36420 | 0.36313 | **0.35904** |
|  | Raw R² | 0.21691 | 0.21695 | 0.21922 | 0.21831 | **0.22156** |
|  | Raw MSE | 9.406e-08 | 9.405e-08 | 9.378e-08 | 9.389e-08 | **9.350e-08** |
| HAR/Lasso | QLIKE | 0.38486 | 0.38434 | 0.37073 | 0.36949 | **0.36487** |
|  | Raw R² | 0.20766 | 0.20776 | 0.21174 | 0.21094 | **0.21299** |
|  | Raw MSE | 9.517e-08 | 9.516e-08 | 9.468e-08 | 9.477e-08 | **9.453e-08** |
| HAR/Ridge | QLIKE | 0.37734 | 0.37674 | 0.36555 | 0.36443 | **0.36020** |
|  | Raw R² | 0.21406 | 0.21415 | 0.21679 | 0.21607 | **0.21873** |
|  | Raw MSE | 9.440e-08 | 9.439e-08 | 9.407e-08 | 9.416e-08 | **9.384e-08** |
| HAR-X/OLS | QLIKE | 0.36737 | 0.36639 | 0.35686 | 0.35582 | **0.35193** |
|  | Raw R² | 0.21496 | 0.21468 | 0.21831 | 0.21926 | **0.22128** |
|  | Raw MSE | 9.429e-08 | 9.433e-08 | 9.389e-08 | 9.378e-08 | **9.353e-08** |
| HAR-X/Lasso | QLIKE | 0.36752 | 0.36654 | 0.36201 | 0.36404 | **0.35985** |
|  | Raw R² | 0.21507 | 0.21480 | **0.21758** | 0.20933 | 0.20978 |
|  | Raw MSE | 9.428e-08 | 9.431e-08 | **9.398e-08** | 9.497e-08 | 9.491e-08 |
| HAR-X/Ridge | QLIKE | 0.36935 | 0.36841 | 0.35778 | 0.35675 | **0.35261** |
|  | Raw R² | 0.21510 | 0.21475 | 0.21832 | 0.21920 | **0.22060** |
|  | Raw MSE | 9.427e-08 | 9.432e-08 | 9.389e-08 | 9.378e-08 | **9.361e-08** |
| MIDAS/OLS | QLIKE | 0.37840 | 0.37756 | 0.36694 | 0.36610 | **0.36190** |
|  | Raw R² | 0.20894 | 0.20919 | 0.21242 | 0.21111 | **0.21520** |
|  | Raw MSE | 9.502e-08 | 9.499e-08 | 9.460e-08 | 9.475e-08 | **9.426e-08** |
| LSTM | QLIKE | 0.38008 | 0.37839 | 0.36558 | 0.36723 | **0.35935** |
|  | Raw R² | 0.18471 | 0.18521 | 0.19376 | 0.19010 | **0.19646** |
|  | Raw MSE | 9.793e-08 | 9.786e-08 | 9.684e-08 | 9.728e-08 | **9.651e-08** |

</details>

### 8.2 H5: 5일 변동성

| Model | Metric | Direct LM | Embedding (0.5B) | Embedding (7B) | LLM activation |
|---|---|---:|---:|---:|---:|
| AR | QLIKE 개선률 | ~0% | 1.34% | 0.25% | **4.16%** |
|  | Δ raw R² | +0.00001 | +0.00279 | -0.01303 | **+0.01383** |
|  | Raw MSE 개선률 | 0.00% | 0.50% | -2.32% | **2.47%** |
| HAR/OLS | QLIKE 개선률 | ~0% | 1.66% | 0.65% | **4.84%** |
|  | Δ raw R² | +0.00008 | +0.00210 | -0.01169 | **+0.01148** |
|  | Raw MSE 개선률 | 0.01% | 0.40% | -2.22% | **2.18%** |
| HAR/Lasso | QLIKE 개선률 | ~0% | 2.35% | 0.00% | **5.16%** |
|  | Δ raw R² | -0.00003 | +0.00325 | -0.01691 | **+0.00979** |
|  | Raw MSE 개선률 | -0.01% | 0.61% | -3.18% | **1.84%** |
| HAR/Ridge | QLIKE 개선률 | ~0% | 1.85% | 0.81% | **4.94%** |
|  | Δ raw R² | +0.00001 | +0.00256 | -0.00990 | **+0.01044** |
|  | Raw MSE 개선률 | 0.00% | 0.49% | -1.88% | **1.99%** |
| HAR-X/OLS | QLIKE 개선률 | 0.02% | 1.57% | 0.49% | **4.66%** |
|  | Δ raw R² | +0.00003 | +0.00371 | -0.00486 | **+0.01281** |
|  | Raw MSE 개선률 | 0.01% | 0.73% | -0.96% | **2.53%** |
| HAR-X/Lasso | QLIKE 개선률 | 0.01% | 2.27% | 3.11% | **5.09%** |
|  | Δ raw R² | +0.00007 | +0.00298 | +0.00223 | **+0.00933** |
|  | Raw MSE 개선률 | 0.01% | 0.58% | 0.44% | **1.83%** |
| HAR-X/Ridge | QLIKE 개선률 | 0.02% | 1.74% | 0.41% | **4.76%** |
|  | Δ raw R² | +0.00005 | +0.00324 | -0.00576 | **+0.01046** |
|  | Raw MSE 개선률 | 0.01% | 0.64% | -1.15% | **2.08%** |
| MIDAS/OLS | QLIKE 개선률 | ~0% | 1.62% | 0.40% | **4.73%** |
|  | Δ raw R² | -0.00011 | +0.00310 | -0.01229 | **+0.01207** |
|  | Raw MSE 개선률 | -0.02% | 0.58% | -2.32% | **2.28%** |
| LSTM | QLIKE 개선률 | 0.11% | 2.35% | 0.52% | **4.79%** |
|  | Δ raw R² | +0.00070 | +0.00410 | -0.00523 | **+0.01751** |
|  | Raw MSE 개선률 | 0.12% | 0.72% | -0.91% | **3.05%** |

<details>
<summary>H5 raw-scale 절대 성능 보기</summary>

| Model | Metric | Baseline | Direct LM | Embedding (0.5B) | Embedding (7B) | LLM activation |
|---|---|---:|---:|---:|---:|---:|
| AR | QLIKE | 0.15284 | 0.15284 | 0.15079 | 0.15246 | **0.14648** |
|  | Raw R² | 0.43948 | 0.43949 | 0.44228 | 0.42646 | **0.45332** |
|  | Raw MSE | 2.473e-08 | 2.473e-08 | 2.460e-08 | 2.530e-08 | **2.412e-08** |
| HAR/OLS | QLIKE | 0.15098 | 0.15098 | 0.14847 | 0.15001 | **0.14368** |
|  | Raw R² | 0.47253 | 0.47261 | 0.47464 | 0.46084 | **0.48401** |
|  | Raw MSE | 2.327e-08 | 2.327e-08 | 2.318e-08 | 2.378e-08 | **2.276e-08** |
| HAR/Lasso | QLIKE | 0.15504 | 0.15505 | 0.15140 | 0.15504 | **0.14704** |
|  | Raw R² | 0.46838 | 0.46836 | 0.47163 | 0.45147 | **0.47817** |
|  | Raw MSE | 2.345e-08 | 2.345e-08 | 2.331e-08 | 2.420e-08 | **2.302e-08** |
| HAR/Ridge | QLIKE | 0.15197 | 0.15197 | 0.14916 | 0.15073 | **0.14446** |
|  | Raw R² | 0.47415 | 0.47416 | 0.47671 | 0.46425 | **0.48460** |
|  | Raw MSE | 2.320e-08 | 2.320e-08 | 2.308e-08 | 2.363e-08 | **2.274e-08** |
| HAR-X/OLS | QLIKE | 0.14512 | 0.14509 | 0.14284 | 0.14441 | **0.13836** |
|  | Raw R² | 0.49449 | 0.49452 | 0.49820 | 0.48963 | **0.50730** |
|  | Raw MSE | 2.230e-08 | 2.230e-08 | 2.214e-08 | 2.251e-08 | **2.173e-08** |
| HAR-X/Lasso | QLIKE | 0.14888 | 0.14887 | 0.14550 | 0.14426 | **0.14130** |
|  | Raw R² | 0.48865 | 0.48872 | 0.49163 | 0.49088 | **0.49798** |
|  | Raw MSE | 2.256e-08 | 2.255e-08 | 2.243e-08 | 2.246e-08 | **2.215e-08** |
| HAR-X/Ridge | QLIKE | 0.14548 | 0.14545 | 0.14295 | 0.14488 | **0.13856** |
|  | Raw R² | 0.49696 | 0.49701 | 0.50019 | 0.49119 | **0.50742** |
|  | Raw MSE | 2.219e-08 | 2.219e-08 | 2.205e-08 | 2.245e-08 | **2.173e-08** |
| MIDAS/OLS | QLIKE | 0.15146 | 0.15146 | 0.14901 | 0.15085 | **0.14429** |
|  | Raw R² | 0.47061 | 0.47050 | 0.47371 | 0.45832 | **0.48268** |
|  | Raw MSE | 2.335e-08 | 2.336e-08 | 2.322e-08 | 2.390e-08 | **2.282e-08** |
| LSTM | QLIKE | 0.15567 | 0.15550 | 0.15202 | 0.15486 | **0.14821** |
|  | Raw R² | 0.42680 | 0.42751 | 0.43091 | 0.42157 | **0.44432** |
|  | Raw MSE | 2.529e-08 | 2.525e-08 | 2.510e-08 | 2.552e-08 | **2.451e-08** |

</details>

---

## 9. 현재 해석

현재 증거가 지지하는 주장은 다음과 같다.

> Cutoff-safe LLM의 중간 activation에는 뉴스가 기업의 비정상 변동성 확대에 미치는 정보를 포착하는 구조가 있고, 단순한 train-only Ridge axis만으로도 Embedding (0.5B), Embedding (7B), Direct LM보다 더 강한 forecasting signal을 얻는다.

근거:

- 동일 LLM checkpoint의 Direct rating은 실패했다.
- 작은 MLP도 LLM activation Ridge를 이기지 못했다.
- Embedding (0.5B, 7B)의 모든 레이어를 validation으로 선택해도 LLM activation이 높았다.
- Market adjustment를 제거해도 activation correlation 우위가 유지됐다.
- AR/HAR/HAR-X/MIDAS/LSTM과 H1/H5에서 QLIKE 우위가 유지됐다.

따라서 gain은 LLM에게 숫자를 직접 생성하게 해서 얻은 것이 아니라 **중간 activation에 포함된 정보를 supervised impact axis로 읽어낸 것**에 가깝다.

---

## 10. 해석상의 한계

1. Embedding (0.5B, 7B)은 cutoff-safe 모델로 주장하지 않고 embedding control로만 사용한다.
2. Embedding (7B)은 2023 이후 모델이므로 엄격한 temporal comparison에서는 LLM activation만 cutoff-safe하다.
3. 전 레이어 탐색은 validation에서 수행했으므로 test leakage는 없지만, 레이어 수가 많아 validation multiple-selection 효과가 있을 수 있다.
4. 최종 90행 forecasting 표는 point estimate이며 전체 family에 대한 paired bootstrap은 아직 붙이지 않았다.
5. 이전 OLS-only MLP forecasting에는 2,000회 date-block bootstrap이 있으며 LLM activation gain은 유의했다.
6. Embedding (0.5B) L19와 Embedding (7B) L28에 MLP를 다시 학습하는 조합은 아직 수행하지 않았다.
7. 최신 full forecast run은 metric CSV 저장까지 완료됐지만 row-level prediction Parquet 저장이 ticker dtype 문제로 실패했다. 코드는 수정했으며 metric 결과에는 영향이 없다.
8. Forecast classical model에는 ticker fixed effect가 있다. ticker effect까지 완전히 제거하지 않는 forecast가 필요하면 별도 control이 필요하다.
9. Absolute post-volatility target의 높은 correlation은 document impact보다 firm identity를 반영할 가능성이 높다.
