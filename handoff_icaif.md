# Market-Impact Activation Axis 연구 Handoff

마지막 업데이트: 2026-07-21 UTC

이 문서는 `/home/hob/workspace/familiar`에서 진행한 연구를 다른 연구자 또는 다른 서버가 이어받을 수 있도록 정리한 단일 handoff 문서다. 연구 질문의 변화, 완료한 실험, 현재 가장 중요한 결과, 파일 구조, 환경, 데이터, 정확한 재현 순서와 남은 한계를 함께 기록한다.

---

## 0. 가장 먼저 읽을 요약

### 현재 연구 질문

최초 질문은 **LLM 내부에 익숙한 정보와 낯선 정보를 구분하는 familiar/unfamiliar axis가 있는가**였다. 그러나 금융에서는 같은 사건도 기업에 따라 익숙함의 의미가 달라졌고, 인위적인 기업-뉴스 mismatch나 기업의 과거 activation에서 먼 정도가 실제 중요한 뉴스를 안정적으로 찾지 못했다.

현재 질문은 다음처럼 바뀌었다.

> LLM이 금융 뉴스나 공시를 읽을 때, 공개 이후 시장에 큰 영향을 미치는 정보를 나타내는 선형 방향이 중간 activation에 존재하는가? 그 방향의 projection score는 direct LLM rating이나 전용 embedding보다 실제 market impact를 잘 순위화하고, 기존 volatility model에 추가 예측 신호를 제공하는가?

현재 axis는 무감독 familiarity axis가 아니다. 실제 사후 시장반응으로 학습한 **supervised market-impact activation axis**다.

### 현재 가장 중요한 main 후보

- 문서별 ground truth는 Parkinson high-low volatility를 이용한다.
- 기업의 공개 전 20일 대비 공개 후 5일 변동성 증가율에서 같은 기간 시장 전체 증가율을 뺀다.
- News main에는 train ticker별 평균 label과 평균 representation을 추가로 제거한다.
- Qwen2.5-7B 중간 activation, BGE-M3 embedding, Qwen direct 1~9 rating을 비교한다.
- Axis와 embedding Ridge는 train에서 학습하고, layer와 Ridge alpha는 validation에서 선택한다.
- Test에서는 document score와 실제 normalized impact의 Spearman, decile 관계, 그리고 AR/HAR/HAR-X/MIDAS forecasting을 본다.

### 최신 News 결과

| 방법 | 2023 test Spearman |
|---|---:|
| Direct Qwen 1~9 rating | 0.0151 |
| BGE-M3 + Ridge | 0.0905 |
| Qwen2.5 activation L20 + Ridge | **0.1500** |

- News activation layer: L20
- Activation Ridge alpha: 1
- BGE Ridge alpha: 10
- Activation-BGE test Spearman 차이: 0.0595
- Date-block 95% CI: [0.0427, 0.0755]
- Ticker-block 95% CI: [0.0378, 0.0813]

News Parkinson forecasting 8개 설정의 평균 QLIKE 개선:

| Text feature | 평균 개선 |
|---|---:|
| BGE score | 2.42% |
| Activation score | **4.50%** |

### 최신 8-K 결과

문서 score와 normalized Parkinson impact의 2025 test Spearman:

| 방법 | Test Spearman |
|---|---:|
| BGE-M3 + Ridge | **0.4100** |
| Qwen2.5 activation L27 + Ridge | 0.3813 |

8-K forecasting 8개 설정의 평균 QLIKE 개선:

| Text feature | 평균 개선 |
|---|---:|
| BGE score | 15.95% |
| Activation score | **18.58%** |

다만 8-K에서는 activation이 모든 지표에서 우월하지 않다. 1일 QLIKE는 activation이 강했지만, Raw MSE와 Raw R2는 대체로 embedding이 더 좋았다. 따라서 현재 가장 안전한 결론은 다음과 같다.

> News에서는 activation이 direct rating과 BGE보다 일관되게 강하다. 8-K에서는 activation과 embedding 모두 강한 추가 신호이며, activation의 우위는 QLIKE와 1일 horizon에 집중된다.

---

## 1. 연구가 어떻게 변했는가

### V0: 보편적인 familiar/unfamiliar axis

처음에는 모델이 자주 접했을 법한 정보와 낯선 정보를 forward했을 때 activation 차이가 하나의 전역 방향으로 나타나는지 보려 했다.

```text
familiar text activation
vs
unfamiliar text activation
-> global familiarity direction
```

문제는 금융에서 familiarity가 텍스트만의 속성이 아니라는 점이었다. 예를 들어 AI accelerator 증설은 NVIDIA에는 전형적이지만 지역은행에는 매우 이례적이다.

### V3: Business-description contrast

동일한 뉴스 본문 앞에 올바른 기업 business description 또는 다른 기업 description을 붙였다.

```text
Correct: Apple description + Apple news
Wrong:   JPMorgan description + same Apple news
```

- 기업 설명은 SEC 10-K business description과 일부 yfinance fallback으로 만들었다.
- Qwen2.5-7B의 뉴스 token hidden state만 평균했다.
- Mean difference, logistic coefficient, pair-delta PC1을 비교했다.
- Layer 27 logistic axis는 단순 ticker mean 제거 후 pair accuracy 0.970, AUC 0.955였다.
- context 기업 identity까지 제거하면 accuracy 0.698, AUC 0.645로 크게 떨어졌다.
- 2023 test 495건에서 5일 absolute return/CAR과 탐색적 관계가 있었지만 ticker/date clustered inference에서는 유의하지 않았다.
- RV5 forecasting 개선은 RMSE 약 0.03%로 사실상 없었다.

결론: 높은 pair 분류 성능의 상당 부분이 familiarity가 아니라 기업 identity와 prompt compatibility였다.

### V4: Ticker/name header contrast

Business description 대신 올바른 ticker/company name 또는 잘못된 ticker header를 같은 뉴스에 붙였다.

```text
Correct: Company AAPL + Apple news
Wrong:   Company MSFT + same Apple news
```

Pair를 잘 구분하는 axis와 실제 미래 변동성과 연결되는 axis가 일치하지 않았다. 일부 PC1 score는 변동성과 음의 방향으로 나타났고, 부호를 사후에 뒤집어야 했기 때문에 familiarity 해석이 약했다.

결론: 인위적인 mismatch를 학습하는 것과 실제 surprise/importance를 학습하는 것은 다르다.

### V7: Ticker activation prior distance

인위적인 wrong pair를 없애고, ticker별 과거 뉴스/filing activation 분포를 prior로 만든 뒤 새 뉴스가 prior에서 얼마나 먼지 측정했다.

- Cosine distance
- Relative cosine
- Diagonal Mahalanobis distance
- 모든 layer 비교

가장 강한 결과는 filing-prior Mahalanobis L21과 RV5 Spearman 0.0244였다. 거의 0에 가까웠다.

결론: 과거 기업 activation에서 멀다는 사실만으로 중요한 뉴스를 찾을 수 없었다. 문서 장르 차이도 크게 섞였다.

### V8: Supervised market-impact axis

질문을 familiarity에서 실제 market impact로 바꿨다.

- 뉴스 또는 8-K 본문 activation을 입력 X로 사용
- 공개 이후 시장반응을 이용한 연속형 label y 생성
- Ridge coefficient를 impact가 높아지는 activation direction으로 해석
- Validation으로 layer를 고정하고 미래 test에서 평가

초기 V8은 absolute abnormal return, RV5, range, volume에서 metadata로 예상되는 부분을 nuisance Ridge로 제거한 composite label을 사용했다.

초기 8-K 결과:

| 설정 | Axis | Test Spearman |
|---|---|---:|
| 기본 composite | Ridge L28 | 0.209 |
| 중복 및 ticker/date strict | Matched-PC1 L8 | 0.147 |

초기 결과는 forecasting도 개선했지만 label 통제가 복잡하고 해석이 작위적이라는 문제가 있었다.

### V9: News+8-K joint raw axis

News와 8-K를 domain-balanced 방식으로 함께 학습하는 joint pipeline을 완성했다.

- Qwen2.5 joint: Ridge L25, z_range5
- Qwen3.5 joint: Ridge L32, z_range5
- Qwen2.5 axis cosine: news-vs-filing 0.217, joint-vs-news 0.605, joint-vs-filing 0.722
- Raw range-level test correlation은 매우 높았지만 within-ticker correlation은 작거나 8-K에서 음수였다.

이 버전은 raw future range level을 사용해 평소 변동성이 큰 기업과 문서 장르 차이를 많이 포함한다. 따라서 완료된 exploratory 결과지만 현재 main으로 사용하지 않는다. 새로운 normalized label로 joint axis를 다시 학습하는 작업은 남아 있다.

### V10: 단순 normalized Parkinson impact와 representation 비교

교수 피드백에 따라 복잡한 nuisance residual label을 단순하고 직관적인 volatility expansion으로 교체했다. 또한 activation의 의미를 주장하려면 다음 두 baseline을 이겨야 한다고 설정했다.

1. LLM에 직접 1~9점을 물어보는 zero-shot baseline
2. 전용 embedding을 실제 impact label에 Ridge로 연결하는 supervised baseline

현재 최신 결과는 이 V10이다.

---

## 2. 현재 main label의 정확한 정의

### 2.1 일별 Parkinson variance

기업 i, 거래일 d의 고가 H와 저가 L을 사용한다.

```text
ParkinsonVar(i,d) = log(H(i,d) / L(i,d))^2 / (4 * log(2))
```

종가 수익률 제곱보다 장중 가격 범위를 사용하기 때문에 일별 변동성 측정이 상대적으로 안정적이다.

### 2.2 기업의 공개 전후 변동성

문서의 reaction origin을 t라고 할 때:

```text
firm_pre20_var  = 문서 반응 시작 전 20거래일 ParkinsonVar 평균
firm_post5_var  = 문서 반응 시작 후 5거래일 ParkinsonVar 평균

firm_pre20_sigma = sqrt(firm_pre20_var)
firm_post5_sigma = sqrt(firm_post5_var)
```

기업의 변동성 확장률은 다음과 같다.

```text
firm expansion = log(firm_post5_sigma / firm_pre20_sigma)
```

### 2.3 같은 기간 시장 변동성 제거

각 날짜에 universe 기업의 Parkinson variance를 equal-weight 평균해 market daily variance를 만든다. 같은 방식으로 market pre20과 post5 volatility를 계산한다.

최종 label:

```text
y(i,t)
  = log(firm_post5_sigma / firm_pre20_sigma)
  - log(market_post5_sigma / market_pre20_sigma)
```

동일한 식을 variance 단위로 쓰면 다음과 같다.

```text
y(i,t)
  = 0.5 * [
      log(firm_post5_var) - log(firm_pre20_var)
      - log(market_post5_var) + log(market_pre20_var)
    ]
```

예를 들어 기업 변동성이 평소의 2배, 시장 변동성이 같은 기간 1.2배가 됐다면:

```text
y = log(2) - log(1.2) = log(2 / 1.2) = 0.511
```

즉 시장 전체 상승을 고려해도 해당 기업이 평소보다 상대적으로 크게 흔들렸다는 뜻이다.

### 2.4 News timing

- FinTexTS 뉴스는 정확한 공개 시각이 없고 날짜만 있다.
- News date를 t라고 할 때 pre20은 t 이전 20거래일이다.
- Post5와 forecast target은 t 다음 실제 거래일 t+1부터 t+5다.
- 따라서 뉴스 당일 가격을 label 또는 forecast target에 넣지 않는다.

### 2.5 8-K timing

- SEC acceptance timestamp가 09:30 ET 이전이면 당일 session이 첫 reaction session이다.
- 09:30 ET 이후면 다음 거래 session이 첫 reaction session이다.
- Pre20은 첫 reaction session 직전까지다.
- Post5는 첫 reaction session을 포함한 5거래일이다.

### 2.6 Ticker centering

News 최신 main은 위 비율 정규화에 더해 train ticker별 평균을 label과 representation 양쪽에서 제거한다.

```text
centered_y(i) = y(i) - train_mean_y(ticker_i)
centered_h(i) = h(i) - train_mean_h(ticker_i)
centered_e(i) = e(i) - train_mean_e(ticker_i)
```

Validation/test 평균은 절대 사용하지 않는다. Train에 없는 ticker는 train 전체 평균을 fallback으로 쓴다.

Pre20 정규화가 각 event 직전의 volatility scale을 조정한다면, ticker centering은 그래도 반복적으로 남아 있는 기업별 평균 label과 representation identity를 제거한다. 이 추가 단계가 반드시 유일한 정답은 아니므로 논문에서는 다음을 함께 보고하는 것이 좋다.

- Ratio + market adjustment만 사용
- 위 설정 + train ticker centering

8-K 최신 normalized forecast는 ticker centering을 axis 학습에 적용하지 않았다. 대신 forecasting panel에 ticker fixed effect가 들어간다. News와 8-K의 이 차이는 향후 통일해야 한다.

### 2.7 Market 계산의 현재 구현상 주의

- News market은 FinTexTS 100개 ticker의 날짜별 equal-weight Parkinson variance다.
- 8-K label 구현은 cached price panel의 날짜별 모든 series를 평균한다. 현재 코드에서는 SPY를 명시적으로 제외하지 않아 SPY 한 종목이 equal-weight 평균에 포함될 수 있다.
- 8-K HAR-X forecast의 market HAR는 별도로 SPY를 사용한다.

SPY 한 종목의 영향은 작지만, 최종 논문 재실험에서는 label market을 `SPY` 하나로 할지, investable universe equal-weight로 할지 사전에 고정하고 코드에서 명시해야 한다.

---

## 3. 데이터와 시간 분할

### 3.1 News

- Source: Hugging Face `EXAONE-BI/FinTexTS`
- Cached snapshot: `6a4ce04ac3f2e8a4ae15a5826e6980b50560eb29`
- Text: `targetCompany_category1`, `targetCompany_category2`, `targetCompany_category3`를 중복 제거 후 결합
- Unit: ticker-date별 target-company news summary 하나
- Universe: 100 tickers
- Filing 관련 column은 최신 News text에 사용하지 않는다.

| Split | 기간 | 최신 normalized axis 표본 |
|---|---|---:|
| Train | 2019-2021 | 25,767 |
| Validation | 2022 | 12,051 |
| Test | 2023 | 12,478 |

원래 calendar-corrected news record는 train 25,806건이다. 최신 label의 pre20/post5 요건을 만족하지 못한 일부 행이 빠져 axis 표본은 25,767건이다.

FinTexTS에 휴장일 OHLC가 전일 값으로 복제된 45개 날짜가 있어 해당 가격 행과 그 날짜의 뉴스는 이전 preprocessing에서 제거했다.

### 3.2 8-K

- Source: Hugging Face `Disclosures-SSRC/8k_disclosure_dataset`
- Form: root form 8-K
- Document choice: accession별 EX-99.1, EX-99, 8-K 순으로 preferred document 하나
- Text length: 500~5,000 characters
- 최소 문서 수와 split coverage를 만족하는 기업만 유지
- SEC submissions metadata로 acceptance timestamp를 결합

Document-level normalized label 표본:

| Split | 기간 | 문서 수 |
|---|---|---:|
| Train | 2022-2023 | 700 |
| Validation | 2024 | 1,008 |
| Test | 2025 | 707 |

Forecast event panel은 같은 ticker/reaction session의 여러 문서 score 중 최대값을 사용한다.

| Split | Forecast event 수 |
|---|---:|
| Train | 399 |
| Validation | 625 |
| Test | 453 |

8-K 원천 dataset과 yfinance 가격은 revision이 고정되지 않았으므로 **숫자까지 정확히 재현하려면 현재 저장된 parquet을 복사해야 한다.** 원천에서 다시 받으면 수정주가나 데이터 revision 때문에 값이 달라질 수 있다.

---

## 4. Text representation과 axis 학습

### 4.1 Direct LLM baseline

Model: Qwen2.5-7B-Instruct

Prompt:

```text
Assess the magnitude of the stock-price volatility that the document itself
is likely to cause relative to the company's usual level. Judge magnitude,
not whether the price will rise or fall. Use 1 for routine information with
little expected effect and 9 for information likely to cause major repricing
or uncertainty. Answer with exactly one digit from 1 to 9.

Document:
{text}

Score:
```

실제로 한 digit을 greedy decode하지 않는다. 다음 token의 1~9 logit만 모아 softmax한 뒤 기대값을 사용한다.

```text
direct_score = sum(k * P(next token = k | prompt, text), k=1..9)
```

- 별도 supervised fitting이 없다.
- Validation과 test 문서에만 계산했다.
- News 최대 512 document tokens, filing 최대 2,048 tokens다.
- 과거 20일 가격이나 시장 변동성을 prompt에 주지 않는다.
- 따라서 정규화된 label을 직접 계산하는 baseline이 아니라, LLM에 텍스트만 주고 물어보는 것으로 같은 ranking을 얻을 수 있는지 보는 baseline이다.

### 4.2 BGE embedding baseline

- Model: `BAAI/bge-m3`, 약 568M parameters
- Output dimension: 1,024
- Pooling: final hidden state의 CLS token
- Row-wise L2 normalization
- News max 512 tokens, filing max 2,048 tokens
- Train impact label을 예측하는 Ridge를 학습

최신 News에서는 alpha grid `[0.01, 0.1, 1, 10, 100, 1000]` 중 validation Spearman이 가장 높은 alpha=10을 선택했다.

### 4.3 Qwen activation

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Exact snapshot: `a09a35458c702b33eeacc393d103063234e8bc28`
- Layers: 1~28
- Hidden size: 3,584
- Activation storage: float16
- Model forward: BF16 또는 FP16
- Pooling: 본문 token hidden state의 mean pooling
- 각 document-layer activation을 L2 normalize

News 기존 extraction은 prefix `Financial news:\n`와 suffix newline을 사용하되, pooling은 뉴스 본문 token span에만 수행했다. Ticker/date header는 넣지 않았다.

Joint/8-K extraction은 body text만 최대 2,048 tokens로 처리하고, 긴 문서는 256-token overlap chunk를 만든 뒤 새 token 수를 weight로 사용해 chunk mean을 합쳤다.

### 4.4 Ridge axis

Train의 centered activation을 X, centered impact label을 y로 둔다.

```text
y ~= b + X beta
```

Ridge coefficient beta가 impact가 높아지는 activation direction이다.

```text
unit_axis = beta / norm(beta)
```

현재 저장된 평가 score는 `Ridge.predict`, 즉 `b + beta dot h`다. 저장된 `unit_axis`에 projection한 값은 positive affine scaling만 다르므로 Spearman ranking은 동일하다. Forecasting에 넣기 전 score를 표준화하므로 coefficient scale도 결과에 영향을 주지 않는다.

### 4.5 Validation 선택

News 최신 protocol:

1. Train에서 BGE alpha별 Ridge 학습
2. Train에서 layer 1~28 x alpha별 activation Ridge 학습
3. 2022 validation에서 score와 normalized impact의 Spearman 계산
4. BGE alpha 하나, activation layer와 alpha 하나를 고정
5. 2023 test를 평가

선택 결과:

| Representation | Validation 선택 |
|---|---|
| BGE | alpha=10, validation rho=0.0929 |
| Activation | L20, alpha=1, validation rho=0.1176 |

8-K 최신 normalized 결과는 layer는 validation으로 L27을 선택했지만 Ridge alpha는 10으로 고정했다. 8-K도 News처럼 alpha grid를 validation에서 선택하는 재실험이 필요하다.

---

## 5. 최신 document-ranking 결과

### 5.1 News

Ground truth는 market-adjusted Parkinson Post5/Pre20 expansion에 train ticker centering을 적용한 값이다.

| Method | Test Spearman | Test Pearson |
|---|---:|---:|
| Direct Qwen | 0.0151 | 0.0104 |
| BGE-M3 Ridge | 0.0905 | 0.1215 |
| Qwen activation L20 Ridge | **0.1500** | **0.1899** |

Activation-BGE Spearman 차이 0.0595는 date-block과 ticker-block bootstrap 모두에서 95% CI가 0보다 컸다.

### 5.2 8-K

Ground truth는 market-adjusted Parkinson Post5/Pre20 expansion이다. Ticker centering은 하지 않았다.

| Method | Test Spearman | Test Pearson | Ticker-FE Spearman |
|---|---:|---:|---:|
| BGE-M3 Ridge | **0.4100** | 0.3771 | **0.2855** |
| Qwen activation L27 Ridge | 0.3813 | **0.3837** | 0.2401 |

8-K에서는 embedding이 rank correlation에서 더 높다. 따라서 activation 고유의 우위를 주장하려면 현재 8-K 결과만으로는 부족하다.

### 5.3 Market-only diagnostic

기업의 pre20 normalization을 하지 않고 미래 5일 변동성 level에서 같은 날짜 market level만 빼면 News 결과가 매우 커졌다.

| Method | Test Spearman |
|---|---:|
| Direct Qwen | 약 0.241 |
| BGE | 약 0.551 |
| Activation | 약 0.611 |

하지만 이 설정은 Tesla나 biotech가 평소부터 더 volatile한 기업 간 차이를 포함한다. 높은 수치는 representation이 기업/sector volatility level을 잘 인코딩한다는 diagnostic이지, 문서 고유 surprise의 main evidence가 아니다.

---

## 6. Forecasting 설정

### 6.1 Forecast target

Axis 학습 label과 forecasting target은 다르다.

- Axis label: Post5/Pre20 firm expansion에서 market expansion을 뺀 문서별 normalized impact
- Forecast target: 미래 실제 Parkinson variance

1일 target:

```text
future Parkinson variance of first reaction day
```

5일 target:

```text
mean daily Parkinson variance over five reaction days
```

Output은 5개가 아니라 평균 variance 하나다.

### 6.2 Panel model

Ticker마다 별도 모형을 학습하지 않는다. 모든 기업을 합친 panel OLS를 사용하고 ticker one-hot fixed effect를 넣는다.

```text
log(future variance)
  = ticker fixed effect
  + historical volatility features
  + optional controls
  + optional standardized text score
```

- Baseline과 +score는 각각 처음부터 다시 fitting한다.
- Forecast fitting에는 train+validation을 사용한다.
- Test는 평가에만 사용한다.
- MIDAS hyperparameter만 train fitting, validation QLIKE 선택 후 train+validation refit을 한다.

### 6.3 Baselines

| Model | 입력 |
|---|---|
| AR(5) | 최근 5개 일별 log variance lag |
| HAR | 과거 1일, 5일 평균, 22일 평균 log variance |
| HAR-X | HAR + absolute return, negative return, 20일 momentum, market HAR; 8-K는 log volume도 포함 |
| MIDAS | 30/50/80일 variance를 Beta decay weight로 요약; k와 theta는 validation QLIKE로 선택 |

각 baseline에 다음 두 variant를 별도로 붙인다.

```text
Baseline + one standardized BGE score
Baseline + one standardized Activation score
```

Direct LLM score는 최신 forecasting 표에는 넣지 않았다. 현재 direct baseline의 최신 역할은 document ranking 비교다.

### 6.4 왜 log variance를 예측하는가

Variance는 0 이상이고 오른쪽 꼬리가 매우 길다. Raw variance를 OLS로 직접 예측하면 큰 event가 loss를 지배하고 음수 예측도 가능하다. 따라서 다음처럼 log variance를 fitting하고 exp로 복원한다.

```text
model.fit(X, log(actual_variance))
predicted_variance = exp(model.predict(X))
```

### 6.5 평가 지표

QLIKE:

```text
QLIKE(y, yhat) = y / yhat - log(y / yhat) - 1
```

- y와 yhat에는 volatility가 아니라 variance를 넣는다.
- 낮을수록 좋다.
- 큰 volatility를 과소예측할 때 강하게 벌점한다.

추가 지표:

- Raw MSE
- Raw R2
- Log-MSE
- Log-R2
- Spearman(actual variance, predicted variance)
- Date-block bootstrap of per-observation QLIKE gain, 2,000 repetitions

---

## 7. 최신 News forecasting 결과

Test: 2023 News 12,478 ticker-days.

### QLIKE

| H | Model | Baseline | +BGE | +Activation |
|---|---|---:|---:|---:|
| 1 | AR(5) | 0.3705 | 0.3617 (-2.39%) | **0.3537 (-4.55%)** |
| 1 | HAR | 0.3750 | 0.3653 (-2.59%) | **0.3566 (-4.91%)** |
| 1 | HAR-X | 0.3674 | 0.3583 (-2.46%) | **0.3498 (-4.79%)** |
| 1 | MIDAS | 0.3784 | 0.3681 (-2.72%) | **0.3590 (-5.13%)** |
| 5 | AR(5) | 0.1528 | 0.1497 (-2.06%) | **0.1472 (-3.71%)** |
| 5 | HAR | 0.1510 | 0.1473 (-2.43%) | **0.1444 (-4.35%)** |
| 5 | HAR-X | 0.1451 | 0.1418 (-2.25%) | **0.1391 (-4.16%)** |
| 5 | MIDAS | 0.1515 | 0.1478 (-2.44%) | **0.1448 (-4.40%)** |

모든 BGE/Activation improvement는 date-block bootstrap one-sided p 약 0.0005였다.

### Raw MSE와 Raw R2

모든 8개 설정에서 activation이 BGE보다 Raw MSE를 더 줄이고 Raw R2를 더 높였다. 자세한 값은 다음 파일에 있다.

```text
outputs/news_ticker_centered_parkinson_forecast_tuned/forecast_results.csv
```

평균 QLIKE improvement:

- BGE: 2.42%
- Activation: 4.50%
- Activation 1일 평균: 4.84%
- Activation 5일 평균: 4.15%

---

## 8. 최신 8-K forecasting 결과

Test: 2025 event panel 453건.

### QLIKE

| H | Model | Baseline | +BGE | +Activation |
|---|---|---:|---:|---:|
| 1 | AR(5) | 0.6964 | 0.5616 (-19.36%) | **0.5179 (-25.64%)** |
| 1 | HAR | 0.7394 | 0.5504 (-25.57%) | **0.5152 (-30.32%)** |
| 1 | HAR-X | 0.6302 | 0.5154 (-18.22%) | **0.4865 (-22.80%)** |
| 1 | MIDAS | 0.7128 | 0.5605 (-21.36%) | **0.5188 (-27.22%)** |
| 5 | AR(5) | 0.2932 | 0.2631 (-10.29%) | **0.2623 (-10.53%)** |
| 5 | HAR | 0.3011 | **0.2599 (-13.70%)** | 0.2613 (-13.23%) |
| 5 | HAR-X | 0.2833 | **0.2579 (-8.95%)** | 0.2602 (-8.14%) |
| 5 | MIDAS | 0.3084 | 0.2771 (-10.15%) | **0.2751 (-10.79%)** |

- 1일 improvement는 모든 model에서 date-block bootstrap p<0.013이었다.
- 5일 point estimate는 8~14% 개선이지만 p>0.05였다.
- 8-K test는 145개 event date와 453건으로 News보다 작아 bootstrap uncertainty가 크다.

### Raw MSE / Raw R2 해석

- Raw MSE와 Raw R2에서는 대부분 BGE가 activation보다 좋았다.
- 예외적으로 1일 HAR-X는 BGE Raw MSE가 14.56% 악화된 반면 activation은 9.90% 개선했다.
- Activation은 tail-sensitive QLIKE에서 강하고, BGE는 squared-error 평균에서 강한 패턴이다.

자세한 값:

```text
outputs/sec8k_market_adjusted_parkinson_forecast/forecast_results.csv
```

---

## 9. BGE로 설명되지 않는 activation 정보

탐색 실험에서는 BGE 1,024차원, ticker dummy, event flags, text length로 activation score와 impact를 각각 예측하고 residual끼리 test correlation을 계산했다.

```text
activation residual vs impact residual Spearman = 0.0908
```

Date/ticker block bootstrap에서도 양의 결과였다. 또한 BGE가 이미 들어간 forecast에 activation을 추가하면 BGE-only보다 QLIKE가 더 개선됐다.

주의:

- 이 저장 결과는 최신 alpha-tuned score가 아니라 이전 fixed-alpha L20 score를 사용했다.
- 최신 L20 alpha=1 score로 다시 수행해야 논문 main robustness로 사용할 수 있다.
- BGE+Activation이 Activation-only보다 항상 좋지는 않았다. Activation이 BGE 정보를 상당 부분 포함할 가능성이 있다.

관련 파일:

```text
outputs/activation_incremental_over_embedding/
script/activation_incremental_over_embedding.py
```

---

## 10. 정성 및 robustness 실험

### Getty Images-Shutterstock 합병

2025-01-08 Getty Images와 Shutterstock의 약 37억 달러 합병 공시는 직관적인 high-impact 사례다.

- GETY 공시 중 axis score 1위
- GETY 공시 중 실제 impact 1위
- 첫 반응일 absolute abnormal return 17.65%
- 거래량: 과거 20일 중앙값의 20.2배

Token projection heatmap:

```text
outputs/gety_token_axis_heatmap/gety_token_heatmap_preview.png
outputs/gety_token_axis_heatmap/gety_token_heatmap.html
```

이는 axis가 전략적 구조 변화와 관련된 표현을 포착한다는 사례지만 token 인과성을 증명하지는 않는다.

### 다른 volatility estimator

Parkinson 외에도 close-to-close, Garman-Klass, Rogers-Satchell을 계산했다. 초기 composite axis에서는 News와 8-K 대부분의 estimator/horizon/model 조합에서 score 추가가 개선 방향이었다. 현재 단순 normalized label의 최신 BGE-vs-Activation 비교는 Parkinson만 다시 완료했다.

### 다른 LLM

초기 composite protocol에서 Qwen2.5-7B, Qwen3.5-4B, Llama-3.1-8B를 비교했고 세 모델 모두 평균 QLIKE improvement가 양수였다. 그러나 Qwen3.5/Llama model cache와 대용량 activation 일부는 공간 확보 과정에서 삭제됐다. 현재 단순 normalized Parkinson protocol은 Qwen2.5만 완료했다.

### 미래 volatility를 axis label에서 제거한 ablation

초기 composite protocol에서 label의 volatility component를 제거하고 News는 abnormal return, 8-K는 abnormal return+volume으로 axis를 다시 학습해도 forecasting improvement가 남았다. 이는 forecast target이 label에 직접 들어 있어 생긴 기계적 결과만은 아니라는 보조 근거다. 다만 현재 normalized Parkinson protocol에서는 아직 같은 ablation을 하지 않았다.

---

## 11. 현재 주장 가능한 것과 불가능한 것

### 방어 가능한 주장

1. 실제 market impact로 감독한 Qwen 중간 activation direction은 시간 분리된 News test에서 direct rating과 BGE보다 impact 순위를 잘 맞춘다.
2. News activation score는 AR/HAR/HAR-X/MIDAS에 하나의 변수로 추가했을 때 BGE score보다 큰 forecasting improvement를 보였다.
3. 8-K에서도 activation과 embedding score 모두 기존 가격 model에 추가 정보를 준다.
4. Activation 결과는 단순 firm/market level을 사용하는 것보다 엄격한 normalized label에서도 양수다.

### 아직 강하게 말하면 안 되는 것

1. 이 axis가 무감독 familiar/unfamiliar axis라는 주장
2. Activation이 모든 domain과 모든 metric에서 embedding보다 우월하다는 주장
3. 하나의 universal News+8-K axis가 이미 검증됐다는 주장
4. Activation이 숫자, modality, event structure 중 무엇 때문에 우월한지 인과적으로 설명했다는 주장
5. Axis가 문서의 causal market impact를 측정한다는 주장

### Test reuse 문제

2023 News와 2025 8-K test는 연구 과정에서 여러 label과 layer 분석에 반복적으로 사용됐다. 최초 실험에서는 untouched였지만 현재는 더 이상 완전한 confirmatory holdout이 아니다.

논문 최종본에는 다음 중 하나가 필요하다.

- 새로운 시간 holdout
- Company-disjoint holdout
- Frozen protocol로 이후 기간을 한 번만 평가
- 최소한 현재 test를 exploratory로 명시하고 별도 confirmatory replication 제공

### Knowledge cutoff

- News 2023은 Qwen2.5 release 이전이므로 pretraining memorization 가능성을 완전히 배제할 수 없다.
- 8-K 2025는 사용한 Qwen2.5 snapshot release 이후 사건이므로 이 우려가 상대적으로 작다.
- 모델이 이미 정답 가격을 알고 있어서가 아니라 text representation이 어떤지를 주장하려면 post-release holdout이 더 강하다.

### 최우선 TODO

1. **Cutoff-safe replication:** 현재 2023 News 결과는 모델 공개 이전 사건이므로, 모델이 뉴스와 사후 시장반응을 pretraining에서 접했을 가능성이 핵심 confound다. 학습 시점 이후 문서로 새 holdout을 구성하거나 cutoff가 명확히 더 이른 모델을 사용하고, protocol을 validation에서 고정한 뒤 test를 한 번만 평가한다.
2. **Embedding 대비 차별성:** 동일 label/split/tuning에서 전용 embedding, 같은 LLM의 final-layer representation, 중간 activation과 Direct LLM을 비교한다. 성능뿐 아니라 embedding으로 설명되지 않는 activation residual과 domain-transfer 이점을 검증한다.
3. **Mechanistic insight:** 숫자 규모, 불확실성, 사건 구조를 통제한 counterfactual 및 token/sentence attribution으로 axis가 실제 무엇에 반응하는지 밝힌다.
4. **Recall/familiarity axis 재설계:** 현재 supervised market-impact axis와 별도로, 모델이 recall 가능한 기존 기업 사실과 cutoff 이후 또는 기존 지식과 충돌하는 새로운 사실을 contrast set으로 만들어 familiarity/knowledge-conflict axis를 추출한다. 그 축과 market-impact axis의 관계 및 downstream 중요도 선별 성능을 평가한다.

이 네 항목 중 cutoff-safe replication이 가장 먼저 통과해야 할 조건이다. 현재 2023 News 결과는 이 검증 전까지 exploratory evidence로 취급한다.

---

## 12. 현재 디렉터리 구조

```text
/home/hob/workspace/familiar/
├── plan.md                         # 발표용 overview; 일부 수치는 이전 composite protocol
├── handoff.md                      # 이 문서
├── script/                         # 모든 실험 Python script
├── joint_impact_axis/              # joint News+8-K pipeline
├── docs/                           # 이전 버전별 상세 문서와 회의록
├── outputs/                        # 데이터, activation, score, 결과
├── research_tracks/                # Jacobian-lens 관련 별도 연구 tracks
├── shared_cache/                   # 일부 model/cache; Qwen3.5/Llama는 삭제된 상태
└── shared_env/                     # 일부 별도 Python env
```

주의: 루트의 `.git` directory는 비어 있어 현재 폴더는 유효한 git repository가 아니다. GitHub의 `market_axis` 저장소에는 전체 scripts와 artifacts가 있다고 가정하면 안 된다. 다른 서버로 옮길 때는 아래 rsync/tar 절차를 사용한다.

---

## 13. 현재 실행 환경

현재 확인한 환경:

```text
Python              3.10.19
torch               2.9.1+cu128
CUDA runtime        12.8
transformers        4.52.4
accelerate          1.7.0
datasets            5.0.0
huggingface_hub     0.32.3
numpy               1.26.4
pandas              2.3.3
scipy               1.15.3
scikit-learn        1.7.2
pyarrow             22.0.0
matplotlib          3.10.8
yfinance            1.0
seaborn             0.13.2
tokenizers          0.21.1
safetensors         0.5.3
sentencepiece       0.2.1
```

현재 GPU:

```text
4 x NVIDIA RTX A5000 24GB
```

Qwen2.5-7B model snapshot은 현재 다음 경로에 있다.

```text
/home/hob/workspace/financial_axis/.cache/huggingface/hub/
models--Qwen--Qwen2.5-7B-Instruct/snapshots/
a09a35458c702b33eeacc393d103063234e8bc28
```

Model cache 용량은 약 15GB다.

---

## 14. 새 서버 설치

### 14.1 권장 하드웨어

CPU-only replay:

- RAM 32GB 이상, 64GB 권장
- Free disk 20GB 이상
- GPU 불필요

Activation을 처음부터 재추출:

- 24GB VRAM GPU 1개 이상
- 두 shard 병렬 실행 시 24GB GPU 2개
- RAM 64GB 권장
- Model+activation+temporary shard를 고려해 free disk 50~60GB 이상 권장

### 14.2 Conda 환경

```bash
conda create -n market-axis python=3.10.19 -y
conda activate market-axis

# CUDA에 맞는 PyTorch를 설치한다. 아래는 현재 환경을 맞추는 예시다.
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

pip install \
  transformers==4.52.4 \
  accelerate==1.7.0 \
  datasets==5.0.0 \
  huggingface_hub==0.32.3 \
  numpy==1.26.4 \
  pandas==2.3.3 \
  scipy==1.15.3 \
  scikit-learn==1.7.2 \
  pyarrow==22.0.0 \
  matplotlib==3.10.8 \
  yfinance==1.0 \
  seaborn==0.13.2 \
  tqdm==4.68.3 \
  safetensors==0.5.3 \
  sentencepiece==0.2.1 \
  tokenizers==0.21.1
```

CUDA driver가 CUDA 12.8 wheel과 맞지 않으면 서버에 맞는 PyTorch wheel을 사용하되, 나머지 package version은 우선 고정한다.

### 14.3 PYTHONPATH

여러 script가 `script/`의 sibling module을 직접 import하므로 반드시 설정한다.

```bash
export ROOT=/path/to/familiar
export PYTHONPATH="$ROOT/script:$ROOT"
cd "$ROOT"
```

### 14.4 Hard-coded model path 수정

다음 파일에 현재 서버 절대경로가 들어 있다.

```text
script/poc_familiar_axis.py
script/impact_ratio_representation_benchmark.py
joint_impact_axis/pipeline.py
```

가장 쉬운 방법은 새 서버에서도 Qwen snapshot을 같은 구조로 두는 것이다. 다른 경로를 쓰면 위 파일의 `MODEL_PATH`, `QWEN25`, `MODEL_SPECS['qwen25']['path']`를 수정한다.

모델 다운로드 예시:

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --local-dir /path/to/models/Qwen2.5-7B-Instruct-a09a354
```

---

## 15. 권장 재현 경로 A: 저장 artifact로 수치까지 재현

이 경로를 가장 권장한다. Hugging Face dataset과 yfinance 가격은 시간이 지나며 바뀔 수 있으므로, exact replay는 현재 prepared data와 activation을 복사해야 한다.

### 15.1 최소 이전 목록

현재 약 11GB다.

```text
script/
docs/
joint_impact_axis/
plan.md
handoff.md

outputs/fintexts_news_axis_e2e/news_layers1-28.npy
outputs/fintexts_news_axis_e2e/hf_cache/
outputs/fintexts_news_axis_calendar_corrected/
outputs/layer_axis_robustness/qwen25_current_news/

outputs/joint_impact_axis_v1/filing_documents.parquet
outputs/joint_impact_axis_v1/activations/qwen25/filing_layers.npy

outputs/sec8k_eventtime_impact/prices_with_volume.parquet
outputs/sec8k_harx_activation_forecast/daily_panel.parquet
outputs/ohlc_volatility_forecast_comparison/sec8k_adjusted_open.parquet

outputs/impact_ratio_representation_benchmark/
outputs/news_ticker_centered_parkinson_axis_tuned/
outputs/news_ticker_centered_parkinson_forecast_tuned/
outputs/sec8k_simple_label_family_benchmark/
outputs/sec8k_market_adjusted_parkinson_forecast/
outputs/activation_incremental_over_embedding/
```

### 15.2 rsync 예시

현재 서버에서 실행한다.

```bash
cd /home/hob/workspace/familiar

rsync -a --info=progress2 --relative \
  script docs joint_impact_axis plan.md handoff.md \
  outputs/fintexts_news_axis_e2e/news_layers1-28.npy \
  outputs/fintexts_news_axis_e2e/hf_cache \
  outputs/fintexts_news_axis_calendar_corrected \
  outputs/layer_axis_robustness/qwen25_current_news \
  outputs/joint_impact_axis_v1/filing_documents.parquet \
  outputs/joint_impact_axis_v1/activations/qwen25/filing_layers.npy \
  outputs/sec8k_eventtime_impact/prices_with_volume.parquet \
  outputs/sec8k_harx_activation_forecast/daily_panel.parquet \
  outputs/ohlc_volatility_forecast_comparison/sec8k_adjusted_open.parquet \
  outputs/impact_ratio_representation_benchmark \
  outputs/news_ticker_centered_parkinson_axis_tuned \
  outputs/news_ticker_centered_parkinson_forecast_tuned \
  outputs/sec8k_simple_label_family_benchmark \
  outputs/sec8k_market_adjusted_parkinson_forecast \
  outputs/activation_incremental_over_embedding \
  USER@NEW_SERVER:/path/to/familiar/
```

### 15.3 핵심 artifact checksum

```text
45c4011c979c8ee1407071f6813e4dae1a53ab12b1f747445561ba00cd9e901a
  outputs/fintexts_news_axis_e2e/news_layers1-28.npy

f41730f4b38da1fd1053c3b9b126142bf723895e996b4e1b00dfea3dd9d8b5ae
  outputs/joint_impact_axis_v1/activations/qwen25/filing_layers.npy

526b1cb30bc297eb0a14a4c359c12fc19e5b80971e28131f2ca436cc8b26e79a
  outputs/impact_ratio_representation_benchmark/embedding_vectors.npy

3a59cfda5eb8d7fa674a8c8aee1dc43a1c89d40b4d99945b3faa10a23ec68ea8
  outputs/fintexts_news_axis_calendar_corrected/news_records.parquet

51ac02f48e3d4dc4a27bf2091680ff4b1debe4ba8ae597ebd6eeea9797bc6258
  outputs/joint_impact_axis_v1/filing_documents.parquet

4267f5d5ae934739b9050597fe6ab7cbbcc290ca13e9db35d1115930a98fdba8
  outputs/sec8k_eventtime_impact/prices_with_volume.parquet

23bd34b34c90ace4aec2e9b0d5e2242678b1d4545fab1455a04552e41127f4f0
  outputs/ohlc_volatility_forecast_comparison/sec8k_adjusted_open.parquet
```

검증:

```bash
sha256sum \
  outputs/fintexts_news_axis_e2e/news_layers1-28.npy \
  outputs/joint_impact_axis_v1/activations/qwen25/filing_layers.npy \
  outputs/impact_ratio_representation_benchmark/embedding_vectors.npy
```

### 15.4 Array shape 검증

```bash
python - <<'PY'
import numpy as np

paths = [
    "outputs/fintexts_news_axis_e2e/news_layers1-28.npy",
    "outputs/joint_impact_axis_v1/activations/qwen25/filing_layers.npy",
    "outputs/impact_ratio_representation_benchmark/embedding_vectors.npy",
]
for path in paths:
    x = np.load(path, mmap_mode="r")
    print(path, x.shape, x.dtype)
PY
```

기대값:

```text
news_layers1-28.npy    (51729, 28, 3584) float16
filing_layers.npy      (2417, 28, 3584) float16
embedding_vectors.npy  (52750, 1024) float16
```

### 15.5 CPU exact replay 명령

```bash
export ROOT=/path/to/familiar
export PYTHONPATH="$ROOT/script:$ROOT"
cd "$ROOT"

# News: validation에서 BGE alpha와 activation layer/alpha를 다시 선택하고 test score 계산
python script/news_ticker_centered_parkinson_axis_tuned.py

# News: AR/HAR/HAR-X/MIDAS baseline vs BGE vs Activation
python script/news_ticker_centered_parkinson_forecast_tuned.py

# 8-K: label family layer sweep와 document-level 비교
python script/sec8k_simple_label_family_benchmark.py

# 8-K: Parkinson event forecasting
python script/sec8k_market_adjusted_parkinson_forecast.py
```

이 네 명령에는 GPU가 필요하지 않다.

### 15.6 결과 sanity check

```bash
python - <<'PY'
import json
import pandas as pd

selection = json.load(open(
    "outputs/news_ticker_centered_parkinson_axis_tuned/selection.json"
))
assert selection["activation_selected_layer"] == 20
assert selection["activation_selected_alpha"] == 1.0
assert selection["bge_selected_alpha"] == 10.0

news = pd.read_csv(
    "outputs/news_ticker_centered_parkinson_axis_tuned/test_metrics.csv"
).set_index("method")
assert abs(news.loc["qwen25_activation", "test_spearman"] - 0.15000084) < 1e-5
assert abs(news.loc["bge_m3_embedding", "test_spearman"] - 0.09050058) < 1e-5

sec = json.load(open(
    "outputs/sec8k_market_adjusted_parkinson_forecast/metadata.json"
))
assert sec["activation_layer"] == 27
assert sec["panel_counts"] == {"train": 399, "val": 625, "test": 453}

print("MAIN REPLAY SANITY PASS")
PY
```

---

## 16. 재현 경로 B: Prepared text부터 GPU forward 재실행

원천 다운로드가 아니라 현재 prepared documents를 복사한 뒤 activation과 embedding만 새로 만드는 경로다. 데이터 revision 문제를 피하면서 model forward를 검증할 수 있다.

### 16.1 필요한 prepared input

```text
outputs/fintexts_news_axis_e2e/news_items.jsonl
outputs/fintexts_news_axis_calendar_corrected/news_records.parquet
outputs/joint_impact_axis_v1/filing_documents.parquet
outputs/joint_impact_axis_v1/filing_items.jsonl
outputs/sec8k_eventtime_impact/prices_with_volume.parquet
outputs/sec8k_harx_activation_forecast/daily_panel.parquet
outputs/ohlc_volatility_forecast_comparison/sec8k_adjusted_open.parquet
```

### 16.2 News activation 재추출

기존 512-token extraction을 정확히 재현한다.

```bash
export ROOT=/path/to/familiar
export PYTHONPATH="$ROOT/script:$ROOT"
cd "$ROOT"

CUDA_VISIBLE_DEVICES=0 python script/fintexts_news_axis_pipeline.py \
  --stage extract-shard --shard-id 0 --num-shards 2 \
  --batch-size 8 --max-length 512 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python script/fintexts_news_axis_pipeline.py \
  --stage extract-shard --shard-id 1 --num-shards 2 \
  --batch-size 8 --max-length 512 &
PID1=$!

wait "$PID0"
wait "$PID1"

python script/fintexts_news_axis_pipeline.py \
  --stage merge --num-shards 2
```

하나의 GPU만 있으면 shard 0과 1을 순차 실행한다. 각 process 내부에서는 보이는 GPU를 `cuda:0`으로 사용하므로 `CUDA_VISIBLE_DEVICES`로 physical GPU를 지정한다.

### 16.3 8-K activation 재추출

현재 normalized 8-K 결과는 joint pipeline에서 만든 2,048-token body-only activation을 사용한다.

```bash
CUDA_VISIBLE_DEVICES=0 python -m joint_impact_axis.pipeline \
  --stage extract-shard \
  --out-dir outputs/joint_impact_axis_v1 \
  --model qwen25 --domain filing \
  --shard-id 0 --num-shards 2 \
  --document-batch-size 1 --chunk-batch-size 1 \
  --max-length 2048 --overlap 256 --disk-floor-gib 20 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python -m joint_impact_axis.pipeline \
  --stage extract-shard \
  --out-dir outputs/joint_impact_axis_v1 \
  --model qwen25 --domain filing \
  --shard-id 1 --num-shards 2 \
  --document-batch-size 1 --chunk-batch-size 1 \
  --max-length 2048 --overlap 256 --disk-floor-gib 20 &
PID1=$!

wait "$PID0"
wait "$PID1"

python -m joint_impact_axis.pipeline \
  --stage merge \
  --out-dir outputs/joint_impact_axis_v1 \
  --model qwen25 --domain filing \
  --num-shards 2 --disk-floor-gib 20
```

### 16.4 BGE embedding 재추출

주의: BGE-M3 model revision이 manifest에 pin되어 있지 않다. 숫자까지 exact match가 필요하면 기존 `embedding_vectors.npy`를 복사한다.

```bash
CUDA_VISIBLE_DEVICES=0 python script/impact_ratio_representation_benchmark.py \
  extract-embedding --model BAAI/bge-m3 --batch-news 16 --batch-filing 4
```

### 16.5 Direct Qwen score 재추출

```bash
CUDA_VISIBLE_DEVICES=0 python script/impact_ratio_representation_benchmark.py \
  extract-direct --batch-news 8 --batch-filing 2
```

이 script는 정확히 하나의 CUDA device만 보여야 실행된다.

---

## 17. 재현 경로 C: 원천 데이터부터 다시 구축

이 경로는 구조적 재현에는 유용하지만 bitwise exact 재현은 보장하지 않는다.

### 17.1 FinTexTS 다운로드

`script/test_fintexts_axis_correlation.py`의 `load_fintexts`가 다음 네 parquet shard를 받는다.

```text
EXAONE-BI/FinTexTS
data/train-00000-of-00004.parquet
data/train-00001-of-00004.parquet
data/train-00002-of-00004.parquet
data/train-00003-of-00004.parquet
```

Cached revision은 `6a4ce04ac3f2e8a4ae15a5826e6980b50560eb29`다.

```bash
python script/fintexts_news_axis_pipeline.py \
  --stage prepare --out-dir outputs/fintexts_news_axis_e2e
```

그 후 activation extract/merge를 수행한다. 다만 최신 axis scripts는 calendar-corrected records와 row map을 기대하므로, exact replay에는 저장 artifact 경로 A가 훨씬 안전하다.

### 17.2 8-K 원천 준비

```bash
python script/sec8k_activation_market_research.py \
  --stage prepare \
  --dataset Disclosures-SSRC/8k_disclosure_dataset \
  --out-dir outputs/sec8k_activation_market \
  --min-company-docs 8
```

그 다음 SEC acceptance metadata와 가격 panel을 구축한다.

```bash
python script/sec8k_eventtime_impact_research.py
```

주의: 이 legacy script는 event panel 준비와 이전 activation 분석이 한 main 함수에 결합돼 있어 `sec8k_layers1-28.npy`와 이전 score artifact도 기대한다. 새 서버에서 raw부터 깔끔하게 재현하려면 data-preparation 단계와 analysis 단계를 분리하는 refactor가 필요하다. 그 전까지는 다음 prepared files를 복사하는 것이 canonical하다.

```text
outputs/sec8k_activation_market/records.parquet
outputs/sec8k_eventtime_impact/exact_event_panel.parquet
outputs/sec8k_eventtime_impact/prices_with_volume.parquet
outputs/joint_impact_axis_v1/filing_documents.parquet
```

### 17.3 yfinance revision 주의

8-K 가격은 yfinance의 auto-adjusted OHLC를 사용했다. yfinance는 과거 수정주가를 재작성할 수 있으므로 날짜가 같아도 미래 다운로드가 현재 parquet과 다를 수 있다. 논문용 최종 재현에는 현재 price parquet을 versioned data artifact로 보관해야 한다.

---

## 18. Disk와 process guard

현재 project는 약 23GB이며 큰 파일은 다음 두 activation array다.

```text
outputs/fintexts_news_axis_e2e/news_layers1-28.npy            약 9.7 GiB
outputs/joint_impact_axis_v1/activations/qwen25/news_layers.npy 약 9.7 GiB
```

두 파일은 shape과 내용이 같은 중복 activation이다. 최신 main은 첫 번째만 사용한다. Joint legacy pipeline까지 유지하려면 두 번째 대신 새 서버에서 symlink를 만들 수 있다.

```bash
ln -s \
  "$ROOT/outputs/fintexts_news_axis_e2e/news_layers1-28.npy" \
  "$ROOT/outputs/joint_impact_axis_v1/activations/qwen25/news_layers.npy"
```

GPU 작업 전에는 반드시 확인한다.

```bash
nvidia-smi
df -h "$ROOT"
```

현재 guard scripts:

```text
research_tracks/common/run_gpu_guarded.sh
research_tracks/common/run_guarded.sh
```

기존 GPU guard는 physical GPU 2/3만 허용하도록 hard-code되어 있다. 새 서버에서는 index 정책을 수정하거나 `CUDA_VISIBLE_DEVICES`를 직접 사용한다. 다른 사용자의 GPU process가 있으면 실행하지 않는다.

---

## 19. 핵심 파일 안내

### 최신 main

```text
script/simple_impact_label_benchmark.py
  News의 market-adjusted OHLC expansion label family 생성

script/news_ticker_centered_parkinson_axis_tuned.py
  News ticker-centered Parkinson label
  BGE alpha와 activation layer/alpha validation selection

script/news_ticker_centered_parkinson_forecast_tuned.py
  News baseline vs BGE vs Activation forecasting

script/sec8k_simple_label_family_benchmark.py
  8-K normalized OHLC label family와 layer sweep

script/sec8k_market_adjusted_parkinson_forecast.py
  8-K Parkinson baseline vs BGE vs Activation forecasting

script/impact_ratio_representation_benchmark.py
  Direct Qwen, BGE extraction과 최초 Post5/Pre20 benchmark

script/multimodel_volatility_forecast.py
  AR/HAR/HAR-X/MIDAS/LSTM 공통 구현

script/ohlc_volatility_forecast_comparison.py
  OHLC estimator와 panel feature 생성
```

### 이전 연구 흐름

```text
docs/volatility_axis_research_v3_v4_v7_v8.md
docs/fintexts_news_impact_axis_e2e_ko.md
docs/sec8k_impact_axis_e2e_ko.md
script/poc_familiar_axis.py
script/news_ticker_mismatch_axis_experiment.py
script/fintexts_prior_impact_experiment.py
script/sec8k_eventtime_impact_research.py
```

### Joint exploratory

```text
joint_impact_axis/README.md
joint_impact_axis/pipeline.py
joint_impact_axis/forecast.py
outputs/joint_impact_axis_v1/report.md
```

---

## 20. 별도 Jacobian-lens 연구 tracks

이 폴더에는 market-impact axis와 직접 연결되지 않는 세 개의 별도 연구 track도 있다.

### Track 1: Known-company prior

```text
research_tracks/track1_ticker_prior/final_report.md
```

결론:

- Qwen3.5-4B에 known-company/sector identity prior는 관찰됐다.
- Layer 23 full-state patch는 conflict loss를 약 21% 줄였다.
- 단일 target-free J-coordinate 개입은 test에서 2.3%만 줄여 강한 one-axis mechanism claim은 실패했다.
- Portfolio top-6 utility도 회복하지 못했다.

### Track 2: Demographic bias in financial advice

```text
research_tracks/track2_advice_bias/final_report.md
```

행동 POC에서는 동일한 금융 조건에서도 gender/race cue에 따라 소폭 allocation 차이가 나타났다. 저장된 final report는 causal validation pending 상태로 작성돼 있으므로 최신 결과 artifact와 보고서 상태를 다시 대조해야 한다.

### Track 3: Financial poisoning

```text
research_tracks/track3_finance_poisoning/
```

Threat model과 preregistered design까지만 있고 본 실험은 완료되지 않았다.

이 세 track은 현재 market-impact paper의 재현에 필요하지 않다.

---

## 21. 다음 연구자가 가장 먼저 할 일

우선순위 순서다.

1. 이 문서의 경로 A로 News/8-K 최신 결과를 새 서버에서 CPU replay한다.
2. News L20 alpha=1 score로 `activation_incremental_over_embedding.py`를 업데이트해 BGE conditional test를 다시 실행한다.
3. 8-K BGE와 activation 모두 alpha grid를 validation에서 선택하고 ticker-centering 유무를 사전 정의해 다시 비교한다.
4. 현재 여러 번 사용한 2023/2025 test 대신 새 confirmatory holdout을 만든다.
5. Direct Qwen score도 forecasting에 추가해 baseline, +Direct, +BGE, +Activation을 동일 표본에서 비교한다.
6. 새로운 normalized Parkinson label로 News-only, 8-K-only, domain-balanced joint axis를 다시 학습한다.
7. Qwen final-layer representation을 동일 dimension/동일 Ridge protocol로 비교해 중간 activation의 고유 이점을 검증한다.
8. 숫자 magnitude, certainty, M&A/event structure를 한 요소씩 바꾼 counterfactual로 어떤 내부 정보가 성능 차이를 만드는지 분석한다.
9. Budgeted document triage 또는 retrieval downstream을 추가해 forecasting 이외의 실용적 가치를 보여준다.

### 권장 논문 main table

Document ranking:

```text
Direct Qwen
BGE-M3 + tuned Ridge
Qwen final layer + tuned Ridge
Qwen validation-selected middle layer + tuned Ridge
```

Forecasting:

```text
AR/HAR/HAR-X/MIDAS baseline
same baseline + Direct score
same baseline + BGE score
same baseline + Activation score
same baseline + BGE + Activation
```

Holdout:

```text
time-disjoint + company-disjoint robustness
```

---

## 22. 최종 상태 체크리스트

```text
[x] News Qwen2.5 all-layer activation 추출
[x] 8-K Qwen2.5 2,048-token activation 추출
[x] Direct Qwen 1~9 score 추출
[x] BGE-M3 embedding 추출
[x] Simple market-adjusted Parkinson label 구현
[x] News BGE alpha / Activation layer+alpha validation tuning
[x] News 2023 ranking 및 AR/HAR/HAR-X/MIDAS forecast
[x] 8-K 2025 ranking 및 AR/HAR/HAR-X/MIDAS forecast
[x] 초기 composite joint Qwen2.5/Qwen3.5 실험
[ ] Normalized-label joint axis
[ ] 8-K Ridge alpha validation tuning
[ ] Latest tuned score의 BGE-conditional rerun
[ ] Fresh untouched confirmatory test
[ ] Direct score forecasting comparison
[ ] Mechanistic/counterfactual explanation
[ ] Forecasting 외 downstream use case
```

현재 코드와 artifact로 **기존 결과를 재현하는 것은 가능**하다. 다만 publishable confirmatory claim을 위해서는 위 미완료 항목, 특히 fresh holdout과 8-K tuning protocol 통일이 필요하다.
