
# Online Shopping Purchase Prediction with BART

This project applies **Bayesian Additive Regression Trees (BART)** and **Logistic Regression** to predict purchase intentions using online shopping session data.

It originated from my undergraduate exploration of Bayesian statistical modeling, where I studied BART and applied it to a real-world classification problem.

## 1. Dataset

**Online Shoppers Purchasing Intention Dataset**

- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- Observations: 12,330 online shopping sessions
- Features: 17 explanatory variables describing browsing behavior, visitor characteristics, and session information
- Target: `Revenue`, indicating whether a session resulted in a purchase

The dataset is imbalanced, with purchasing sessions representing approximately 15.5% of observations.

## 2. Methodology

### Exploratory Data Analysis

Explored relationships between browsing behavior, visitor characteristics, and purchase outcomes using visualizations, summary statistics, and statistical tests.

### Model Development

Two classification models were evaluated:

- **Bayesian Additive Regression Trees (BART):** A Bayesian ensemble model that combines regression trees and uses posterior inference for prediction.
- **Logistic Regression:** A conventional statistical classification model used as a baseline.

Both models were trained and evaluated using the same stratified 70/30 train-test split (`set.seed(2026)`).

For BART, posterior draws on the latent probit scale were transformed into purchase probabilities using the standard normal cumulative distribution function (`pnorm`).

For logistic regression, constant and linearly dependent dummy variables identified in the training data were excluded to address singularities.

Predictions were classified using a probability threshold of 0.5.

## 3. Results

Both models were evaluated on the same test set of 3,698 sessions, including 572 purchasing sessions.

| Metric | BART | Logistic Regression |
|---|---:|---:|
| Accuracy | **89.70%** | 88.45% |
| Precision (Purchase) | 71.27% | **77.99%** |
| Recall (Purchase) | **55.94%** | 35.31% |
| F1-score (Purchase) | **62.68%** | 48.62% |

### Interpretation

BART achieved higher accuracy, recall, and F1-score than logistic regression, while logistic regression achieved higher precision.

Among the 572 purchasing sessions in the test set, BART correctly identified 320, compared with 202 identified by logistic regression.

These results suggest that BART was more effective at identifying actual purchasers in this experiment, although it also produced more false positives.

Because purchasing sessions represent a minority of the dataset, recall and F1-score provide useful additional perspectives beyond accuracy alone.

## 4. Bayesian Predictive Uncertainty

Unlike a conventional point-estimate approach, BART provides posterior draws that can be used to examine uncertainty in predicted purchase probabilities.

The notebook transforms latent probit draws into probabilities and calculates 95% posterior credible intervals using the 2.5th and 97.5th percentiles.

These intervals describe uncertainty in the estimated purchase probabilities, not prediction intervals for observed binary outcomes.

Further analysis could examine how posterior uncertainty varies across individual shopping sessions.

## 5. Limitations

- Model performance was evaluated using a single train-test split. Repeated cross-validation would provide a more robust comparison.
- Both models used fixed configurations without systematic hyperparameter tuning.
- Rare categorical levels may lead to unstable logistic regression coefficient estimates.
- The classification threshold was fixed at 0.5. Different thresholds may change the precision-recall tradeoff.
- The analysis is observational and does not establish causal relationships between browsing behavior and purchasing decisions.

The results should therefore be interpreted as findings from this particular experimental configuration rather than evidence that BART universally outperforms logistic regression.

## 6. Reproducibility

### Requirements

- R
- Jupyter Notebook with an R kernel
- R packages:
  - `BayesTree`
  - `caret`
  - `readr`
  - `corrplot`
  - `ggplot2`
  - `dplyr`
  - `gridExtra`
  - `reshape2`
  - `car`

### Running the Analysis

1. Download the dataset from the [UCI repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).
2. Extract `online_shoppers_intention.csv`.
3. Place the CSV file in the notebook's working directory.
4. Open `BART_online_shoppers_intention_revised.ipynb` with an R kernel.
5. Execute the notebook cells sequentially.

The modeling section uses a fixed random seed to support reproducibility.

## 7. Project History

This project originated as an undergraduate exploratory analysis of BART and online purchasing behavior.

The evaluation methodology was subsequently revised to improve correctness and reproducibility.

Key corrections included:

- Using identical training and test observations for both models
- Converting BART's latent probit predictions into purchase probabilities
- Evaluating purchase (`Revenue = 1`) as the positive class
- Addressing redundant dummy variables in logistic regression
- Recalculating classification metrics using the corrected evaluation procedure

The original performance results were superseded by the corrected evaluation.

## Reference

Chipman, H. A., George, E. I., & McCulloch, R. E. (2010).

*BART: Bayesian Additive Regression Trees.*

The Annals of Applied Statistics, 4(1), 266–298.

https://doi.org/10.1214/09-AOAS285
