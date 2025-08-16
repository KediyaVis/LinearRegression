# Regularization in Linear Regression

##  What Is Regularization?

Regularization is a technique used to **prevent overfitting** in machine learning models by adding a **penalty term** to the loss function.  
In the context of linear regression, regularization discourages overly complex models by shrinking the coefficient values (weights).

In simple linear regression, we typically minimize the **Residual Sum of Squares (RSS)**:

\[
\text{RSS} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

However, if we use too many features or the data has noise, our model might fit the training data too well, resulting in poor generalization to new data.

---

## ⚙ How Regularization Works

Regularization modifies the original loss function by **adding a penalty** on the magnitude of the coefficients. The new loss function becomes:

\[
\text{Loss} = \text{RSS} + \lambda \cdot \text{Penalty}
\]

Where:

- \(\lambda \ge 0\) is a hyperparameter controlling the strength of the penalty  
- A higher \(\lambda\) leads to **more shrinkage** of coefficients  
- \(\lambda = 0\) corresponds to **ordinary least squares** (no regularization)

---

## Two Most Common Types

### 1. **Ridge Regression (L2 Regularization)**

Adds a **squared penalty** on coefficients:

\[
\text{Loss} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} w_j^2
\]

**Characteristics**:
- Shrinks all coefficients toward zero but **never exactly to zero**
- Useful when many features have small/medium-sized effects
- Helps reduce model variance

---

### 2. **Lasso Regression (L1 Regularization)**

Adds an **absolute value penalty** on coefficients:

\[
\text{Loss} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |w_j|
\]

**Characteristics**:
- Can shrink some coefficients **exactly to zero**
- Performs **feature selection**
- Useful when only a subset of features is truly relevant

---

##  Elastic Net

Combines **L1 and L2** penalties:

\[
\text{Loss} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda_1 \sum_{j=1}^{p}|w_j| + \lambda_2 \sum_{j=1}^{p}w_j^2
\]

**Why use it?**
- Balances the benefits of Ridge and Lasso
- Performs well when features are correlated

---

## ⚖️ Choosing the Regularization Strength

The hyperparameter \(\lambda\) must be chosen carefully:
- Too small → no effect, model may overfit
- Too large → coefficients shrink excessively, model underfits

The most common approach is to use **cross-validation** to select the best value.

---

##  Summary

| Technique      | Penalty Type    | Shrinks Coefficients to Zero? | Performs Feature Selection |
|---------------|------------------|-------------------------------|-----------------------------|
| Ridge         | L2 (squared)     | No                            | No                          |
| Lasso         | L1 (absolute)    | Yes                           | Yes                         |
| Elastic Net   | L1 + L2          | Yes                           | Yes                         |

---

##  Example Use (Pseudo-Code)

```python
# Ridge
model = Ridge(alpha=0.5)
model.fit(X_train, y_train)

# Lasso
model = Lasso(alpha=0.1)
model.fit(X_train, y_train)

# Elastic Net
model = ElasticNet(alpha=0.2, l1_ratio=0.5)
model.fit(X_train, y_train)
