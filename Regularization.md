# Regularization in Linear Regression

##  What Is Regularization?

RRegularization is a technique used to **prevent overfitting** by adding a **penalty term** to the loss function.  
In linear regression, we normally minimize the **Residual Sum of Squares (RSS)**:

$$
\text{Loss} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} w_j^2
$$

Regularization modifies this by adding a penalty on the weight magnitudes:

$$
\text{Loss} = \text{RSS} + \lambda \cdot \text{Penalty}
$$

---
### 1. Ridge Regression (L2)

Adds the **squared magnitude** of the coefficients:

$$
\text{Loss} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} w_j^2
$$

- Shrinks weights continuously toward zero  
- Does **not** make any coefficient exactly zero  
- Helps when all features contribute a little

---

### 2. Lasso Regression (L1)

Adds the **absolute value** of the coefficients:

$$
\text{Loss} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |w_j|
$$

- Can shrink some coefficients **exactly** to zero  
- Performs **feature selection**  
- Useful when only a subset of features matter

---

### 3. Elastic Net

Combines both penalties:

$$
\text{Loss} =
\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 +
\lambda_1 \sum_{j=1}^{p}|w_j| +
\lambda_2 \sum_{j=1}^{p}w_j^2
$$

- Balances feature selection (L1) and coefficient shrinkage (L2)

---

## Hyperparameter λ (lambda)

- **λ = 0** → equivalent to ordinary least squares  
- **Small λ** → low penalty (risk of overfitting)  
- **Large λ** → high penalty (risk of underfitting)

Usually the best value of λ is selected by **cross-validation**.

---

## Summary Table

| Method       | Penalty | Features can be dropped? | Use Case                         |
|--------------|--------|---------------------------|----------------------------------|
| Ridge        | L2     | No                        | Many small effects               |
| Lasso        | L1     | Yes                       | Sparse / feature selection       |
| Elastic Net  | L1+L2  | Yes                       | Correlated or mixed type signals |

---

## Code Example (Python API)

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Ridge
ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)

# Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Elastic Net
enet = ElasticNet(alpha=0.2, l1_ratio=0.5)
enet.fit(X_train, y_train)
