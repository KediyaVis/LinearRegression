# Introduction to Linear Regression

## 1. What is Linear Regression?
Linear Regression is one of the most fundamental supervised machine learning algorithms used for predicting continuous outcomes.

It models the relationship between:
- Dependent variable (Target) — what we want to predict
- Independent variable(s) (Features) — inputs used to make the prediction

---

## 2. Real-World Examples

| Domain       | Example Use Case |
|--------------|------------------|
| Business     | Predict monthly sales from advertising spend |
| Finance      | Estimate house prices from location and size |
| HR           | Predict employee salaries from experience |
| Healthcare   | Estimate blood pressure from age and weight |

---

## 3. Types of Linear Regression

1. **Simple Linear Regression** — one predictor variable  
   Example: Predicting salary from years of experience.
   
2. **Multiple Linear Regression** — two or more predictors  
   Example: Predicting house price from size, bedrooms, and location.

---

## 4. Mathematical Formulation

### Simple Linear Regression
\[
y = \beta_0 + \beta_1 x + \varepsilon
\]

Where:
- \( y \) = Target variable (e.g., Salary)
- \( x \) = Predictor variable (e.g., Years of Experience)
- \( \beta_0 \) = Intercept (value of \(y\) when \(x=0\))
- \( \beta_1 \) = Slope (effect of one unit change in \(x\) on \(y\))
- \( \varepsilon \) = Error term

---

## 5. How Does Linear Regression Work?

It finds the best-fitting line through the data points by minimizing the **Sum of Squared Errors (SSE)** between predicted and actual values.

**Ordinary Least Squares (OLS)** is the most common method used.

---

## 6. Assumptions of Linear Regression

1. **Linearity** — Relationship between predictors and target is linear.  
2. **Independence** — Observations are independent of each other.  
3. **Homoscedasticity** — Constant variance of residuals.  
4. **Normality** — Residuals should be normally distributed.  
5. **No Multicollinearity** — Predictors are not highly correlated (for multiple regression).
