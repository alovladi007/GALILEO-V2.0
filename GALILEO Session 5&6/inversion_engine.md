# Geophysical Inversion Engine Documentation

## Overview

The geophysical inversion engine provides a comprehensive suite of tools for solving inverse problems in geophysics. It implements state-of-the-art algorithms for parameter estimation, regularization, and uncertainty quantification.

## Table of Contents

1. [Solvers](#solvers)
2. [Regularizers](#regularizers)
3. [Uncertainty Analysis](#uncertainty-analysis)
4. [Usage Examples](#usage-examples)
5. [API Reference](#api-reference)

---

## Solvers

### 1. Tikhonov Solver

**Linear regularized inversion** for problems of the form:

```
min ||Gm - d||² + λ²||Lm||²
```

where:
- `G` is the forward operator matrix
- `m` is the model vector
- `d` is the data vector
- `L` is the regularization matrix
- `λ` is the regularization parameter

**Key Features:**
- Efficient solution via normal equations
- L-curve analysis for parameter selection
- Resolution matrix computation
- Model uncertainty quantification

**Example:**
```python
from inversion import TikhonovSolver

# Create solver
solver = TikhonovSolver(forward_matrix=G, regularization_matrix=L)

# Solve inverse problem
result = solver.solve(data, lambda_reg=0.1, compute_resolution=True)

# Access results
model = result['model']
resolution = result['resolution_matrix']
```

### 2. Gauss-Newton Solver

**Nonlinear iterative solver** for problems:

```
min ||F(m) - d||² + λ²||L(m - m_ref)||²
```

**Key Features:**
- Iterative refinement via Newton steps
- Line search for robustness
- Convergence history tracking
- Flexible regularization

**Example:**
```python
from inversion import GaussNewtonSolver

def forward_model(m):
    # Your forward modeling function
    return computed_data

def jacobian(m):
    # Sensitivity matrix
    return J

solver = GaussNewtonSolver(forward_model, jacobian)
result = solver.solve(data, m_init, lambda_reg=0.5, max_iter=20)

print(f"Converged: {result['converged']}")
print(f"Final residual: {result['residual']}")
```

### 3. Bayesian MAP Solver

**Probabilistic inversion** with full uncertainty quantification:

```
Posterior: p(m|d) ∝ p(d|m) p(m)
```

Assumes Gaussian distributions:
- Prior: `m ~ N(m_prior, C_m)`
- Likelihood: `d|m ~ N(F(m), C_d)`

**Key Features:**
- Posterior mean (MAP estimate)
- Full posterior covariance matrix
- Marginal uncertainties
- Credible intervals

**Example:**
```python
from inversion import BayesianMAPSolver

solver = BayesianMAPSolver(forward_func, jacobian_func)

result = solver.solve(
    data=observed_data,
    m_prior=prior_mean,
    C_m=prior_covariance,
    C_d=data_covariance
)

# Posterior results
m_map = result['model_map']
uncertainties = result['uncertainties']

# Compute credible intervals
lower, upper = solver.credible_intervals(m_map, result['posterior_covariance'])
```

---

## Regularizers

### 1. Smoothness Regularizer

Enforces smoothness via finite differences:

```python
from inversion import SmoothnessRegularizer

# 1D smoothness
reg = SmoothnessRegularizer(n_model=100, order=2)  # Second derivative

# 2D smoothness for gridded models
L = reg.matrix_2d(nx=50, ny=50)
```

**Parameters:**
- `order`: 1 (first derivative) or 2 (second derivative)
- `alpha`: Anisotropy factor for 2D regularization

### 2. Total Variation (TV)

Edge-preserving regularization:

```
TV(m) = Σ √((∇m_i)² + ε²)
```

**Benefits:**
- Preserves sharp discontinuities
- Ideal for blocky structures
- Geologic realism

```python
from inversion import TotalVariationRegularizer

reg = TotalVariationRegularizer(n_model=100, epsilon=1e-6, beta=1.0)

# For 2D problems
tv_penalty = reg.penalty_2d(model, nx=50, ny=50)
gradient = reg.gradient_2d(model, nx=50, ny=50)
```

### 3. Sparsity (L1)

Promotes sparse solutions:

```python
from inversion import SparsityRegularizer

# Simple L1
reg = SparsityRegularizer(n_model=100)

# Weighted L1 (e.g., in wavelet domain)
reg = SparsityRegularizer(n_model=100, weights=wavelet_transform)
```

### 4. Geologic Prior

Incorporates known structures:

```python
from inversion import GeologicPriorRegularizer

# Define reference model and confidence
m_ref = reference_geology
weights = confidence_map  # Higher where more certain

reg = GeologicPriorRegularizer(n_model=100, m_ref=m_ref, weights=weights)
```

### 5. Cross-Gradient Coupling

For joint inversion of multiple physical properties:

```python
from inversion import CrossGradientRegularizer

# Enforce structural similarity between two models
reg = CrossGradientRegularizer(nx=50, ny=50)

penalty = reg.penalty(model1, model2)
grad1 = reg.gradient(model1, model2, which=1)
```

### 6. Minimum Support

Promotes compact anomalies:

```python
from inversion import MinimumSupportRegularizer

reg = MinimumSupportRegularizer(n_model=100)
penalty = reg.penalty(model)
```

---

## Uncertainty Analysis

### Resolution Matrix

Quantifies how well each model parameter is resolved:

```python
from inversion import UncertaintyAnalysis

# Compute metrics
metrics = UncertaintyAnalysis.compute_resolution_metrics(R)

print(f"Mean resolution: {metrics['mean_resolution']:.3f}")
print(f"Effective rank: {metrics['effective_rank']:.1f}")

# Visualize resolution
resolution_map = UncertaintyAnalysis.plot_resolution_map(R, grid_shape=(50, 50))
```

**Resolution Diagonal:**
- Values near 1: Well-resolved
- Values near 0: Poorly-resolved
- Measures linear combinations of true model

### Uncertainty Maps

Visualize parameter uncertainties:

```python
uncertainty_map = UncertaintyAnalysis.uncertainty_map(
    uncertainties, grid_shape=(50, 50)
)
```

---

## Usage Examples

### Example 1: Simple Linear Inversion

```python
import numpy as np
from inversion import TikhonovSolver

# Generate synthetic problem
n_data, n_model = 100, 50
G = np.random.randn(n_data, n_model)
m_true = np.zeros(n_model)
m_true[20:30] = 1.0  # Anomaly

# Data with noise
d = G @ m_true + 0.05 * np.random.randn(n_data)

# Solve with different regularization parameters
solver = TikhonovSolver(G)
lambdas = np.logspace(-3, 1, 30)
residuals, model_norms = solver.l_curve(d, lambdas)

# Choose optimal lambda from L-curve corner
# (In practice, use automated methods)
lambda_opt = lambdas[15]

# Final inversion
result = solver.solve(d, lambda_opt, compute_resolution=True)

print(f"Residual: {result['residual']:.4f}")
print(f"Mean resolution: {np.mean(result['resolution_diagonal']):.3f}")
```

### Example 2: Nonlinear Gravity Inversion

```python
from inversion import GaussNewtonSolver
import numpy as np

# Forward model: gravity from density
def forward_gravity(density):
    # Compute gravity field (simplified)
    G_const = 6.674e-11
    # ... implementation ...
    return gravity_field

def jacobian_gravity(density):
    # Sensitivity matrix
    # ... implementation ...
    return J

# Observed gravity data
gravity_obs = measured_data

# Initial guess
density_init = np.ones(n_cells) * 2700  # kg/m³

# Solve
solver = GaussNewtonSolver(forward_gravity, jacobian_gravity)
result = solver.solve(
    gravity_obs, 
    density_init,
    lambda_reg=10.0,
    max_iter=20,
    tol=1e-6
)

density_estimated = result['model']

# Plot convergence
import matplotlib.pyplot as plt
plt.semilogy(result['history']['objective'])
plt.xlabel('Iteration')
plt.ylabel('Objective Function')
plt.show()
```

### Example 3: Bayesian Inversion with Uncertainty

```python
from inversion import BayesianMAPSolver
import numpy as np

# Prior information
m_prior = np.zeros(n_model)  # Assume zero anomaly
C_m = 1.0 * np.eye(n_model)  # Prior variance

# Data covariance (measurement errors)
noise_std = 0.1
C_d = (noise_std**2) * np.eye(n_data)

# Solve
solver = BayesianMAPSolver(forward_func, jacobian_func)
result = solver.solve(data, m_prior, C_m, C_d)

# Results with uncertainties
m_map = result['model_map']
uncertainties = result['uncertainties']

# Plot with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(range(n_model), m_map, yerr=1.96*uncertainties, 
             fmt='o', capsize=3, label='95% CI')
plt.xlabel('Model Parameter')
plt.ylabel('Value')
plt.legend()
plt.show()
```

### Example 4: Joint Inversion with Cross-Gradient

```python
from inversion import GaussNewtonSolver, CrossGradientRegularizer
import numpy as np

# Two physical properties (e.g., seismic velocity and density)
nx, ny = 50, 50
n_model = nx * ny

# Cross-gradient coupling
cross_grad = CrossGradientRegularizer(nx, ny)

# Custom objective function
def joint_objective(m1, m2, d1, d2, lambda_cg):
    # Data misfit for property 1
    misfit1 = np.linalg.norm(F1(m1) - d1)**2
    
    # Data misfit for property 2
    misfit2 = np.linalg.norm(F2(m2) - d2)**2
    
    # Cross-gradient coupling
    coupling = cross_grad.penalty(m1, m2)
    
    return misfit1 + misfit2 + lambda_cg * coupling

# Alternating optimization or joint minimization
# ... implementation ...
```

---

## API Reference

### TikhonovSolver

```python
TikhonovSolver(forward_matrix, regularization_matrix=None)
```

**Methods:**
- `solve(data, lambda_reg, compute_resolution=True)` → dict
- `l_curve(data, lambdas)` → (residuals, model_norms)

**Returns:**
- `model`: Estimated model vector
- `residual`: Data fit (L2 norm)
- `resolution_matrix`: Model resolution (optional)
- `predicted_data`: Forward modeled data

### GaussNewtonSolver

```python
GaussNewtonSolver(forward_func, jacobian_func, regularization_matrix=None)
```

**Methods:**
- `solve(data, m_init, lambda_reg, max_iter=20, tol=1e-6, line_search=True)` → dict

**Returns:**
- `model`: Optimized model
- `residual`: Final data residual
- `iterations`: Number of iterations
- `converged`: Boolean convergence flag
- `history`: Convergence history

### BayesianMAPSolver

```python
BayesianMAPSolver(forward_func, jacobian_func)
```

**Methods:**
- `solve(data, m_prior, C_m, C_d, max_iter=20, tol=1e-6)` → dict
- `credible_intervals(model_map, C_post, confidence=0.95)` → (lower, upper)

**Returns:**
- `model_map`: Maximum a posteriori estimate
- `posterior_covariance`: Posterior covariance matrix
- `uncertainties`: Marginal standard deviations
- `resolution_matrix`: Model resolution

### UncertaintyAnalysis

**Static Methods:**
- `compute_resolution_metrics(R)` → dict
  - Returns: resolution_diagonal, mean_resolution, spread, eigenvalues, effective_rank
  
- `plot_resolution_map(R, grid_shape=None)` → ndarray
  - Returns: Resolution values (optionally reshaped)
  
- `uncertainty_map(uncertainties, grid_shape=None)` → ndarray
  - Returns: Uncertainty values (optionally reshaped)

---

## Performance Considerations

### Computational Complexity

| Solver | Time Complexity | Space Complexity |
|--------|----------------|------------------|
| Tikhonov | O(n²m + m³) | O(m²) |
| Gauss-Newton | O(k(n²m + m³)) | O(m²) |
| Bayesian MAP | O(k(n²m + m³)) | O(m²) |

where:
- n = number of data points
- m = number of model parameters
- k = number of iterations

### Recommendations

1. **Large-scale problems**: Use iterative solvers (LSQR, CGLS) instead of direct methods
2. **Sparse problems**: Leverage scipy.sparse matrices
3. **2D/3D grids**: Use structured regularizers
4. **Real-time applications**: Consider pre-computed sensitivity matrices

---

## Best Practices

### Choosing Regularization Parameters

1. **L-curve method**: Plot ||Gm-d|| vs ||Lm||, choose corner
2. **Cross-validation**: Minimize prediction error on held-out data
3. **Generalized Cross-Validation (GCV)**: Automated statistical approach
4. **Discrepancy principle**: Match residual to expected noise level

### Model Validation

1. **Synthetic tests**: Recover known models
2. **Resolution analysis**: Understand which features are resolvable
3. **Uncertainty quantification**: Report credible intervals
4. **Stability checks**: Test sensitivity to noise and parameters

### Common Pitfalls

- **Over-regularization**: Smooths out real features
- **Under-regularization**: Amplifies noise
- **Poor initialization**: May converge to local minimum (nonlinear)
- **Ignoring uncertainties**: Leads to overconfident interpretations

---

## References

1. Aster, R. C., Borchers, B., & Thurber, C. H. (2018). *Parameter Estimation and Inverse Problems*. Academic Press.

2. Tarantola, A. (2005). *Inverse Problem Theory and Methods for Model Parameter Estimation*. SIAM.

3. Hansen, P. C. (2010). *Discrete Inverse Problems: Insight and Algorithms*. SIAM.

4. Vogel, C. R. (2002). *Computational Methods for Inverse Problems*. SIAM.

---

## Support

For issues, questions, or contributions, please refer to the project repository.
