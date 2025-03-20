import torch
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. B-spline basis using recursion (Cox–de Boor)
# -------------------------------
def bspline_basis(x, knots, degree):
    """
    Compute B-spline basis functions of a given degree.
    x: tensor of shape (N,)
    knots: tensor of shape (num_knots,)
    degree: integer degree
    Returns a tensor of shape (N, n_basis) where n_basis = len(knots)-degree-1.
    """
    N = x.shape[0]
    n_basis = knots.shape[0] - degree - 1
    if degree == 0:
        B = torch.zeros(N, n_basis)
        for i in range(n_basis):
            # Use [knots[i], knots[i+1]) except include the last knot in the last basis.
            B[:, i] = ((x >= knots[i]) & (x < knots[i+1])).float()
        B[x == knots[-1], -1] = 1.0
        return B
    else:
        B_lower = bspline_basis(x, knots, degree - 1)
        B = torch.zeros(N, n_basis)
        for i in range(n_basis):
            # First term
            denom1 = knots[i + degree] - knots[i]
            term1 = ((x - knots[i]) / denom1) * B_lower[:, i] if denom1 > 0 else 0.0
            # Second term
            denom2 = knots[i + degree + 1] - knots[i + 1]
            term2 = ((knots[i + degree + 1] - x) / denom2) * B_lower[:, i + 1] if denom2 > 0 else 0.0
            B[:, i] = term1 + term2
        return B

# -------------------------------
# 2. M-spline basis: a scaled (nonnegative, normalized) version of B-splines.
# -------------------------------
def mspline_basis(x, knots, degree):
    """
    Compute M-spline basis functions by scaling the B-spline basis.
    M_i(x) = (degree+1) / (knots[i+degree+1]-knots[i]) * B_i(x)
    """
    B = bspline_basis(x, knots, degree)
    n_basis = B.shape[1]
    # Compute scaling factors for each basis function
    factors = (degree + 1) / (knots[degree + 1:] - knots[: -degree - 1])
    M = B * factors  # broadcasting along columns
    return M

# -------------------------------
# 3. Natural cubic spline basis (using a truncated power basis transformation)
# -------------------------------
def natural_cubic_basis(x, interior_knots, x_min, x_max):
    """
    Construct a natural cubic spline design matrix with a constant and linear term plus 
    transformed cubic terms for each interior knot.
    
    The transformation follows:
      h(x,k) = (x-k)_+^3 - ((x_max-k)/(x_max-x_min))*(x-x_min)_+^3 
               - ((k-x_min)/(x_max-x_min))*(x_max-x)_+^3.
    """
    ones = torch.ones_like(x)
    linear = x
    transformed = []
    for knot in interior_knots:
        term1 = torch.pow(torch.clamp(x - knot, min=0), 3)
        term2 = ((x_max - knot) / (x_max - x_min)) * torch.pow(torch.clamp(x - x_min, min=0), 3)
        term3 = ((knot - x_min) / (x_max - x_min)) * torch.pow(torch.clamp(x_max - x, min=0), 3)
        transformed.append(term1 - term2 - term3)
    # Stack into a design matrix: [1, x, h(x, k1), h(x, k2), ...]
    B = torch.stack([ones, linear] + transformed, dim=1)
    return B

# -------------------------------
# 4. Smoothing spline: fit a spline to noisy data with a penalty on curvature.
# Here we use a B-spline basis and penalize the second finite differences of the weights.
# -------------------------------
def smoothing_spline_fit(x_data, y_data, knots, degree, lam=1e-2, epochs=2000, lr=1e-2):
    """
    Fit a smoothing spline f(x) = B(x) @ w to data (x_data, y_data) using a penalty on
    second differences of the weights as a proxy for the second derivative penalty.
    """
    B = bspline_basis(x_data, knots, degree)
    N, n_basis = B.shape
    w = torch.nn.Parameter(torch.zeros(n_basis, 1))
    optimizer = torch.optim.Adam([w], lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = B @ w  # predicted values
        mse_loss = torch.mean((y_data.unsqueeze(1) - y_pred) ** 2)
        # Second difference penalty: approximate f''(x)
        diff = w[2:] - 2 * w[1:-1] + w[:-2]
        penalty = torch.sum(diff ** 2)
        loss = mse_loss + lam * penalty
        loss.backward()
        optimizer.step()
    return w.detach(), B

# -------------------------------
# Setup common x values and knots for demonstration.
# -------------------------------
# Evaluation points for plotting
x = torch.linspace(0, 1, steps=200)

# Degree for B-, M-, and smoothing splines (cubic spline => degree 3)
degree = 3
# Equally spaced knots for B-, M- and smoothing splines (number of knots = n_basis + degree + 1)
knots = torch.linspace(0, 1, steps=8)

# -------------------------------
# A. B-splines demonstration
# -------------------------------
B_basis = bspline_basis(x, knots, degree)
# For demonstration, choose increasing weights (one per basis function)
weights_B = torch.linspace(0, 1, steps=B_basis.shape[1]).unsqueeze(1)
y_B = B_basis @ weights_B

# -------------------------------
# B. M-splines demonstration
# -------------------------------
M_basis = mspline_basis(x, knots, degree)
# For demonstration, choose decreasing weights
weights_M = torch.linspace(1, 0, steps=M_basis.shape[1]).unsqueeze(1)
y_M = M_basis @ weights_M

# -------------------------------
# C. Natural cubic splines demonstration
# -------------------------------
# Choose two interior knots and set endpoints
interior_knots = torch.tensor([0.33, 0.66])
B_natural = natural_cubic_basis(x, interior_knots, 0, 1)
# For natural splines, we have 4 basis columns: constant, linear, and one for each interior knot.
weights_nat = torch.tensor([[0.5], [1.0], [0.8], [0.2]])
y_nat = B_natural @ weights_nat

# -------------------------------
# D. Smoothing spline demonstration
# -------------------------------
# Generate synthetic noisy data from a sine function.
x_data = torch.linspace(0, 1, steps=100)
y_true = torch.sin(2 * np.pi * x_data)
torch.manual_seed(0)
noise = 0.2 * torch.randn(x_data.shape)
y_data = y_true + noise
# Fit a smoothing spline (using B-spline basis) to the noisy data.
w_smooth, B_smooth = smoothing_spline_fit(x_data, y_data, knots, degree, lam=1e-2, epochs=2000, lr=1e-2)
# Evaluate the fitted smoothing spline on our common x grid.
B_smooth_full = bspline_basis(x, knots, degree)
y_smooth = B_smooth_full @ w_smooth

# -------------------------------
# Plotting all demonstrations in a 2x4 grid
# -------------------------------
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Column 0: B-splines
ax_top = axes[0, 0]
ax_bottom = axes[1, 0]
for i in range(B_basis.shape[1]):
    ax_top.plot(x.numpy(), B_basis[:, i].numpy(), label=f'B{i}')
ax_top.set_title("B-spline Basis Functions")
ax_top.legend(fontsize=8)
ax_bottom.plot(x.numpy(), y_B.numpy(), 'k-', label='Spline')
# Mark knots on the final fit
ax_bottom.vlines(knots.numpy(), ymin=0, ymax=weights_B.max().item(), colors='gray', linestyles='dotted')
# Annotate basis weights
for i, wt in enumerate(weights_B):
    ax_bottom.text(knots[i].item(), wt.item(), f'{wt.item():.2f}', color='blue')
ax_bottom.set_title("B-spline Spline Fit")

# Column 1: M-splines
ax_top = axes[0, 1]
ax_bottom = axes[1, 1]
for i in range(M_basis.shape[1]):
    ax_top.plot(x.numpy(), M_basis[:, i].numpy(), label=f'M{i}')
ax_top.set_title("M-spline Basis Functions")
ax_top.legend(fontsize=8)
ax_bottom.plot(x.numpy(), y_M.numpy(), 'k-', label='Spline')
ax_bottom.vlines(knots.numpy(), ymin=0, ymax=weights_M.max().item(), colors='gray', linestyles='dotted')
for i, wt in enumerate(weights_M):
    ax_bottom.text(knots[i].item(), wt.item(), f'{wt.item():.2f}', color='blue')
ax_bottom.set_title("M-spline Spline Fit")

# Column 2: Natural splines
ax_top = axes[0, 2]
ax_bottom = axes[1, 2]
for i in range(B_natural.shape[1]):
    ax_top.plot(x.numpy(), B_natural[:, i].numpy(), label=f'N{i}')
ax_top.set_title("Natural Spline Basis Functions")
ax_top.legend(fontsize=8)
ax_bottom.plot(x.numpy(), y_nat.numpy(), 'k-', label='Spline')
# Mark the interior knots
ax_bottom.vlines(interior_knots.numpy(), ymin=0, ymax=weights_nat.max().item(), colors='gray', linestyles='dotted')
# Annotate weights at approximate x positions (constant at x=0, linear at mid, others at the interior knots)
for i, wt in enumerate(weights_nat):
    if i == 0:
        pos = 0.0
    elif i == 1:
        pos = 0.5
    else:
        pos = interior_knots[i - 2].item()
    ax_bottom.text(pos, wt.item(), f'{wt.item():.2f}', color='blue')
ax_bottom.set_title("Natural Spline Fit")

# Column 3: Smoothing splines
ax_top = axes[0, 3]
ax_bottom = axes[1, 3]
# Plot the smoothing spline basis functions evaluated at the data used in fitting.
for i in range(B_smooth.shape[1]):
    ax_top.plot(x_data.numpy(), B_smooth.numpy()[:, i], label=f'S{i}')
ax_top.set_title("Smoothing Spline Basis Functions")
ax_top.legend(fontsize=8)
ax_bottom.plot(x.numpy(), y_smooth.numpy(), 'k-', label='Fitted Spline')
ax_bottom.scatter(x_data.numpy(), y_data.numpy(), color='red', s=10, label='Data')
ax_bottom.vlines(knots.numpy(), ymin=y_smooth.min().item(), ymax=y_smooth.max().item(), colors='gray', linestyles='dotted')
for i, wt in enumerate(w_smooth):
    ax_bottom.text(knots[i].item(), wt.item(), f'{wt.item():.2f}', color='blue')
ax_bottom.set_title("Smoothing Spline Fit")

plt.tight_layout()
plt.show()
