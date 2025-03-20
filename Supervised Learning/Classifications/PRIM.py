import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # For potential future use

# -----------------------------
# Create a synthetic dataset
# -----------------------------
# Generate 1000 random 2D points uniformly in [0,1]
N = 1000
X = torch.rand(N, 2)

# Define a "bump" region where the response is higher:
# For example, points with feature values between 0.4 and 0.6 have a higher response.
bump = ((X[:, 0] >= 0.4) & (X[:, 0] <= 0.6)) & ((X[:, 1] >= 0.4) & (X[:, 1] <= 0.6))
# Set response: baseline 0 for outside bump, 1 for inside bump, plus a little noise.
y = bump.float()
noise = 0.1 * torch.rand(N)
y = y + noise

# -----------------------------
# Define the PRIM algorithm
# -----------------------------
class PRIM:
    def __init__(self, peel_alpha=0.05, min_support=0.1, allow_negative_peel=True):
        """
        peel_alpha: fraction of the range to remove on each peel.
        min_support: minimum fraction of the total data that must remain in the box.
        allow_negative_peel: if True, allow moves that temporarily lower the mean response
                             (hoping that future moves might later improve it).
        """
        self.peel_alpha = peel_alpha
        self.min_support = min_support
        self.allow_negative_peel = allow_negative_peel
        self.history = []  # records each step's box and metrics
        self.final_box = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        support_threshold = self.min_support * n_samples

        # Initial box: full range for each feature
        lower = X.min(dim=0)[0]
        upper = X.max(dim=0)[0]
        current_box = {'lower': lower.clone(), 'upper': upper.clone()}

        # Initial evaluation of the box
        in_box = self._points_in_box(X, current_box)
        current_mean = y[in_box].mean().item()
        current_support = in_box.sum().item()
        self.history.append({
            'lower': lower.clone(),
            'upper': upper.clone(),
            'mean': current_mean,
            'support': current_support
        })

        iteration = 0
        while True:
            iteration += 1
            print(f"Iteration {iteration}: Current Mean = {current_mean:.4f}, Support = {current_support}")

            candidate_found = False  # flag if any candidate move is possible (meets support threshold)
            best_candidate_box = None
            best_candidate_mean = current_mean  # best candidate mean observed in this iteration
            best_candidate_support = current_support

            # Check candidate moves in all dimensions and both sides.
            for j in range(n_features):
                for side in ['lower', 'upper']:
                    new_box = {
                        'lower': current_box['lower'].clone(),
                        'upper': current_box['upper'].clone()
                    }
                    # Calculate the current range in the j-th dimension
                    range_j = current_box['upper'][j] - current_box['lower'][j]
                    # Peel a fraction from the chosen side.
                    if side == 'lower':
                        new_box['lower'][j] += self.peel_alpha * range_j
                    else:
                        new_box['upper'][j] -= self.peel_alpha * range_j

                    # Check if the new box still has enough support.
                    in_new_box = self._points_in_box(X, new_box)
                    support = in_new_box.sum().item()
                    if support < support_threshold:
                        continue  # skip this candidate if not enough points remain

                    candidate_found = True
                    mean_response = y[in_new_box].mean().item()

                    # Update candidate if we see an improvement.
                    if mean_response > best_candidate_mean:
                        best_candidate_mean = mean_response
                        best_candidate_box = new_box
                        best_candidate_support = support
                    # If no improvement and we're allowing negative peels,
                    # choose the candidate with the smallest drop.
                    elif self.allow_negative_peel and best_candidate_box is None:
                        best_candidate_mean = mean_response
                        best_candidate_box = new_box
                        best_candidate_support = support

            # If no candidate move is available at all, break the loop.
            if not candidate_found:
                print("No candidate moves possible in any dimension. Stopping.")
                break

            # If an improvement was found (or negative peel allowed), update the box.
            # When negative peeling is not allowed, we only update if the mean increases.
            if best_candidate_box is not None and (best_candidate_mean > current_mean or self.allow_negative_peel):
                current_box = best_candidate_box
                current_mean = best_candidate_mean
                current_support = best_candidate_support
                self.history.append({
                    'lower': current_box['lower'].clone(),
                    'upper': current_box['upper'].clone(),
                    'mean': current_mean,
                    'support': current_support
                })
            else:
                print("No candidate move improves the mean response. Terminating iterations.")
                break

        self.final_box = current_box

    def _points_in_box(self, X, box):
        """
        Returns a boolean mask indicating which points in X fall inside the given box.
        """
        return ((X >= box['lower']) & (X <= box['upper'])).all(dim=1)

# -----------------------------
# Run PRIM on the synthetic data
# -----------------------------
prim = PRIM(peel_alpha=0.05, min_support=0.1, allow_negative_peel=True)
prim.fit(X, y)

final_lower = prim.final_box['lower']
final_upper = prim.final_box['upper']
print("\nFinal Box Boundaries:")
print("Lower:", final_lower)
print("Upper:", final_upper)

# -----------------------------
# Plot both graphs in a grid layout
# -----------------------------
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Left subplot: Scatter plot with the final PRIM box
axs[0].scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=y.numpy(), cmap='viridis', alpha=0.5, label="Data Points")
rect = plt.Rectangle((final_lower[0].item(), final_lower[1].item()),
                     final_upper[0].item() - final_lower[0].item(),
                     final_upper[1].item() - final_lower[1].item(),
                     edgecolor='red', facecolor='none', linewidth=2, label="PRIM Box")
axs[0].add_patch(rect)
axs[0].set_xlabel("Feature 1")
axs[0].set_ylabel("Feature 2")
axs[0].set_title("PRIM Algorithm: Bump Hunting")
axs[0].legend()

# Right subplot: Evolution of mean response over iterations
iterations = range(len(prim.history))
means = [step['mean'] for step in prim.history]
axs[1].plot(iterations, means, marker='o')
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("Mean Response")
axs[1].set_title("Evolution of Mean Response in the Box")

plt.tight_layout()
plt.show()
