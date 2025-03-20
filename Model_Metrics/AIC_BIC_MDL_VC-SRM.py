import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

def generate_data(n_samples=200):
    """
    Generate a 2D synthetic binary classification dataset.
    The two classes are separated by a linear decision boundary with added noise.
    """
    # Random 2D points
    X = torch.randn(n_samples, 2)
    # Create a linear decision boundary: class 1 if sum of features > 0, else class 0 (plus some noise)
    y = (X[:, 0] + X[:, 1] + 0.3 * torch.randn(n_samples) > 0).long()
    return X, y

def knn_predict(x_train, y_train, x_query, k, leave_one_out=False):
    """
    Given training data (x_train, y_train), predict labels for x_query using kNN.
    Returns both predicted labels and the probability distribution (as vote fractions) for each query.
    
    If leave_one_out is True, and x_query is from x_train,
    the distance from a point to itself is set to infinity so that it is not counted.
    """
    # Compute pairwise Euclidean distances
    distances = torch.cdist(x_query, x_train)
    
    if leave_one_out and x_query.size(0) == x_train.size(0):
        # For each training sample, set the self-distance to infinity
        diag_indices = torch.arange(x_train.size(0))
        distances[diag_indices, diag_indices] = float('inf')
    
    # Get the indices of the k smallest distances
    knn_indices = distances.topk(k, largest=False).indices  # shape: (n_query, k)
    
    # Determine predictions and vote probabilities
    n_query = x_query.size(0)
    # Assume binary classification (classes 0 and 1)
    probas = torch.zeros(n_query, 2)
    pred_labels = torch.zeros(n_query, dtype=torch.long)
    
    for i in range(n_query):
        neighbor_labels = y_train[knn_indices[i]]
        # Count votes for each class (0 and 1)
        counts = torch.tensor([torch.sum(neighbor_labels == 0).item(),
                               torch.sum(neighbor_labels == 1).item()], dtype=torch.float)
        # Compute probabilities (add a tiny constant for safety)
        prob = counts / (k + 1e-8)
        probas[i] = prob
        pred_labels[i] = torch.argmax(prob)
        
    return pred_labels, probas

def compute_metrics(x_train, y_train, x_test, y_test, k):
    """
    For a given k, compute training error, test error, training negative log likelihood,
    and simulated criteria: AIC, BIC, MDL, and a VC–SRM inspired bound.
    """
    n_train = x_train.size(0)
    
    # --- Training predictions (use leave-one-out to avoid self-vote) ---
    train_pred, train_probas = knn_predict(x_train, y_train, x_train, k, leave_one_out=True)
    train_error = (train_pred != y_train).float().mean().item()
    
    # Compute negative log likelihood (NLL) on training set
    # For each sample, pick the probability for the true class and compute -log(prob)
    true_probs = train_probas[torch.arange(n_train), y_train]
    # Clip probabilities to avoid log(0)
    true_probs = torch.clamp(true_probs, 1e-8, 1.0)
    train_nll = (-torch.log(true_probs)).mean().item()
    
    # --- Test predictions ---
    test_pred, _ = knn_predict(x_train, y_train, x_test, k, leave_one_out=False)
    test_error = (test_pred != y_test).float().mean().item()
    
    # --- Simulated criteria ---
    # We define an "effective complexity" as 1/k (so lower k means higher complexity)
    complexity = 1.0 / k
    # AIC: typically 2*(# parameters) - 2*log(L); here we simulate as:
    AIC = 2 * complexity + 2 * train_nll
    # BIC: log(n)*(# parameters) - 2*log(L)
    BIC = np.log(n_train) * complexity + 2 * train_nll
    # MDL: we simulate as train_nll plus a complexity penalty
    MDL = train_nll + complexity * np.log(n_train)
    
    # --- VC–SRM bound (simulated) ---
    # In VC theory, a bound can be of the form: training error + sqrt((complexity*log(n) + 1)/n)
    VC_bound = train_error + np.sqrt((complexity * np.log(n_train) + 1) / n_train)
    
    return train_error, test_error, train_nll, AIC, BIC, MDL, VC_bound

def main():
    # Settings
    num_iterations = 20  # number of random train/test splits (for boxplot variability)
    n_samples = 200
    train_ratio = 0.75
    k_list = [1, 3, 5, 7, 9, 11, 13, 15]
    
    # Containers for metrics (per k)
    metrics = {k: {"train_error": [], "test_error": [], "train_nll": [],
                   "AIC": [], "BIC": [], "MDL": [], "VC_bound": []} for k in k_list}
    
    for it in range(num_iterations):
        # Generate dataset and split into train/test
        X, y = generate_data(n_samples)
        # Shuffle indices
        indices = torch.randperm(n_samples)
        X, y = X[indices], y[indices]
        n_train = int(train_ratio * n_samples)
        x_train, y_train = X[:n_train], y[:n_train]
        x_test, y_test = X[n_train:], y[n_train:]
        
        for k in k_list:
            te, tse, tnll, aic, bic, mdl, vc_bound = compute_metrics(x_train, y_train, x_test, y_test, k)
            metrics[k]["train_error"].append(te)
            metrics[k]["test_error"].append(tse)
            metrics[k]["train_nll"].append(tnll)
            metrics[k]["AIC"].append(aic)
            metrics[k]["BIC"].append(bic)
            metrics[k]["MDL"].append(mdl)
            metrics[k]["VC_bound"].append(vc_bound)
    
    # Compute mean values over iterations for each k (for line plots)
    k_vals = np.array(k_list)
    mean_train_error = np.array([np.mean(metrics[k]["train_error"]) for k in k_list])
    mean_test_error = np.array([np.mean(metrics[k]["test_error"]) for k in k_list])
    mean_train_nll = np.array([np.mean(metrics[k]["train_nll"]) for k in k_list])
    mean_AIC = np.array([np.mean(metrics[k]["AIC"]) for k in k_list])
    mean_BIC = np.array([np.mean(metrics[k]["BIC"]) for k in k_list])
    mean_MDL = np.array([np.mean(metrics[k]["MDL"]) for k in k_list])
    mean_VC_bound = np.array([np.mean(metrics[k]["VC_bound"]) for k in k_list])
    
    # Create a grid of subplots (2 rows x 3 cols)
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("KNN Model Complexity and Model Selection Criteria vs. k", fontsize=16)
    
    # Subplot 1: Training and Test Error vs. k (with boxplot for test error)
    ax = axs[0, 0]
    ax.plot(k_vals, mean_train_error, marker='o', label='Train Error')
    ax.plot(k_vals, mean_test_error, marker='o', label='Test Error (mean)', color='orange')
    ax.set_xlabel("k")
    ax.set_ylabel("Error Rate")
    ax.set_title("Train & Test Error vs. k")
    ax.legend()
    # Add boxplots for test error at each k (using positions = k values)
    test_error_data = [metrics[k]["test_error"] for k in k_list]
    bp = ax.boxplot(test_error_data, positions=k_vals, widths=0.5, patch_artist=True,
                    boxprops=dict(facecolor='orange', alpha=0.3), showfliers=False)
    
    # Subplot 2: Training Negative Log Likelihood vs. k
    ax = axs[0, 1]
    ax.plot(k_vals, mean_train_nll, marker='s', color='green')
    ax.set_xlabel("k")
    ax.set_ylabel("Train NLL")
    ax.set_title("Training Negative Log Likelihood vs. k")
    
    # Subplot 3: VC–SRM Bound vs. k
    ax = axs[0, 2]
    ax.plot(k_vals, mean_VC_bound, marker='^', color='purple')
    ax.set_xlabel("k")
    ax.set_ylabel("VC Bound")
    ax.set_title("VC–SRM Bound vs. k")
    
    # Subplot 4: AIC vs. k
    ax = axs[1, 0]
    ax.plot(k_vals, mean_AIC, marker='o', color='red')
    ax.set_xlabel("k")
    ax.set_ylabel("AIC")
    ax.set_title("AIC vs. k")
    
    # Subplot 5: BIC vs. k
    ax = axs[1, 1]
    ax.plot(k_vals, mean_BIC, marker='o', color='brown')
    ax.set_xlabel("k")
    ax.set_ylabel("BIC")
    ax.set_title("BIC vs. k")
    
    # Subplot 6: MDL vs. k
    ax = axs[1, 2]
    ax.plot(k_vals, mean_MDL, marker='o', color='blue')
    ax.set_xlabel("k")
    ax.set_ylabel("MDL")
    ax.set_title("MDL vs. k")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
if __name__ == "__main__":
    main()
