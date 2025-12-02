import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# -----------------------
# 1. Data
# -----------------------

def make_data():
    X, y = make_moons(n_samples=2000, noise=0.2, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).long()

    return (X_train_t, y_train_t, X_test_t, y_test_t)


# -----------------------
# 2. Network with bottleneck T
# -----------------------

class Net(nn.Module):
    def __init__(self, bottleneck_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, bottleneck_dim)
        self.fc3 = nn.Linear(bottleneck_dim, 2)  # binary labels

    def forward(self, x, return_T=False):
        h = torch.tanh(self.fc1(x))
        T = torch.tanh(self.fc2(h))
        out = self.fc3(T)
        if return_T:
            return out, T
        return out


# -----------------------
# 3. Info-theory helpers
# -----------------------

def discretize_T(T, n_bins=20):
    """
    Discretize a 2D bottleneck T into a single discrete state index per sample.
    T: numpy array shape (N, d).
    Returns: state indices shape (N,), values 0..num_states-1
    """
    T = np.asarray(T)
    # normalize each dimension to [0,1]
    T_min = T.min(axis=0, keepdims=True)
    T_max = T.max(axis=0, keepdims=True)
    rng = (T_max - T_min) + 1e-9
    T_norm = (T - T_min) / rng
    T_norm = np.clip(T_norm, 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)

    # bin each dimension
    idx = np.digitize(T_norm, bins) - 1  # 0..n_bins-1
    idx = np.clip(idx, 0, n_bins - 1)

    # flatten multi-d index to a single state
    d = T.shape[1]
    powers = (n_bins ** np.arange(d)).reshape(1, -1)  # shape (1, d)
    states = (idx * powers).sum(axis=1)
    return states.astype(int)


def entropy_from_counts(counts):
    counts = counts.astype(np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return -np.sum(p * np.log2(p))


def entropy_T(states):
    """H(T) in bits from discrete states."""
    counts = np.bincount(states)
    return entropy_from_counts(counts)


def mi_state_label(states, labels):
    """
    Approximate I(T;Y) where T is given by 'states' (discrete) and Y by labels.
    """
    states = np.asarray(states, dtype=int)
    labels = np.asarray(labels, dtype=int)

    num_states = states.max() + 1
    num_classes = labels.max() + 1

    joint = np.zeros((num_states, num_classes), dtype=np.float64)
    for s, c in zip(states, labels):
        joint[s, c] += 1.0

    total = joint.sum()
    if total == 0:
        return 0.0
    pxy = joint / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)

    mi = 0.0
    for i in range(num_states):
        for j in range(num_classes):
            if pxy[i, j] > 0:
                mi += pxy[i, j] * np.log2(
                    pxy[i, j] / (px[i, 0] * py[0, j])
                )
    return mi


# -----------------------
# 4. Training + info plane
# -----------------------

def run_experiment(
    bottleneck_dim=2, epochs=60, n_bins=20, lr=1e-2
):
    X_train, y_train, X_test, y_test = make_data()

    net = Net(bottleneck_dim=bottleneck_dim)
    opt = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    IXT_list = []
    ITY_list = []
    acc_list = []

    for epoch in range(epochs):
        net.train()
        opt.zero_grad()
        logits = net(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        opt.step()

        # evaluate on training set
        net.eval()
        with torch.no_grad():
            logits, T = net(X_train, return_T=True)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y_train).float().mean().item()

            T_np = T.numpy()
            y_np = y_train.numpy()

            states = discretize_T(T_np, n_bins=n_bins)
            H_T = entropy_T(states)        # ≈ I(X;T)
            I_TY = mi_state_label(states, y_np)

        IXT_list.append(H_T)
        ITY_list.append(I_TY)
        acc_list.append(acc)

        print(
            f"Epoch {epoch+1:02d}: "
            f"acc={acc:.3f}, I(X;T)~{H_T:.3f}, I(T;Y)~{I_TY:.3f}"
        )

    return np.array(IXT_list), np.array(ITY_list), np.array(acc_list)


def plot_results(IXT, ITY, acc):
    epochs = np.arange(1, len(IXT) + 1)

    # Info-plane trajectory
    plt.figure(figsize=(5, 4))
    sc = plt.scatter(IXT, ITY, c=epochs, cmap="viridis")
    for i in range(len(IXT) - 1):
        plt.arrow(
            IXT[i],
            ITY[i],
            IXT[i + 1] - IXT[i],
            ITY[i + 1] - ITY[i],
            length_includes_head=True,
            head_width=0.02,
            alpha=0.4,
        )
    plt.xlabel("≈ I(X;T)  (entropy of T)")
    plt.ylabel("I(T;Y)")
    plt.title("Information plane trajectory")
    plt.colorbar(sc, label="epoch")
    plt.tight_layout()

    # Accuracy curve
    plt.figure(figsize=(5, 3))
    plt.plot(epochs, acc, "-o")
    plt.xlabel("Epoch")
    plt.ylabel("Train accuracy")
    plt.title("Learning curve")
    plt.tight_layout()

    plt.show()


def sweep_bottleneck_dims(widths=(1, 2, 4, 6), epochs=60):
    X_train, y_train, X_test, y_test = make_data()
    results = []

    for w in widths:
        print(f"\n=== Bottleneck dim = {w} ===")
        net = Net(bottleneck_dim=w)
        opt = optim.Adam(net.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            net.train()
            opt.zero_grad()
            logits = net(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            opt.step()

        # after training: measure compression + accuracy
        net.eval()
        with torch.no_grad():
            logits, T = net(X_train, return_T=True)
            preds = torch.argmax(logits, dim=1)
            train_acc = (preds == y_train).float().mean().item()

            logits_test = net(X_test)
            preds_test = torch.argmax(logits_test, dim=1)
            test_acc = (preds_test == y_test).float().mean().item()

            T_np = T.numpy()
            y_np = y_train.numpy()
            states = discretize_T(T_np, n_bins=15)
            H_T = entropy_T(states)
            I_TY = mi_state_label(states, y_np)

        results.append(
            dict(
                width=w,
                IXT=H_T,
                ITY=I_TY,
                train_acc=train_acc,
                test_acc=test_acc,
            )
        )

    return results


def plot_width_sweep(results):
    widths = [r["width"] for r in results]
    IXT = [r["IXT"] for r in results]
    ITY = [r["ITY"] for r in results]
    train_acc = [r["train_acc"] for r in results]
    test_acc = [r["test_acc"] for r in results]

    plt.figure(figsize=(10, 3))

    plt.subplot(1, 3, 1)
    plt.plot(widths, IXT, "o-")
    plt.xlabel("Bottleneck dim")
    plt.ylabel("≈ I(X;T)")
    plt.title("Compression vs width")

    plt.subplot(1, 3, 2)
    plt.plot(widths, ITY, "o-")
    plt.xlabel("Bottleneck dim")
    plt.ylabel("I(T;Y)")
    plt.title("Predictive info vs width")

    plt.subplot(1, 3, 3)
    plt.plot(widths, train_acc, "o-", label="train")
    plt.plot(widths, test_acc, "o-", label="test")
    plt.xlabel("Bottleneck dim")
    plt.ylabel("Accuracy")
    plt.title("Generalization vs width")
    plt.legend()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # 1) Info-plane trajectory for a single bottleneck
    IXT, ITY, acc = run_experiment(
        bottleneck_dim=2, epochs=60, n_bins=15, lr=1e-2
    )
    plot_results(IXT, ITY, acc)

    # 2) Sweep bottleneck sizes
    results = sweep_bottleneck_dims(widths=(1, 2, 4, 6), epochs=60)
    plot_width_sweep(results)