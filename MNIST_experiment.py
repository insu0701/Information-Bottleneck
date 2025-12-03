# ib_mnist_experiment.py
#
# Information Bottleneck-style experiment on MNIST:
# - Small MLP with bottleneck layer T
# - Approximate I(X;T) and I(T;Y) via discretization
# - Plot information plane trajectory over training
# - Sweep bottleneck width and show compression vs accuracy

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# -----------------------
# 1. Data: MNIST
# -----------------------

def make_data_mnist(n_train=None, n_test=None):
    """
    Load MNIST, flatten images to 784-d vectors.
    Optionally subsample n_train / n_test for speed.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),                    # (1,28,28), values in [0,1]
        transforms.Lambda(lambda x: x.view(-1))   # flatten to (784,)
    ])

    train_set = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Optional subsampling for speed
    if n_train is None or n_train > len(train_set):
        n_train = len(train_set)
    if n_test is None or n_test > len(test_set):
        n_test = len(test_set)

    idx_train = torch.randperm(len(train_set))[:n_train]
    idx_test = torch.randperm(len(test_set))[:n_test]

    X_train = torch.stack([train_set[i][0] for i in idx_train])  # (n_train, 784)
    y_train = torch.tensor([train_set[i][1] for i in idx_train], dtype=torch.long)
    X_test = torch.stack([test_set[i][0] for i in idx_test])    # (n_test, 784)
    y_test = torch.tensor([test_set[i][1] for i in idx_test], dtype=torch.long)

    return X_train, y_train, X_test, y_test


# -----------------------
# 2. Network with bottleneck T
# -----------------------

class Net(nn.Module):
    def __init__(self, bottleneck_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, bottleneck_dim)
        self.fc3 = nn.Linear(bottleneck_dim, 10)  # 10 digit classes

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

def entropy_from_counts(counts):
    counts = counts.astype(np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return -np.sum(p * np.log2(p))


def discretize_T_per_dim(T, n_bins=20):
    """
    Discretize each dimension of T separately into n_bins.
    T: numpy array shape (N, d)
    Returns: bin_idx of shape (N, d), each entry in {0,...,n_bins-1}
    """
    T = np.asarray(T)
    N, d = T.shape
    # normalize each dimension to [0,1]
    T_min = T.min(axis=0, keepdims=True)
    T_max = T.max(axis=0, keepdims=True)
    rng = (T_max - T_min) + 1e-9
    T_norm = (T - T_min) / rng
    T_norm = np.clip(T_norm, 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(T_norm, bins) - 1  # 0..n_bins-1
    idx = np.clip(idx, 0, n_bins - 1)
    return idx  # (N, d)


def entropy_T_factorized(bin_idx, n_bins):
    """
    Approximate H(T) by assuming independent dimensions:
    H(T) ≈ sum_j H(T_j).
    bin_idx: (N, d) integer matrix, each in 0..n_bins-1.
    """
    N, d = bin_idx.shape
    H_total = 0.0
    for j in range(d):
        states_j = bin_idx[:, j]
        counts = np.bincount(states_j, minlength=n_bins)
        H_total += entropy_from_counts(counts)
    return H_total


def mi_state_label_1d(states, labels, n_states):
    """
    Approximate I(T_j;Y) where T_j is 1D discrete (states), Y is discrete label.
    states: (N,), ints in 0..n_states-1
    labels: (N,), ints in 0..num_classes-1
    """
    states = np.asarray(states, dtype=int)
    labels = np.asarray(labels, dtype=int)
    num_states = n_states
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
            if pxy[i, j] > 0 and px[i, 0] > 0 and py[0, j] > 0:
                mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i, 0] * py[0, j]))
    return mi


def mi_TY_factorized(bin_idx, labels, n_bins):
    """
    Approximate I(T;Y) as sum_j I(T_j;Y).
    bin_idx: (N, d), labels: (N,)
    """
    N, d = bin_idx.shape
    labels = np.asarray(labels, dtype=int)
    I_total = 0.0
    for j in range(d):
        states_j = bin_idx[:, j]
        I_total += mi_state_label_1d(states_j, labels, n_bins)
    return I_total


# -----------------------
# 4. Training + info-plane
# -----------------------

def run_experiment(X_train, y_train, X_test, y_test,
                   bottleneck_dim=2, epochs=30, n_bins=15, lr=1e-3):
    """
    Train a single network with given bottleneck_dim and track:
    - approx I(X;T) ≈ sum_j H(T_j)
    - approx I(T;Y) ≈ sum_j I(T_j;Y)
    over epochs.
    """

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

            T_np = T.cpu().numpy()
            y_np = y_train.cpu().numpy()

            bin_idx = discretize_T_per_dim(T_np, n_bins=n_bins)
            H_T = entropy_T_factorized(bin_idx, n_bins)      # ≈ I(X;T)
            I_TY = mi_TY_factorized(bin_idx, y_np, n_bins)   # ≈ I(T;Y)

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
            head_width=0.05,
            alpha=0.4,
        )
    plt.xlabel("≈ I(X;T)  (sum of entropies of T_j)")
    plt.ylabel("≈ I(T;Y)  (sum of I(T_j;Y))")
    plt.title("Information plane trajectory (MNIST)")
    plt.colorbar(sc, label="epoch")
    plt.tight_layout()

    # Accuracy curve
    plt.figure(figsize=(5, 3))
    plt.plot(epochs, acc, "-o")
    plt.xlabel("Epoch")
    plt.ylabel("Train accuracy")
    plt.title("Learning curve (MNIST)")
    plt.tight_layout()

    plt.show()


def sweep_bottleneck_dims(X_train, y_train, X_test, y_test,
                          widths=(10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, 700, 800), epochs=30, n_bins=20, lr=1e-3):
    """
    Train separate networks with different bottleneck widths,
    measure final compression and accuracy.
    """

    results = []

    for w in widths:
        print(f"\n=== Bottleneck dim = {w} ===")
        net = Net(bottleneck_dim=w)
        opt = optim.Adam(net.parameters(), lr=lr)
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

            T_np = T.cpu().numpy()
            y_np = y_train.cpu().numpy()
            bin_idx = discretize_T_per_dim(T_np, n_bins=n_bins)
            H_T = entropy_T_factorized(bin_idx, n_bins)
            I_TY = mi_TY_factorized(bin_idx, y_np, n_bins)

        results.append(
            dict(
                width=w,
                IXT=H_T,
                ITY=I_TY,
                train_acc=train_acc,
                test_acc=test_acc,
            )
        )

        print(
            f" width={w}, train_acc={train_acc:.3f}, "
            f"test_acc={test_acc:.3f}, I(X;T)~{H_T:.3f}, I(T;Y)~{I_TY:.3f}"
        )

    return results


def plot_width_sweep(results):
    widths = [r["width"] for r in results]
    IXT = [r["IXT"] for r in results]
    ITY = [r["ITY"] for r in results]

    IXT_per_unit = [ix / w for ix, w in zip(IXT, widths)]
    IXY_per_unit = [iy / w for iy, w in zip(ITY, widths)]

    train_acc = [r["train_acc"] for r in results]
    test_acc = [r["test_acc"] for r in results]

    plt.figure(figsize=(10, 3))

    plt.subplot(1, 3, 1)
    plt.plot(widths, IXT_per_unit, "o-")
    plt.xlabel("Bottleneck dim")
    plt.ylabel("≈ I(X;T)")
    plt.title("Compression vs width")

    plt.subplot(1, 3, 2)
    plt.plot(widths, IXY_per_unit, "o-")
    plt.xlabel("Bottleneck dim")
    plt.ylabel("≈ I(T;Y)")
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


# -----------------------
# 5. Main
# -----------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # Load data (you can lower n_train/n_test if it's too slow)
    X_train, y_train, X_test, y_test = make_data_mnist(
        n_train=20000,  # e.g. 20k train samples
        n_test=5000     # e.g. 5k test samples
    )

    # 1) Info-plane trajectory for a single bottleneck
    IXT, ITY, acc = run_experiment(
        X_train, y_train, X_test, y_test,
        bottleneck_dim=2,
        epochs=30,
        n_bins=15,
        lr=1e-3,
    )
    plot_results(IXT, ITY, acc)

    # 2) Sweep bottleneck sizes
    results = sweep_bottleneck_dims(
        X_train, y_train, X_test, y_test,
        widths=(10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, 700, 800),
        epochs=30,
        n_bins=20,
        lr=1e-3,
    )
    plot_width_sweep(results)
