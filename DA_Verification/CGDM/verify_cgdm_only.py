import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from src.da_models import CGDM, DAModel, train_cgdm  # noqa: E402


# Ensure Ubicomp root is on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))



def generate_source_target_data(
    num_classes=8,
    num_dims=8,
    num_samples_source=1600,
    num_samples_target=1600,
    class_ratios_source=None,
    class_ratios_target=None,
    source_mean_scale=3.0,
    source_cov_base=1.0,
    target_mean_offset_range=4.0,
    target_cov_noise_scale=0.1,
    target_cov_scale=1.0,
    random_seed=20,
):
    np.random.seed(random_seed)

    if class_ratios_source is None:
        class_ratios_source = [1 / num_classes] * num_classes
    if class_ratios_target is None:
        class_ratios_target = [1 / num_classes] * num_classes

    source_samples_per_class = (np.array(class_ratios_source) * num_samples_source).astype(int)
    target_samples_per_class = (np.array(class_ratios_target) * num_samples_target).astype(int)

    def small_random_mean(dim, scale):
        return np.random.uniform(low=-scale, high=scale, size=(dim,))

    def small_random_cov(dim, base):
        a = 0.2 * np.random.randn(dim, dim)
        cov = 0.5 * (a + a.T) + base * np.eye(dim)
        return cov

    def generate_gaussian_data(mean, cov, n):
        return np.random.multivariate_normal(mean, cov, n)

    source_means = []
    source_covs = []
    for _ in range(num_classes):
        m = small_random_mean(num_dims, source_mean_scale)
        cov = small_random_cov(num_dims, source_cov_base)
        source_means.append(m)
        source_covs.append(cov)

    target_means = []
    target_covs = []
    for c in range(num_classes):
        base_mean = source_means[c].copy()
        offset = np.random.uniform(
            low=-target_mean_offset_range,
            high=target_mean_offset_range,
            size=(num_dims,),
        )
        target_mean = base_mean + offset
        target_means.append(target_mean)
        base_cov = source_covs[c].copy()
        a = target_cov_noise_scale * np.random.randn(num_dims, num_dims)
        noise_cov = 0.5 * (a + a.T)
        target_cov = target_cov_scale * base_cov + noise_cov
        target_covs.append(target_cov)

    x_source_list = []
    y_source_list = []
    for c in range(num_classes):
        n_samples = source_samples_per_class[c]
        x_c = generate_gaussian_data(source_means[c], source_covs[c], n_samples)
        y_c = np.full(n_samples, c)
        x_source_list.append(x_c)
        y_source_list.append(y_c)
    x_source = np.vstack(x_source_list)
    y_source = np.concatenate(y_source_list)

    x_target_list = []
    y_target_list = []
    for c in range(num_classes):
        n_samples = target_samples_per_class[c]
        x_c = generate_gaussian_data(target_means[c], target_covs[c], n_samples)
        y_c = np.full(n_samples, c)
        x_target_list.append(x_c)
        y_target_list.append(y_c)
    x_target = np.vstack(x_target_list)
    y_target = np.concatenate(y_target_list)

    def shuffle_data(x, y):
        idx = np.arange(len(y))
        np.random.shuffle(idx)
        return x[idx], y[idx]

    x_source, y_source = shuffle_data(x_source, y_source)
    x_target, y_target = shuffle_data(x_target, y_target)

    return x_source, y_source, x_target, y_target


def _sample_for_plot(x, max_points, seed):
    if max_points is None or x.shape[0] <= max_points:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=max_points, replace=False)
    return x[idx]


def plot_pca(x_source, x_target, out_path, max_points_per_domain=10000, seed=20):
    pca = PCA(n_components=2, random_state=42)
    all_data = np.concatenate([x_source, x_target], axis=0)
    pca_feats = pca.fit_transform(all_data)

    ns = x_source.shape[0]
    pca_s = pca_feats[:ns]
    pca_t = pca_feats[ns:]

    pca_s = _sample_for_plot(pca_s, max_points_per_domain, seed)
    pca_t = _sample_for_plot(pca_t, max_points_per_domain, seed)

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_s[:, 0], pca_s[:, 1], c="red", alpha=0.2, label="Source")
    plt.scatter(pca_t[:, 0], pca_t[:, 1], c="green", alpha=0.2, label="Target")
    plt.title("PCA (Source=Red, Target=Green)")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_tsne_raw(x_source, x_target, out_path, perplexity=500, max_points_per_domain=5000, seed=20):
    x_source_plot = _sample_for_plot(x_source, max_points_per_domain, seed)
    x_target_plot = _sample_for_plot(x_target, max_points_per_domain, seed + 1)

    all_data = np.concatenate([x_source_plot, x_target_plot], axis=0)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    all_2d = tsne.fit_transform(all_data)

    ns = x_source_plot.shape[0]
    s_2d = all_2d[:ns]
    t_2d = all_2d[ns:]

    plt.figure(figsize=(8, 6))
    plt.scatter(s_2d[:, 0], s_2d[:, 1], c="red", alpha=0.2, label="Source")
    plt.scatter(t_2d[:, 0], t_2d[:, 1], c="green", alpha=0.2, label="Target")
    plt.title("t-SNE Raw (Source=Red, Target=Green)")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_tsne_model(
    model,
    x_source,
    x_target,
    out_path,
    device,
    perplexity=500,
    max_points_per_domain=5000,
    seed=20,
    title="t-SNE Features (Source=Red, Target=Green)",
):
    model.eval()
    with torch.no_grad():
        x_s = torch.tensor(x_source, dtype=torch.float32, device=device)
        x_t = torch.tensor(x_target, dtype=torch.float32, device=device)
        feats_s = model.feature_extractor(x_s).cpu().numpy()
        feats_t = model.feature_extractor(x_t).cpu().numpy()

    feats_s = _sample_for_plot(feats_s, max_points_per_domain, seed)
    feats_t = _sample_for_plot(feats_t, max_points_per_domain, seed + 1)

    all_feats = np.concatenate([feats_s, feats_t], axis=0)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    all_2d = tsne.fit_transform(all_feats)

    ns = feats_s.shape[0]
    s_2d = all_2d[:ns]
    t_2d = all_2d[ns:]

    plt.figure(figsize=(8, 6))
    plt.scatter(s_2d[:, 0], s_2d[:, 1], c="red", alpha=0.2, label="Source")
    plt.scatter(t_2d[:, 0], t_2d[:, 1], c="green", alpha=0.2, label="Target")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def predict_logits(model, x, device):
    if hasattr(model, "predict"):
        logits = model.predict(x)
    else:
        logits = model(x)
    if isinstance(logits, tuple):
        if len(logits) == 2 and logits[0].shape == logits[1].shape:
            logits = (logits[0] + logits[1]) / 2.0
        else:
            logits = logits[0]
    return logits


def target_accuracy(model, x_target, y_target, device):
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_target, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_target, dtype=torch.long, device=device)
        logits = predict_logits(model, x_t, device)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y_t).float().mean().item()
    return acc


def train_mlp_baseline(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    epochs=20,
    batch_size=64,
    lr=1e-3,
    patience=5,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_acc = -1.0
    best_model_state = None
    patience_counter = 0

    for _ in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_acc = correct / max(1, total)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CGDM data settings (exactly from DomainAdaptation/CGDM/data_generation.py)
    num_classes = 6
    num_dims = 8
    num_samples_source = 1600
    num_samples_target = 1600

    class_ratios_source = [0.166] * num_classes
    class_ratios_target = [0.166, 0.166, 0.166, 0.166, 0.166, 0.166]

    source_mean_scale = 2.0
    source_cov_base = 1.0
    target_mean_offset_range = 2.0
    target_cov_noise_scale = 0.1
    target_cov_scale = 1.0

    seed = 20

    x_source, y_source, x_target, y_target = generate_source_target_data(
        num_classes=num_classes,
        num_dims=num_dims,
        num_samples_source=num_samples_source,
        num_samples_target=num_samples_target,
        class_ratios_source=class_ratios_source,
        class_ratios_target=class_ratios_target,
        source_mean_scale=source_mean_scale,
        source_cov_base=source_cov_base,
        target_mean_offset_range=target_mean_offset_range,
        target_cov_noise_scale=target_cov_noise_scale,
        target_cov_scale=target_cov_scale,
        random_seed=seed,
    )

    # Train/val split for source domain
    x_train, x_val, y_train, y_val = train_test_split(
        x_source, y_source, test_size=0.2, random_state=seed, stratify=y_source
    )

    out_dir = Path(__file__).resolve().parent / "cgdm_only"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_pca(x_source, x_target, out_dir / "pca_raw.jpg", max_points_per_domain=10000, seed=seed)
    plot_tsne_raw(x_source, x_target, out_dir / "tsne_raw.jpg", perplexity=500, max_points_per_domain=5000, seed=seed)

    mlp_hparams = {
        "backbone": "MLP",
        "hidden_dim": 256,
        "num_layers": 3,
        "dropout": 0.3,
    }
    mlp_model = DAModel(input_dim=num_dims, num_classes=num_classes, hparams=mlp_hparams)
    mlp_model = train_mlp_baseline(
        mlp_model,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=20,
        batch_size=64,
        lr=1e-3,
        patience=5,
        device=device,
    )
    mlp_acc = target_accuracy(mlp_model, x_target, y_target, device)
    plot_tsne_model(
        mlp_model,
        x_source,
        x_target,
        out_dir / "tsne_mlp.jpg",
        device,
        perplexity=500,
        max_points_per_domain=5000,
        seed=seed,
        title="t-SNE MLP Baseline (Source=Red, Target=Green)",
    )
    torch.save(mlp_model.state_dict(), out_dir / "mlp_model.pt")

    model = CGDM(input_dim=num_dims, num_classes=num_classes)
    model = train_cgdm(
        model,
        x_train,
        y_train,
        x_target,
        y_target=y_target,
        X_val=x_val,
        y_val=y_val,
        epochs=20,
        batch_size=64,
        lr=1e-4,
        device=device,
    )

    acc = target_accuracy(model, x_target, y_target, device)
    plot_tsne_model(
        model,
        x_source,
        x_target,
        out_dir / "tsne_cgdm.jpg",
        device,
        perplexity=500,
        max_points_per_domain=5000,
        seed=seed,
        title="t-SNE CGDM (Source=Red, Target=Green)",
    )

    ckpt_path = out_dir / "cgdm_model.pt"
    torch.save(model.state_dict(), ckpt_path)

    metrics = {
        "target_acc": round(float(acc), 4),
        "mlp_target_acc": round(float(mlp_acc), 4),
        "num_classes": num_classes,
        "num_dims": num_dims,
        "num_samples_source": num_samples_source,
        "num_samples_target": num_samples_target,
        "epochs": 20,
        "batch_size": 64,
        "perplexity": 500,
        "max_tsne": 5000,
        "device": device,
        "mlp_hparams": mlp_hparams,
        "mlp_lr": 1e-3,
        "mlp_patience": 5,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved outputs in {out_dir}")
    print(f"Target accuracy (CGDM): {acc:.4f}")
    print(f"Target accuracy (MLP): {mlp_acc:.4f}")


if __name__ == "__main__":
    main()
