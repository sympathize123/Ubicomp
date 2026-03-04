import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from src.da_models import DANN, train_dann  # noqa: E402


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
    # Same data generation logic as /home/iclab/minseo/DomainAdaptation/CGDM/data_generation.py
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


def run_tsne(
    model,
    x_source,
    x_target,
    out_path,
    device,
    perplexity=200,
    max_points_per_domain=5000,
    seed=20,
):
    model.eval()
    with torch.no_grad():
        feats_s = model.feature_extractor(torch.tensor(x_source, dtype=torch.float32, device=device))
        feats_t = model.feature_extractor(torch.tensor(x_target, dtype=torch.float32, device=device))

    feats_s_np = feats_s.cpu().numpy()
    feats_t_np = feats_t.cpu().numpy()

    rng = np.random.default_rng(seed)
    if max_points_per_domain is not None:
        if feats_s_np.shape[0] > max_points_per_domain:
            idx_s = rng.choice(feats_s_np.shape[0], size=max_points_per_domain, replace=False)
            feats_s_np = feats_s_np[idx_s]
        if feats_t_np.shape[0] > max_points_per_domain:
            idx_t = rng.choice(feats_t_np.shape[0], size=max_points_per_domain, replace=False)
            feats_t_np = feats_t_np[idx_t]

    all_feats = np.concatenate([feats_s_np, feats_t_np], axis=0)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    all_feats_2d = tsne.fit_transform(all_feats)

    ns = feats_s_np.shape[0]
    feats_s_2d = all_feats_2d[:ns]
    feats_t_2d = all_feats_2d[ns:]

    plt.figure(figsize=(8, 6))
    plt.scatter(feats_s_2d[:, 0], feats_s_2d[:, 1], c="red", alpha=0.2, label="Source")
    plt.scatter(feats_t_2d[:, 0], feats_t_2d[:, 1], c="green", alpha=0.2, label="Target")
    plt.title("t-SNE Visualization (Source=Red, Target=Green)")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main():
    # Settings copied from DomainAdaptation/CGDM/data_generation.py and train.py
    num_classes = 6
    num_dims = 8
    num_samples_source = 160000
    num_samples_target = 160000

    class_ratios_source = [0.166] * num_classes
    class_ratios_target = [0.166, 0.166, 0.166, 0.166, 0.166, 0.166]

    source_mean_scale = 2.0
    source_cov_base = 1.0
    target_mean_offset_range = 2.0
    target_cov_noise_scale = 0.1
    target_cov_scale = 1.0

    random_seed = 20

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
        random_seed=random_seed,
    )

    # Train/val split for source domain
    x_train, x_val, y_train, y_val = train_test_split(
        x_source, y_source, test_size=0.2, random_state=random_seed, stratify=y_source
    )

    d_train = np.zeros(len(y_train), dtype=np.int64)
    d_val = np.zeros(len(y_val), dtype=np.int64)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DANN(
        input_dim=num_dims,
        num_classes=num_classes,
        num_domains=2,
        hparams={
            "backbone": "MLP",
            "hidden_dim": 256,
            "num_layers": 3,
            "dropout": 0.3,
        },
    )

    model = train_dann(
        model,
        x_train,
        y_train,
        d_train,
        x_val,
        y_val,
        d_val,
        epochs=20,
        batch_size=64,
        lr=1e-4,
        patience=20,
        device=device,
        X_target=x_target,
    )

    out_dir = Path(__file__).resolve().parent
    tsne_path = out_dir / "tsne_adapt_ubicomp_dann.jpg"
    run_tsne(
        model,
        x_source,
        x_target,
        tsne_path,
        device=device,
        perplexity=200,
        max_points_per_domain=5000,
        seed=random_seed,
    )

    # Save model checkpoint for inspection
    ckpt_path = out_dir / "ubicomp_dann_cgdm.pt"
    torch.save(model.state_dict(), ckpt_path)

    # Simple target accuracy (optional diagnostic)
    model.eval()
    with torch.no_grad():
        logits = model.predict(torch.tensor(x_target, dtype=torch.float32, device=device))
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    target_acc = (preds == y_target).mean()

    metrics_path = out_dir / "metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"target_acc={target_acc:.4f}\n")
        f.write("notes=DA_Verification/CGDM with Ubicomp DANN and CGDM data settings\n")

    print(f"Saved t-SNE to {tsne_path}")
    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Target accuracy: {target_acc:.4f}")


if __name__ == "__main__":
    main()
