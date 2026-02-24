import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# Ensure Ubicomp root is on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.da_models import (
    DAModel,
    DANN,
    CDAN,
    DAN,
    MCC,
    DeepCORAL,
    ADDA,
    JAN,
    MCD,
    SHOT,
    CBST,
    CGDM,
    train_standard,
    train_dann,
    train_adversarial_da,
    train_mcc,
    train_deepcoral,
    train_adda,
    train_dan,
    train_jan,
    train_mcd,
    train_shot,
    train_cbst,
    train_cgdm,
)
from src.domainbed_algos import (
    ERM,
    IRM,
    VREx,
    GroupDRO,
    MixStyle,
    MLDG,
    MASF,
    Fish,
    CSD,
    SagNet,
)


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


def plot_tsne_raw(x_source, x_target, out_path, perplexity=200, max_points_per_domain=5000, seed=20):
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


def extract_features(model, x, device):
    if hasattr(model, "feature_extractor"):
        return model.feature_extractor(x)
    if hasattr(model, "featurizer"):
        return model.featurizer(x)
    if hasattr(model, "target_encoder"):
        return model.target_encoder(x)
    return model(x)


def plot_tsne_model(model, x_source, x_target, out_path, device, perplexity=200, max_points_per_domain=5000, seed=20):
    model.eval()
    with torch.no_grad():
        x_s = torch.tensor(x_source, dtype=torch.float32, device=device)
        x_t = torch.tensor(x_target, dtype=torch.float32, device=device)
        feats_s = extract_features(model, x_s, device).cpu().numpy()
        feats_t = extract_features(model, x_t, device).cpu().numpy()

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
    plt.title(f"t-SNE ({out_path.stem})")
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


def train_dg(model, X_train, y_train, d_train, epochs, batch_size, domains_per_batch, device, seed):
    rng = np.random.default_rng(seed)
    model.to(device)
    model.train()

    unique_domains = np.unique(d_train)
    domain_indices = {d: np.where(d_train == d)[0] for d in unique_domains}
    steps_per_epoch = max(10, int(len(X_train) / (batch_size * max(1, domains_per_batch))))

    import inspect
    update_sig = inspect.signature(model.update)
    has_domain_indices = "domain_indices" in update_sig.parameters

    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            domain_batch = rng.choice(
                unique_domains,
                size=min(domains_per_batch, len(unique_domains)),
                replace=len(unique_domains) < domains_per_batch,
            )
            minibatches = []
            for d in domain_batch:
                idx_pool = domain_indices[d]
                idx = rng.choice(idx_pool, size=batch_size, replace=len(idx_pool) < batch_size)
                x = torch.tensor(X_train[idx], dtype=torch.float32, device=device)
                y = torch.tensor(y_train[idx], dtype=torch.long, device=device)
                minibatches.append((x, y))
            if has_domain_indices:
                model.update(minibatches, domain_indices=domain_batch)
            else:
                model.update(minibatches)

    return model


def main():
    parser = argparse.ArgumentParser(description="Verify DA/DG models on CGDM synthetic data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--perplexity", type=int, default=500)
    parser.add_argument("--max-tsne", type=int, default=5000)
    parser.add_argument("--max-pca", type=int, default=10000)
    parser.add_argument("--dg-domains", type=int, default=4)
    parser.add_argument("--skip-dg", action="store_true", help="Skip DG algorithms (only baseline + DA)")
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=Path("DA_Verification/CGDM/suite"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CGDM data settings
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
        random_seed=args.seed,
    )

    # Train/val split for source domain
    x_train, x_val, y_train, y_val = train_test_split(
        x_source, y_source, test_size=0.2, random_state=args.seed, stratify=y_source
    )
    d_train = np.zeros(len(y_train), dtype=np.int64)
    d_val = np.zeros(len(y_val), dtype=np.int64)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Raw PCA/TSNE
    plot_pca(x_source, x_target, out_dir / "pca_raw.jpg", max_points_per_domain=args.max_pca, seed=args.seed)
    plot_tsne_raw(x_source, x_target, out_dir / "tsne_raw.jpg", perplexity=args.perplexity, max_points_per_domain=args.max_tsne, seed=args.seed)

    results = []

    hparams = {
        "backbone": "MLP",
        "hidden_dim": 256,
        "num_layers": 3,
        "dropout": 0.3,
    }

    # Baseline: MLP (source supervised)
    baseline_model = DAModel(input_dim=num_dims, num_classes=num_classes, hparams=hparams)
    print("\n=== Baseline MLP ===")
    start = perf_counter()
    baseline_trained = train_standard(
        baseline_model,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=1e-4,
        patience=args.epochs,
        device=device,
    )
    elapsed = perf_counter() - start
    acc = target_accuracy(baseline_trained, x_target, y_target, device)
    plot_tsne_model(baseline_trained, x_source, x_target, out_dir / "tsne_mlp.jpg", device, perplexity=args.perplexity, max_points_per_domain=args.max_tsne, seed=args.seed)
    results.append({
        "name": "MLP",
        "family": "Baseline",
        "target_acc": acc,
        "seconds": round(elapsed, 2),
    })

    da_algorithms = [
        ("DANN", DANN(input_dim=num_dims, num_classes=num_classes, num_domains=2, hparams=hparams), lambda m: train_dann(
            m, x_train, y_train, d_train, x_val, y_val, d_val, epochs=args.epochs, batch_size=args.batch_size, lr=1e-4, patience=args.epochs, device=device, X_target=x_target
        )),
        ("CDAN", CDAN(input_dim=num_dims, num_classes=num_classes, num_domains=2, hparams=hparams), lambda m: train_adversarial_da(
            m, x_train, y_train, d_train, x_val, y_val, d_val, epochs=args.epochs, batch_size=args.batch_size, lr=1e-4, patience=args.epochs, device=device, X_target=x_target
        )),
        ("DAN", DAN(input_dim=num_dims, num_classes=num_classes, hparams=hparams), lambda m: train_dan(
            m, x_train, y_train, d_train, x_val, y_val, d_val, epochs=args.epochs, batch_size=args.batch_size, lr=1e-4, patience=args.epochs, device=device, X_target=x_target
        )),
        ("MCC", MCC(input_dim=num_dims, num_classes=num_classes, hparams={"mcc_temp": 2.0, **hparams}), lambda m: train_mcc(
            m, x_train, y_train, d_train, x_val, y_val, d_val, epochs=args.epochs, batch_size=args.batch_size, lr=1e-4, patience=args.epochs, device=device, X_target=x_target
        )),
        ("DeepCORAL", DeepCORAL(input_dim=num_dims, num_classes=num_classes, hparams=hparams), lambda m: train_deepcoral(
            m, x_train, y_train, d_train, x_val, y_val, d_val, epochs=args.epochs, batch_size=args.batch_size, lr=1e-4, patience=args.epochs, device=device, X_target=x_target
        )),
        ("ADDA", ADDA(input_dim=num_dims, num_classes=num_classes, hparams=hparams), lambda m: train_adda(
            m, x_train, y_train, d_train, x_val, y_val, d_val, epochs=args.epochs, batch_size=args.batch_size, lr=1e-4, patience=args.epochs, device=device, X_target=x_target
        )),
        ("JAN", JAN(input_dim=num_dims, num_classes=num_classes, hparams=hparams), lambda m: train_jan(
            m, x_train, y_train, d_train, x_val, y_val, d_val, epochs=args.epochs, batch_size=args.batch_size, lr=1e-4, patience=args.epochs, device=device, X_target=x_target
        )),
        ("MCD", MCD(input_dim=num_dims, num_classes=num_classes, hparams=hparams), lambda m: train_mcd(
            m, x_train, y_train, d_train, x_val, y_val, d_val, epochs=args.epochs, batch_size=args.batch_size, lr=1e-4, patience=args.epochs, device=device, X_target=x_target
        )),
        ("SHOT", SHOT(input_dim=num_dims, num_classes=num_classes, hparams=hparams), lambda m: train_shot(
            m, x_train, y_train, d_train, x_val, y_val, d_val, epochs=args.epochs, batch_size=args.batch_size, lr=1e-4, patience=args.epochs, device=device, X_target=x_target
        )),
        ("CBST", CBST(input_dim=num_dims, num_classes=num_classes, hparams=hparams), lambda m: train_cbst(
            m, x_train, y_train, d_train, x_val, y_val, d_val, epochs=args.epochs, batch_size=args.batch_size, lr=1e-4, patience=args.epochs, device=device, X_target=x_target
        )),
        ("CGDM", CGDM(input_dim=num_dims, num_classes=num_classes), lambda m: train_cgdm(
            m, x_train, y_train, x_target, y_target=y_target, X_val=x_val, y_val=y_val,
            epochs=args.epochs, batch_size=args.batch_size, lr=1e-4, device=device
        )),
    ]

    for name, model, train_fn in da_algorithms:
        print(f"\n=== DA {name} ===")
        start = perf_counter()
        trained = train_fn(model)
        elapsed = perf_counter() - start
        acc = target_accuracy(trained, x_target, y_target, device)
        plot_tsne_model(trained, x_source, x_target, out_dir / f"tsne_{name.lower()}.jpg", device, perplexity=args.perplexity, max_points_per_domain=args.max_tsne, seed=args.seed)
        results.append({
            "name": name,
            "family": "DA",
            "target_acc": acc,
            "seconds": round(elapsed, 2),
        })

    if not args.skip_dg:
        # DG models: create synthetic domains on source
        rng = np.random.default_rng(args.seed)
        d_source = rng.integers(0, args.dg_domains, size=len(x_source))
        x_train_dg, x_val_dg, y_train_dg, y_val_dg, d_train_dg, d_val_dg = train_test_split(
            x_source, y_source, d_source, test_size=0.2, random_state=args.seed, stratify=y_source
        )

        dg_algorithms = [
            ("ERM", ERM(input_dim=num_dims, num_classes=num_classes, hparams={"lr": 1e-4, **hparams})),
            ("IRM", IRM(input_dim=num_dims, num_classes=num_classes, hparams={"lr": 1e-4, **hparams})),
            ("VREx", VREx(input_dim=num_dims, num_classes=num_classes, hparams={"lr": 1e-4, **hparams})),
            ("GroupDRO", GroupDRO(input_dim=num_dims, num_classes=num_classes, hparams={"lr": 1e-4, "num_domains": args.dg_domains, **hparams})),
            ("MixStyle", MixStyle(input_dim=num_dims, num_classes=num_classes, hparams={"lr": 1e-4, **hparams})),
            ("MLDG", MLDG(input_dim=num_dims, num_classes=num_classes, hparams={"lr": 1e-4, **hparams})),
            ("MASF", MASF(input_dim=num_dims, num_classes=num_classes, hparams={"lr": 1e-4, **hparams})),
            ("Fish", Fish(input_dim=num_dims, num_classes=num_classes, hparams={"lr": 1e-4, **hparams})),
            ("CSD", CSD(input_dim=num_dims, num_classes=num_classes, hparams={"lr": 1e-4, **hparams})),
            ("SagNet", SagNet(input_dim=num_dims, num_classes=num_classes, hparams={"lr": 1e-4, **hparams})),
        ]

        for name, model in dg_algorithms:
            print(f"\n=== DG {name} ===")
            start = perf_counter()
            trained = train_dg(
                model,
                x_train_dg,
                y_train_dg,
                d_train_dg,
                epochs=args.epochs,
                batch_size=args.batch_size,
                domains_per_batch=min(args.dg_domains, 4),
                device=device,
                seed=args.seed,
            )
            elapsed = perf_counter() - start
            acc = target_accuracy(trained, x_target, y_target, device)
            plot_tsne_model(trained, x_source, x_target, out_dir / f"tsne_{name.lower()}.jpg", device, perplexity=args.perplexity, max_points_per_domain=args.max_tsne, seed=args.seed)
            results.append({
                "name": name,
                "family": "DG",
                "target_acc": acc,
                "seconds": round(elapsed, 2),
            })

    # Save results
    results_path = out_dir / "metrics.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    settings_path = out_dir / "settings.json"
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_classes": num_classes,
            "num_dims": num_dims,
            "num_samples_source": num_samples_source,
            "num_samples_target": num_samples_target,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "perplexity": args.perplexity,
            "max_tsne": args.max_tsne,
            "max_pca": args.max_pca,
            "dg_domains": args.dg_domains,
            "seed": args.seed,
            "device": device,
        }, f, indent=2)

    print(f"Done. Results: {results_path}")


if __name__ == "__main__":
    main()
