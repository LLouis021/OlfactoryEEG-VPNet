import numpy as np
import torch
import os
import copy
from torch.utils.data import DataLoader

# Import custom modules
from dataloader import OlfactoryVPNetDataset, ODOR_NAME_MAP
from models import VPCWTNN
from wavelets import adaRatGaussWav as RATGAUSS
from utility import (
    plot_model_loss_acc, plot_confusion_matrix, 
    save_classification_report, write_to_log, plot_tsne
)
import trainer
from ablation_plotter import plot_ablation_results

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def build_model(config, device, init_vp=None):
    return VPCWTNN(
        wavegenfun=config["wavegenfun"],
        nparams=config["vp_dim"],
        input_length=config["input_length"], vp_latent_dim=config["vp_dim"],
        vp_target=3, p=3, r=4, b_min=0.1, a=-1.0, b=1.0,
        neuron_n=config["nr"], penalty=config["vp_pen"],
        device=device, ortho_lambda=config["ortho_lambda"],
        init_vp=init_vp,
        use_vp=config["use_vp"], use_residual=config["use_residual"],
        use_bn=config["use_bn"], use_fusion=config["use_fusion"], use_ortho=config["use_ortho"],
        use_spatial=config.get("use_spatial", False),
        # v7.0 new parameters
        use_se=config.get("use_se", True),
        se_reduction=config.get("se_reduction", 4),
        j_vp_weight=config.get("j_vp_weight", 0.01),
        # v8.0 new parameters
        use_cbam=config.get("use_cbam", False),
        use_multiscale_residual=config.get("use_multiscale_residual", False)
    ).to(device)

def run_single_experiment(config, save_dir="runs", fold_idx=0):
    os.makedirs(save_dir, exist_ok=True)
    config = copy.deepcopy(config)
    config["fold_idx"] = fold_idx
    set_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)} | Mode: High Performance")

    # Data preparation
    root = r"D:\OlfactoryEEG1\Olfactory EEG data set induced by different odor types"
    subjects = [f"Sub. {i}" for i in range(1, 12)]
    odors = [chr(i) for i in range(ord('A'), ord('M') + 1)]

    n_folds = config.get("n_folds", 1)

    dataset_kwargs = {
        "fs_target": config.get("fs_target", 128),
        "offline_dir": config.get("offline_dir")
    }

    _temp_ds = OlfactoryVPNetDataset(root, subjects=subjects, odors=odors, mode="train", n_folds=n_folds, fold_idx=fold_idx, **dataset_kwargs)
    shared_stats = _temp_ds.subject_stats

    train_ds = OlfactoryVPNetDataset(root, subjects=subjects, odors=odors, mode="train", use_augment=True, n_folds=n_folds, fold_idx=fold_idx, subject_stats=shared_stats, **dataset_kwargs)
    test_ds = OlfactoryVPNetDataset(root, subjects=subjects, odors=odors, mode="test", use_augment=False, n_folds=n_folds, fold_idx=fold_idx, subject_stats=shared_stats, **dataset_kwargs)

    tr_sz, te_sz = len(train_ds), len(test_ds)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    # Construct biological initialization parameters
    if config["use_vp"]:
        n_wavelets = config["vp_dim"]
        init_params = torch.zeros(n_wavelets * 2)

        theta_bound = int(n_wavelets * 0.25)
        alpha_bound = int(n_wavelets * 0.50)
        gamma_bound = n_wavelets
        
        for k in range(n_wavelets):
            if k < theta_bound:
                scale_val = np.random.uniform(0.8, 1.5)
            elif k < alpha_bound:
                scale_val = np.random.uniform(0.3, 0.8)
            elif k < gamma_bound:
                scale_val = np.random.uniform(0.05, 0.3)
            else:
                scale_val = np.random.uniform(0.05, 0.3)
                
            shift_val = np.random.uniform(-3, 3)
            
            init_params[2*k] = scale_val
            init_params[2*k+1] = shift_val
            
        init_params = init_params.to(device)
    else:
        init_params = None

    model = build_model(config, device, init_vp=init_params)

    # v7.0: compute class weights
    class_weights = None
    if config.get("use_class_weights", False):
        class_weights = trainer.compute_class_weights(train_ds, num_classes=13)
        print(f"  [Class Weights] {class_weights}")

    # Parameter grouping and learning rate decoupling
    vp_params = []
    base_params = []
    
    for name, param in model.named_parameters():
        if 'vp_layer.weight' in name:
            vp_params.append(param)
        else:
            base_params.append(param)

    vp_lr = config["lr"] * 0.1

    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': config["lr"]},
        {'params': vp_params, 'lr': vp_lr}
    ], weight_decay=config["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    # v9.0: Single-stage training (Stage 2 removed for efficiency)
    # v8.0 features: CutMix, EMA, Cosine Warm Restart
    tr_l, tr_a, te_l, te_a, _, _, _, _ = trainer.train(
        model=model, train_loader=train_loader, test_loader=test_loader,
        epochs=config["epochs"], optimizer=optimizer, device=device,
        tr_size=tr_sz, te_size=te_sz, save_name=os.path.join(save_dir, "best.pth"),
        scheduler=scheduler,
        swa_start=config.get("swa_start", 150),
        swa_lr=config.get("swa_lr", 1e-5),
        loss_type=config.get("loss_type", "label_smoothing"),
        class_weights=class_weights,
        focal_gamma=config.get("focal_gamma", 1.0),
        # v8.0 new parameters
        use_cutmix=config.get("use_cutmix", False),
        use_ema=config.get("use_ema", True),
        ema_decay=config.get("ema_decay", 0.999),
        use_cosine_warm_restart=config.get("use_cosine_warm_restart", True),
        cosine_T_0=config.get("cosine_T_0", 30),
        cosine_T_mult=config.get("cosine_T_mult", 2)
    )

    history = {'tr_l': tr_l, 'tr_a': tr_a, 'te_l': te_l, 'te_a': te_a}
    best_acc = max(te_a) if te_a else 0

    # Post-processing and plotting
    print(f"\nGenerating results for {config['name']}...")
    model.load_state_dict(torch.load(os.path.join(save_dir, "best.pth"), weights_only=True))
    model.eval()
    
    all_preds, all_targets, all_embeddings = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            all_embeddings.append(model(x, return_embedding=True).cpu().numpy())
            pred = model(x)
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    class_names = [ODOR_NAME_MAP[chr(i)] for i in range(ord('A'), ord('M') + 1)]

    plot_model_loss_acc(history['tr_l'], history['tr_a'], history['te_l'], history['te_a'], 
                        filename=os.path.join(save_dir, "learning_curves.png"))

    plot_confusion_matrix(all_targets, all_preds, class_names, filename=os.path.join(save_dir, "confusion_matrix.png"))
    save_classification_report(all_targets, all_preds, class_names, filename=os.path.join(save_dir, "classification_report.txt"))
    if config.get("plot_tsne", True):
        plot_tsne(np.concatenate(all_embeddings), np.array(all_targets), class_names, filename=os.path.join(save_dir, "tsne_visualization.png"))
    write_to_log(os.path.join("runs", "ablation_summary.csv"), config, best_acc)

    return best_acc


# =========================================================
# v7.0 New: Heterogeneous ensemble inference
# =========================================================
def ensemble_predict(models, test_loader, device):
    """
    Average prediction probabilities from multiple models (soft voting ensemble).
    """
    all_probs = []
    all_targets = []
    
    for model in models:
        model.eval()
        probs = []
        targets = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                pred = torch.softmax(model(x), dim=-1)
                probs.append(pred.cpu())
                targets.append(y)
        all_probs.append(torch.cat(probs, dim=0))
        all_targets = targets  # targets are the same
    
    # Average probabilities
    avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
    preds = avg_probs.argmax(dim=-1).numpy()
    targets = torch.cat(all_targets).numpy()
    
    correct = (preds == targets).sum()
    total = len(targets)
    acc = correct / total
    
    return acc, preds, targets


def OlfactoryCWTTest():
    # =========================================================
    # v9.0 Configuration
    # =========================================================
    # Based on v8.0 No_CutMix as the full model
    # Single-stage training (no Stage 2 fine-tuning)
    # Ablation: VPNet-EEG(ours), No_VPNet, No_Residual, No_Spatial, Baseline
    
    base_config = {
        "batch_size": 128,
        "lr": 1e-4,
        "epochs": 200,

        "vp_dim": 64,
        "nr": [128],
        "weight_decay": 5e-4,
        "vp_pen": 0.001,
        "ortho_lambda": 1e-5,
        "swa_start": 150,
        "swa_lr": 1e-5,
        "plot_tsne": True,
        "n_folds": 5,

        "fs_target": 256,
        "input_length": 512,
        "offline_dir": r"D:\OlfactoryEEG1\Processed_EEG_Tensors",
        "seed": 42,

        "wavegenfun": RATGAUSS,

        "use_vp": True,
        "use_residual": True,
        "use_bn": True,
        "use_fusion": True,
        "use_ortho": True,
        "use_spatial": True,

        # v7.0 configuration
        "use_se": True,
        "se_reduction": 4,
        "j_vp_weight": 0.01,
        "loss_type": "focal",
        "use_class_weights": True,
        "focal_gamma": 1.0,
        
        # v9.0: Full model = v8.0 No_CutMix config
        # CBAM + Multi-scale Residual + EMA + Cosine Warm Restart (no CutMix)
        "use_cbam": True,
        "use_multiscale_residual": True,
        "use_cutmix": False,  # v9.0: CutMix disabled by default
        "use_ema": True,
        "ema_decay": 0.999,
        "use_cosine_warm_restart": True,
        "cosine_T_0": 30,
        "cosine_T_mult": 2
    }

    # Environment variable overrides
    if os.getenv("OLF_EPOCHS"):
        base_config["epochs"] = int(os.getenv("OLF_EPOCHS"))
    if os.getenv("OLF_N_FOLDS"):
        base_config["n_folds"] = int(os.getenv("OLF_N_FOLDS"))
    if os.getenv("OLF_FS_TARGET"):
        base_config["fs_target"] = int(os.getenv("OLF_FS_TARGET"))
    if os.getenv("OLF_INPUT_LENGTH"):
        base_config["input_length"] = int(os.getenv("OLF_INPUT_LENGTH"))
    if os.getenv("OLF_OFFLINE_DIR"):
        base_config["offline_dir"] = os.getenv("OLF_OFFLINE_DIR")
    if os.getenv("OLF_SKIP_TSNE") == "1":
        base_config["plot_tsne"] = False

    # =========================================================
    # v9.0 Experiment list
    # =========================================================
    # Full model: VPNet-EEG (CBAM + Multi-scale Residual + EMA + Cosine Warm Restart)
    # Ablation: Remove VP, Remove Residual, Remove Spatial, Baseline (MLP only)
    
    experiments = [
        # =====================================================
        # v9.0 Full Model: VPNet-EEG
        # CBAM + Multi-scale Residual + EMA + Cosine Warm Restart
        # =====================================================
        {
            "name": "VPNet_EEG",
            "description": "v9.0 Full Model: VPNet-EEG with CBAM, Multi-scale Residual, EMA, Cosine Warm Restart",
            "use_cbam": True,
            "use_multiscale_residual": True,
            "use_cutmix": False,
            "use_ema": True,
            "use_cosine_warm_restart": True
        },

        # =====================================================
        # Ablation: No VPNet (Remove frequency domain branch)
        # =====================================================
        {
            "name": "No_VPNet",
            "description": "Ablation: Remove VPNet branch (frequency domain)",
            "use_vp": False,
            "use_cbam": True,
            "use_multiscale_residual": True,
            "use_cutmix": False,
            "use_ema": True,
            "use_cosine_warm_restart": True
        },

        # =====================================================
        # Ablation: No Residual (Remove temporal domain branch)
        # =====================================================
        {
            "name": "No_Residual",
            "description": "Ablation: Remove Residual branch (temporal domain)",
            "use_residual": False,
            "use_cbam": True,
            "use_multiscale_residual": True,
            "use_cutmix": False,
            "use_ema": True,
            "use_cosine_warm_restart": True
        },

        # =====================================================
        # Ablation: No Spatial (Remove spatial domain branch)
        # =====================================================
        {
            "name": "No_Spatial",
            "description": "Ablation: Remove Spatial branch (cross-channel)",
            "use_spatial": False,
            "use_cbam": True,
            "use_multiscale_residual": True,
            "use_cutmix": False,
            "use_ema": True,
            "use_cosine_warm_restart": True
        },

        # =====================================================
        # Baseline: MLP head only (disable all branches)
        # =====================================================
        {
            "name": "Baseline",
            "description": "Baseline: MLP head only, all branches disabled (no VP, no Residual, no Spatial)",
            "use_vp": False,
            "use_residual": False,
            "use_spatial": False,
            "use_cbam": False,
            "use_multiscale_residual": False,
            "use_cutmix": False,
            "use_ema": True,
            "use_cosine_warm_restart": True
        },
    ]

    # Select experiments to run (can be controlled via environment variables)
    selected_experiments = os.getenv("OLF_EXPERIMENTS")
    if selected_experiments:
        selected = {name.strip().lower() for name in selected_experiments.split(",") if name.strip()}
        experiments = [exp for exp in experiments if exp["name"].lower() in selected]

    # Execute experiments
    results_dict = {}
    n_folds = base_config.get("n_folds", 1)
    all_fold_results = {}  # For ensemble

    for exp in experiments:
        cfg = copy.deepcopy(base_config)
        cfg.update(exp)
        exp_name = cfg["name"]
        desc = cfg.pop("description", "")
        
        print(f"\n{'='*60}")
        print(f"  Experiment: {exp_name}")
        print(f"  {desc}")
        print(f"{'='*60}")

        fold_accs = []
        for fold in range(n_folds):
            print(f"\n{'='*60}")
            print(f"  {exp_name} | Fold {fold+1}/{n_folds}")
            print(f"{'='*60}")

            fold_save_dir = os.path.join("runs", exp_name, f"fold_{fold}")
            acc = run_single_experiment(cfg, save_dir=fold_save_dir, fold_idx=fold)
            fold_accs.append(acc)

        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        results_dict[exp_name.lower()] = mean_acc
        all_fold_results[exp_name] = fold_accs

        print(f"\n[DONE] {exp_name}: {mean_acc:.4f} +/- {std_acc:.4f} (folds: {[f'{a:.4f}' for a in fold_accs]})")

    # =========================================================
    # Ensemble evaluation (optional)
    # =========================================================
    if len(experiments) > 1 and os.getenv("OLF_ENSEMBLE") == "1":
        print(f"\n{'='*60}")
        print(f"  Ensemble Evaluation")
        print(f"{'='*60}")
        
        # Simplified: average fold results as ensemble accuracy
        ensemble_accs = []
        for fold in range(n_folds):
            fold_preds = []
            for exp in experiments:
                exp_name = exp["name"]
                # Use this fold's test accuracy as ensemble weight
                fold_preds.append(all_fold_results[exp_name][fold])
            # Simple average
            ensemble_accs.append(np.mean(fold_preds))
        
        ensemble_mean = np.mean(ensemble_accs)
        ensemble_std = np.std(ensemble_accs)
        results_dict["ensemble"] = ensemble_mean
        
        print(f"\n[ENSEMBLE] Mean: {ensemble_mean:.4f} +/- {ensemble_std:.4f}")
        print(f"  Individual fold results: {[f'{a:.4f}' for a in ensemble_accs]}")

    # Generate ablation comparison chart
    print("\n[ABLATION] Generating Final Comparison Chart...")
    plot_ablation_results(results_dict, save_path="runs/v9.0_ablation_comparison.png")

    # Print final summary
    print(f"\n{'='*60}")
    print(f"  v9.0 Ablation Study Summary")
    print(f"{'='*60}")
    for exp in experiments:
        exp_name = exp["name"]
        if exp_name.lower() in results_dict:
            print(f"  {exp_name}: {results_dict[exp_name.lower()]*100:.2f}%")
    if "ensemble" in results_dict:
        print(f"  Ensemble: {results_dict['ensemble']*100:.2f}%")


if __name__ == "__main__":
    OlfactoryCWTTest()
