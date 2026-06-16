import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np

# =========================================================
# Data augmentation tools (Mixup)
# =========================================================
def mixup_data(x, y, alpha=0.2, device='cuda'):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =========================================================
# v8.0 New: CutMix Data Augmentation
# =========================================================
def cutmix_data(x, y, alpha=1.0, device='cuda'):
    """
    CutMix: cut and paste a rectangular region from one sample to another.
    
    Args:
        x: input tensor [B, C, T]
        y: label tensor [B]
        alpha: Beta distribution parameter
        device: device to use
    Returns:
        mixed_x, y_a, y_b, lam
    """
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    # Sample lambda from Beta distribution
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    
    # Generate random bounding box
    # For 1D EEG data, we cut a time segment
    T = x.size(2)
    cut_ratio = np.sqrt(1.0 - lam)  # Cut ratio based on lambda
    cut_len = int(T * cut_ratio)
    
    # Random start position
    start = np.random.randint(0, T - cut_len + 1)
    end = start + cut_len
    
    # Create mask (1 for original, 0 for cut region)
    mask = torch.ones_like(x)
    mask[:, :, start:end] = 0
    
    # Apply CutMix
    mixed_x = x * mask + x[index] * (1 - mask)
    
    # Adjust lambda based on actual cut area
    lam = 1 - (end - start) / T
    
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """CutMix loss: weighted combination of two losses"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =========================================================
# v8.0 New: EMA (Exponential Moving Average) Model
# =========================================================
class EMAModel:
    """
    Exponential Moving Average of model parameters.
    Provides better generalization by averaging model weights over training.
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters after each training step"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to model (for evaluation)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters (for continued training)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# =========================================================
# v8.0 New: Cosine Annealing with Warm Restarts
# =========================================================
class CosineAnnealingWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine Annealing with Warm Restarts.
    Learning rate decays following cosine curve and restarts periodically.
    """
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = T_0
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * 
                (1 + np.cos(np.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        self.T_cur += 1
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i = int(self.T_i * self.T_mult)
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


# =========================================================
# v7.0 New: Focal Loss (address hard sample problem)
# =========================================================
class FocalLoss(torch.nn.Module):
    """
    Focal Loss: assigns higher weights to hard-to-classify samples, reducing contribution of easy samples.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    gamma=0 degenerates to standard CE; higher gamma focuses more on hard samples.
    v7.0 uses gamma=1.0 as a moderate starting point.
    """
    def __init__(self, gamma=1.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Can pass class_weights tensor
        self.reduction = reduction

    def forward(self, pred, target):
        # pred: [B, C], target: [B]
        ce_loss = F.cross_entropy(pred, target, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  # p_t = exp(-CE)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# =========================================================
# Label Smoothing Loss (v7.0: supports class_weights)
# =========================================================
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes=13, smoothing=0.1, class_weights=None):
        super().__init__()
        self.classes = classes
        self.smoothing = smoothing
        self.class_weights = class_weights  # v7.0: new class weights

    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_prob)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = torch.sum(-true_dist * log_prob, dim=-1)  # [B]
        
        # v7.0: if class weights are provided, weight each sample
        if self.class_weights is not None:
            weight_vec = self.class_weights.to(pred.device)
            sample_weights = weight_vec[target]  # [B]
            loss = loss * sample_weights
        
        return loss.mean()


# =========================================================
# v7.0 New: Compute class weights (inverse frequency weighting)
# =========================================================
def compute_class_weights(dataset, num_classes=13):
    """
    Compute inverse frequency weights based on class sample counts.
    Classes with fewer samples get higher weights, helping the model focus on minority classes.
    """
    from collections import Counter
    label_counts = Counter()
    for sample in dataset.samples:
        label_counts[sample["label"]] += 1
    
    total = sum(label_counts.values())
    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)  # Avoid division by zero
        weights.append(total / (num_classes * count))
    
    weights = torch.tensor(weights, dtype=torch.float32)
    # Normalize to mean=1
    weights = weights / weights.mean()
    return weights


# =========================================================
# Train single epoch
# =========================================================
def train_single_epoch(dataloader, model, loss_fn, optimizer, device, scaler=None, 
                       use_mixup=True, use_cutmix=True, ema_model=None):
    model.train()
    pbar = tqdm(dataloader, desc="  Training", leave=False)
    
    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            # Randomly choose between Mixup, CutMix, or no augmentation
            aug_type = np.random.choice(['mixup', 'cutmix', 'none'], 
                                       p=[0.4, 0.3, 0.3])
            
            if use_mixup and aug_type == 'mixup':
                inputs, ya, yb, lam = mixup_data(X, y, alpha=0.4, device=device)
                pred = model(inputs)
                base_l = mixup_criterion(loss_fn, pred, ya, yb, lam)
                loss = model.compute_total_loss(pred, base_loss_val=base_l)
            elif use_cutmix and aug_type == 'cutmix':
                inputs, ya, yb, lam = cutmix_data(X, y, alpha=1.0, device=device)
                pred = model(inputs)
                base_l = cutmix_criterion(loss_fn, pred, ya, yb, lam)
                loss = model.compute_total_loss(pred, base_loss_val=base_l)
            else:
                pred = model(X)
                loss = model.compute_total_loss(pred, target=y, base_loss_fn=loss_fn)

        if torch.isnan(loss): continue

        if scaler:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Update EMA model if provided
        if ema_model is not None:
            ema_model.update()
            
        pbar.set_postfix(loss=f"{loss.item():.4f}")

# =========================================================
# Test evaluation
# =========================================================
def test(dataloader, model, loss_fn, size, device, log=True):
    model.eval()
    test_loss, correct = 0, 0
    pbar = tqdm(dataloader, desc="  Evaluating", leave=False)
    
    with torch.no_grad():
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            with torch.amp.autocast('cuda'):
                pred = model(X)
                l = model.compute_total_loss(pred, target=y, base_loss_fn=loss_fn)
            
            test_loss += l.item()
            correct += (pred.argmax(1) == y).sum().item()
            pbar.set_postfix(loss=f"{l.item():.4f}")
            
    test_loss /= len(dataloader)
    correct /= size if size > 0 else 1
    
    if log: 
        print(f"  Result -> Accuracy: {(100 * correct):>0.3f}%, Avg loss: {test_loss:>8f}")
    return test_loss, correct

# =========================================================
# Main training loop (v7.0: supports Focal Loss, save best model by Accuracy)
# =========================================================
def train(model, train_loader, test_loader, epochs, optimizer, device, tr_size, te_size,
          scheduler=None, save_name="best_model.pth", swa_start=150, swa_lr=1e-5,
          # v7.0 new parameters
          loss_type="label_smoothing",  # "label_smoothing" or "focal"
          class_weights=None,
          focal_gamma=1.0,
          # v8.0 new parameters
          use_cutmix=True,
          use_ema=True,
          ema_decay=0.999,
          use_cosine_warm_restart=False,
          cosine_T_0=30,
          cosine_T_mult=2):
    
    if os.path.exists(save_name):
        os.remove(save_name)

    scaler = torch.amp.GradScaler('cuda')

    # v7.0: select loss function
    if class_weights is not None:
        class_weights = class_weights.to(device) 
    
    if loss_type == "focal":
        loss_fn = FocalLoss(gamma=focal_gamma, alpha=class_weights)
        print(f"  [Loss] Using Focal Loss (gamma={focal_gamma})")
    else:
        loss_fn = LabelSmoothingLoss(classes=13, class_weights=class_weights)
        if class_weights is not None:
            print(f"  [Loss] Using Label Smoothing + Class Weights")
        else:
            print(f"  [Loss] Using Label Smoothing")

    # v8.0: EMA model setup
    ema_model = EMAModel(model, decay=ema_decay) if use_ema else None
    if use_ema:
        print(f"  [EMA] Using Exponential Moving Average (decay={ema_decay})")

    # v8.0: Cosine Annealing with Warm Restarts
    if use_cosine_warm_restart:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cosine_T_0, T_mult=cosine_T_mult)
        print(f"  [Scheduler] Using Cosine Annealing with Warm Restarts (T_0={cosine_T_0}, T_mult={cosine_T_mult})")

    # SWA setup
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=swa_lr)
    is_swa_phase = False

    tr_l_list, tr_a_list, te_l_list, te_a_list = [], [], [], []
    
    # v7.0 change: save best model by Accuracy (previously by Loss which may not be most accurate)
    best_acc = 0
    best_loss = float('inf')

    # Early Stopping
    early_stop_patience = 30
    patience_counter = 0

    for ep in range(epochs):
        # --- SWA phase transition ---
        if not is_swa_phase and ep >= swa_start:
            is_swa_phase = True
            print(f"\n{'='*60}")
            print(f"  Entering SWA phase at epoch {ep + 1}")
            print(f"{'='*60}")

        print(f"\n[EPOCH {ep + 1}/{epochs}] - LR: {optimizer.param_groups[0]['lr']:.6f}")

        # v8.0: train with CutMix and EMA
        train_single_epoch(train_loader, model, loss_fn, optimizer, device, scaler=scaler,
                          use_mixup=True, use_cutmix=use_cutmix, ema_model=ema_model)

        # Evaluate with EMA model if available
        if use_ema and ema_model is not None:
            # Apply EMA weights for evaluation
            ema_model.apply_shadow()
            tr_l, tr_a = test(train_loader, model, loss_fn, tr_size, device, log=False)
            te_l, te_a = test(test_loader, model, loss_fn, te_size, device, log=True)
            # Restore original weights for continued training
            ema_model.restore()
            print(f"  [EMA] EMA evaluation - Train Acc: {tr_a*100:.2f}%, Test Acc: {te_a*100:.2f}%")
        else:
            tr_l, tr_a = test(train_loader, model, loss_fn, tr_size, device, log=True)
            te_l, te_a = test(test_loader, model, loss_fn, te_size, device, log=True)

        tr_l_list.append(tr_l); tr_a_list.append(tr_a)
        te_l_list.append(te_l); te_a_list.append(te_a)

        if is_swa_phase:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            if te_a > best_acc:
                best_acc = te_a
                best_loss = te_l
            print(f"  SWA -> Best Acc so far: {best_acc*100:.2f}%")
        else:
            # v7.0: save best model by Accuracy
            if te_a > best_acc:
                best_acc = te_a
                best_loss = te_l
                patience_counter = 0
                # Save EMA model if available, otherwise original model
                if use_ema and ema_model is not None:
                    ema_model.apply_shadow()
                    torch.save(model.state_dict(), save_name)
                    ema_model.restore()
                else:
                    torch.save(model.state_dict(), save_name)
                print(f"  [BEST] New best Acc: {best_acc*100:.2f}% (Loss: {best_loss:.4f}), saved to {save_name}")
            elif te_a == best_acc and te_l < best_loss:
                # Same Acc, choose lower Loss
                best_loss = te_l
                patience_counter = 0
                if use_ema and ema_model is not None:
                    ema_model.apply_shadow()
                    torch.save(model.state_dict(), save_name)
                    ema_model.restore()
                else:
                    torch.save(model.state_dict(), save_name)
                print(f"  [BEST] Same Acc but better Loss: {best_loss:.4f}, saved to {save_name}")
            else:
                patience_counter += 1
                print(f"  [WAIT] No improvement ({patience_counter}/{early_stop_patience})")

            if scheduler and not use_cosine_warm_restart:
                scheduler.step()

        torch.cuda.empty_cache()

        # Early Stop (only before SWA phase)
        if not is_swa_phase and patience_counter >= early_stop_patience:
            print(f"\n[STOP] Early stopping triggered at epoch {ep+1}")
            print(f"Best Acc: {best_acc*100:.2f}% (Loss: {best_loss:.4f})")
            break

    # Finalize SWA model
    if is_swa_phase:
        print("  Updating SWA batch norm statistics...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.module.state_dict(), save_name)
        print(f"  SWA model saved to {save_name}")

    return tr_l_list, tr_a_list, te_l_list, te_a_list, [], [], [], []
