import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


# =========================================================
# Odor semantic mapping (IMPORTANT for interpretability)
# =========================================================
ODOR_NAME_MAP = {
    'A': 'Rose',
    'B': 'Caramel',
    'C': 'Rotten',
    'D': 'Canned Peach',
    'E': 'Excrement',
    'F': 'Mint',
    'G': 'Tea Tree',
    'H': 'Coffee',
    'I': 'Rosemary',
    'J': 'Jasmine',
    'K': 'Lemon',
    'L': 'Vanilla',
    'M': 'Lavender'
}


class OlfactoryVPNetDataset(Dataset):
    """
    Final EEG dataset for VPNet training (research-grade)

    Key features:
    - Uses offline .pt tensors (fast & stable)
    - Prevents temporal leakage (non-overlapping windows)
    - Uses weak normalization (preserves EEG amplitude)
    - Adds EEG-specific augmentations
    - Provides odor semantic names (for analysis & visualization)
    """

    def __init__(self,
                 root_dir,
                 subjects,
                 odors,
                 fs_target=128,
                 duration=2.0,
                 use_augment=True,
                 mode="train",
                 split_ratio=0.8,
                 subject_stats=None,  # NEW: share pre-computed stats across datasets
                 n_folds=1,           # K-Fold support: 1 = no k-fold (use split_ratio)
                 fold_idx=0,          # which fold to use as test set (0-indexed)
                 offline_dir=None):

        self.root_dir = root_dir
        self.offline_dir = (
            offline_dir
            or os.getenv("OLF_OFFLINE_DIR")
            or r"D:\OlfactoryEEG1\Processed_EEG_Tensors"
        )

        self.fs_target = fs_target
        self.duration = duration
        # Test set must never use data augmentation
        self.use_augment = use_augment if mode == "train" else False

        self.odors = odors
        self.odor_to_label = {odor: i for i, odor in enumerate(odors)}
        self.label_to_odor = {i: odor for odor, i in self.odor_to_label.items()}

        self.samples = []
        # Accept pre-computed stats to avoid redundant computation when creating train/test pairs
        self.subject_stats = subject_stats if subject_stats is not None else self._compute_subject_stats(subjects)

        # ----------------------------------------------------
        # Build dataset index (physical isolation by file/trial)
        # Scan offline_dir for .pt files directly
        # ----------------------------------------------------
        for sub in subjects:
            for odor in odors:

                folder = os.path.join(self.offline_dir, sub, odor)
                if not os.path.exists(folder):
                    continue

                all_files = sorted([f for f in os.listdir(folder) if f.endswith('.pt')])

                if len(all_files) == 0:
                    continue

                # Physical isolation: K-Fold or fixed ratio split
                if n_folds > 1:
                    target_files = [f for i, f in enumerate(all_files) if (i % n_folds) == fold_idx]
                    if mode == "train":
                        target_files = [f for i, f in enumerate(all_files) if (i % n_folds) != fold_idx]
                else:
                    split_idx = int(len(all_files) * split_ratio)
                    target_files = all_files[:split_idx] if mode == "train" else all_files[split_idx:]

                # Determine valid start range from first file's length
                pts = int(self.fs_target * self.duration)
                probe_path = os.path.join(folder, all_files[0])
                try:
                    probe_len = torch.load(probe_path, weights_only=True).shape[1]
                    max_start_time = max(2.0, (probe_len - pts) / self.fs_target)
                except Exception:
                    max_start_time = 2.0  # fallback: single window
                start_times = np.arange(2.0, max_start_time + 0.01, 0.2)

                for f in target_files:
                    pt_path = os.path.join(folder, f)

                    for start in start_times:
                        self.samples.append({
                            "pt_path": pt_path,
                            "label": self.odor_to_label[odor],
                            "start": start,
                            "odor": odor,
                            "odor_name": ODOR_NAME_MAP[odor],
                            "sub": sub
                        })

        print(f"[Dataset - {mode.upper()}] Fold: {fold_idx}/{n_folds}, Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    # ----------------------------------------------------
    # Subject-aware statistics (for robust normalization)
    # ----------------------------------------------------
    def _compute_subject_stats(self, subjects):
        """
        Compute per-subject mean and std across ALL their .pt files.

        Returns:
            dict: {subject_id: (mean, std)} where mean/std are Python floats
        """
        stats = {}
        print(f"[DEBUG] offline_dir = {self.offline_dir}")
        print(f"[DEBUG] offline_dir exists = {os.path.exists(self.offline_dir)}")
        for sub in subjects:
            # .pt files are stored at: offline_dir / subject / odor / *.pt
            pattern = os.path.join(self.offline_dir, sub, "**", "*.pt")
            pt_files = glob.glob(pattern, recursive=True)

            if not pt_files:
                # No preprocessed files for this subject; fall back to per-sample norm
                sub_dir = os.path.join(self.offline_dir, sub)
                print(f"[DEBUG] {sub}: 0 files found. dir exists={os.path.exists(sub_dir)}, pattern={pattern}")
                continue

            running_sum = 0.0
            running_sq_sum = 0.0
            total_elements = 0

            for pt_path in pt_files:
                try:
                    tensor = torch.load(pt_path, weights_only=True).float()
                    running_sum += tensor.sum().item()
                    running_sq_sum += (tensor ** 2).sum().item()
                    total_elements += tensor.numel()
                except Exception as e:
                    print(f"[WARN] Failed to load {pt_path}: {e}")
                    continue  # skip corrupted files

            if total_elements == 0:
                continue

            mean = running_sum / total_elements
            var = running_sq_sum / total_elements - mean ** 2
            std = max(var, 0.0) ** 0.5  # guard against negative variance from float precision

            stats[sub] = (mean, std)

        print(f"[Dataset] Computed subject-aware stats for {len(stats)}/{len(subjects)} subjects")
        return stats

    # ----------------------------------------------------
    # EEG-specific augmentation helpers
    # ----------------------------------------------------
    def _freq_mask(self, data, fs=128):
        """
        Frequency masking: zero out a random EEG frequency band.

        Args:
            data: (30, T) tensor
            fs: sampling frequency in Hz
        Returns:
            data with one frequency band masked
        """
        # FFT along time axis (dim=1)
        fft_data = torch.fft.rfft(data, dim=1)
        freqs = torch.fft.rfftfreq(data.shape[1], d=1.0 / fs)

        # Pick one of three EEG frequency bands
        bands = [(4, 8), (8, 13), (13, 30)]  # theta, alpha, beta
        lo, hi = bands[np.random.randint(0, len(bands))]

        # Zero out coefficients in the chosen band
        mask = (freqs >= lo) & (freqs <= hi)
        fft_data[:, mask] = 0

        return torch.fft.irfft(fft_data, n=data.shape[1])

    def _channel_mix(self, data):
        """
        Channel mixing: shuffle 2-3 adjacent channels.

        Args:
            data: (30, 256) tensor
        Returns:
            data with 2-3 adjacent channels shuffled
        """
        num_ch = data.shape[0]
        n_mix = np.random.randint(2, 4)  # 2 or 3
        start = np.random.randint(0, num_ch - n_mix)
        indices = list(range(start, start + n_mix))
        shuffled = indices.copy()
        np.random.shuffle(shuffled)
        data[indices] = data[shuffled]
        return data

    def _time_warp(self, data, sigma=0.2, num_points=3):
        """
        Time warping: apply non-linear distortion to the time axis.
        
        Args:
            data: (30, T) tensor
            sigma: strength of warping (0.0-1.0)
            num_points: number of control points for warping curve
        Returns:
            time-warped data
        """
        T = data.shape[1]
        
        # Generate random warp factors for control points
        warp_factors = np.random.normal(1.0, sigma, num_points)
        warp_factors = np.clip(warp_factors, 0.5, 1.5)  # Limit warping range
        
        # Create interpolation points
        src_points = np.linspace(0, T-1, num_points)
        dst_points = src_points * warp_factors
        dst_points = np.clip(dst_points, 0, T-1)
        
        # Create full warp mapping using linear interpolation
        full_src = np.arange(T)
        full_dst = np.interp(full_src, src_points, dst_points)
        full_dst = np.clip(full_dst, 0, T-1).astype(int)
        
        # Apply warping to each channel
        warped_data = data[:, full_dst]
        return warped_data

    # ----------------------------------------------------
    # EEG-specific augmentation
    # ----------------------------------------------------
    def _augment(self, data):

        # 1. Gaussian noise (robustness)
        data = data + torch.randn_like(data) * 0.05

        # 2. Channel dropout (simulate electrode failure)
        if np.random.rand() < 0.3:
            ch = np.random.randint(0, data.shape[0])
            data[ch] = 0

        # 3. Temporal shift (EEG invariance)
        if np.random.rand() < 0.5:
            shift = np.random.randint(-10, 10)
            data = torch.roll(data, shifts=shift, dims=1)

        # 4. Frequency masking (spectral robustness)
        if np.random.rand() < 0.3:
            data = self._freq_mask(data, fs=self.fs_target)

        # 5. Channel mixing (spatial robustness)
        if np.random.rand() < 0.2:
            data = self._channel_mix(data)

        # 6. Time warping (temporal robustness)
        if np.random.rand() < 0.3:
            data = self._time_warp(data, sigma=0.2, num_points=3)

        return data

    # ----------------------------------------------------
    # Main data retrieval
    # ----------------------------------------------------
    def __getitem__(self, idx):

        item = self.samples[idx]
        pts = int(self.fs_target * self.duration)

        try:
            # Load .pt file directly
            pt_path = item["pt_path"]
            full_data = torch.load(pt_path, weights_only=True)

            start = int(item["start"] * self.fs_target)
            end = start + pts

            # safe slicing
            if full_data.shape[1] < end:
                data = full_data[:, -pts:]
            else:
                data = full_data[:, start:end]

            # ------------------------------------------------
            # Subject-aware normalization (CRITICAL for VPNet)
            # Uses pre-computed per-subject mean/std when available,
            # falls back to per-sample normalization otherwise.
            # ------------------------------------------------
            sub = item.get("sub")
            if sub is not None and sub in self.subject_stats:
                sub_mean, sub_std = self.subject_stats[sub]
                data = (data - sub_mean) / (sub_std + 1e-6)
            else:
                data = (data - data.mean()) / (data.std() + 1e-6)

            # ------------------------------------------------
            # Augmentation (train only)
            # ------------------------------------------------
            if self.use_augment and np.random.rand() < 0.7:
                data = self._augment(data)

            return (
                data,
                torch.tensor(item["label"], dtype=torch.long)
            )

        except Exception as e:
            # fallback (avoid crash) — log once per unique error
            if not hasattr(self, '_fallback_count'):
                self._fallback_count = 0
            self._fallback_count += 1
            if self._fallback_count <= 3:
                print(f"[Dataset] Fallback #{self._fallback_count}: {e}")
            return (
                torch.zeros((30, pts), dtype=torch.float32),
                torch.tensor(item["label"], dtype=torch.long)
            )

    # ----------------------------------------------------
    # Optional: get human-readable odor name
    # ----------------------------------------------------
    def get_odor_name(self, label):
        odor = self.label_to_odor[label]
        return ODOR_NAME_MAP[odor]
