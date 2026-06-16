import os
import neo
import mne
import numpy as np
import torch
from tqdm import tqdm


def preprocess_all_data():
    """
    Offline preprocessing pipeline optimized for VPNet EEG classification.

    Key principles:
    - Preserve raw temporal structure
    - Ensure consistent amplitude scaling across subjects/trials
    - Avoid feature engineering
    - Keep preprocessing deterministic (no randomness here)
    """

    # ----------------------------------------------------
    # 1. Paths & settings
    # ----------------------------------------------------
    root_dir = os.getenv(
        "OLF_RAW_ROOT",
        r"D:\\OlfactoryEEG1\\Olfactory EEG data set induced by different odor types"
    )
    save_dir = os.getenv(
        "OLF_PREPROCESS_OUT",
        r"D:\OlfactoryEEG1\Processed_EEG_Tensors_fs256_h100_no_file_std"
    )
    fs_target = int(os.getenv("OLF_PREPROCESS_FS", "256"))
    l_freq = float(os.getenv("OLF_PREPROCESS_L_FREQ", "0.5"))
    h_freq = float(os.getenv("OLF_PREPROCESS_H_FREQ", "100.0"))

    subjects = [f"Sub. {i}" for i in range(1, 12)]
    odors = [chr(i) for i in range(ord('A'), ord('M') + 1)]

    # ----------------------------------------------------
    # 2. Collect file list
    # ----------------------------------------------------
    file_tasks = []

    for sub in subjects:
        for odor in odors:
            folder_path = os.path.join(root_dir, sub, odor)

            if not os.path.exists(folder_path):
                continue

            for f in os.listdir(folder_path):
                if f.lower().endswith('.ns2'):
                    file_tasks.append((sub, odor, f))

    print(f"Total files to process: {len(file_tasks)}")

    # ----------------------------------------------------
    # 3. Processing loop
    # ----------------------------------------------------
    for sub, odor, f in tqdm(file_tasks, desc="Processing Offline"):

        original_path = os.path.join(root_dir, sub, odor, f)

        # Output path
        target_folder = os.path.join(save_dir, sub, odor)
        os.makedirs(target_folder, exist_ok=True)

        target_path = os.path.join(
            target_folder,
            f.replace('.ns2', '.pt').replace('.NS2', '.pt')
        )

        # Skip if already processed
        if os.path.exists(target_path):
            continue

        try:
            # ----------------------------------------------------
            # 3.1 Load NS2 signal
            # ----------------------------------------------------
            reader = neo.io.BlackrockIO(filename=original_path)
            block = reader.read_block()

            raw_signal = np.array(block.segments[0].analogsignals[0]).T
            fs_original = float(block.segments[0].analogsignals[0].sampling_rate)

            # Keep EEG channels only
            raw_signal = raw_signal[:30, :]

            # ----------------------------------------------------
            # 3.2 Convert to MNE format
            # ----------------------------------------------------
            info = mne.create_info(
                ch_names=30,
                sfreq=fs_original,
                ch_types='eeg'
            )

            raw = mne.io.RawArray(raw_signal, info, verbose=False)

            # ----------------------------------------------------
            # 3.3 Re-referencing (critical for EEG stability)
            # ----------------------------------------------------
            raw.set_eeg_reference('average', projection=False, verbose=False)

            # ----------------------------------------------------
            # 3.4 Bandpass filtering (keep neural information)
            # ----------------------------------------------------
            effective_h_freq = min(h_freq, fs_original / 2.0 - 1e-3)
            if effective_h_freq <= l_freq:
                raise ValueError(
                    f"Invalid filter band for {original_path}: "
                    f"fs_original={fs_original}, l_freq={l_freq}, h_freq={h_freq}"
                )

            raw.filter(
                l_freq=l_freq,
                h_freq=effective_h_freq,
                fir_design='firwin',
                verbose=False
            )

            # ----------------------------------------------------
            # 3.5 Downsampling
            # ----------------------------------------------------
            raw.resample(fs_target, verbose=False)

            data = raw.get_data()  # shape: [30, T]

            # ----------------------------------------------------
            # 3.6 Mild centering (safe for EEG)
            # Keep amplitude/energy differences; subject-level normalization
            # is applied later in dataloader.py.
            # ----------------------------------------------------
            data = data - np.mean(data, axis=1, keepdims=True)

            # ----------------------------------------------------
            # 3.7 Safe clipping (prevent extreme artifacts)
            # ----------------------------------------------------
            data = np.clip(data, -100, 100)

            # ----------------------------------------------------
            # 3.8 Save tensor
            # ----------------------------------------------------
            torch.save(
                torch.tensor(data, dtype=torch.float32),
                target_path
            )

        except Exception as e:
            print(f"[ERROR] {original_path}: {e}")

    print("\n All data preprocessed and saved to:", save_dir)


if __name__ == "__main__":
    preprocess_all_data()
