"""
Split test_balanced_npz into train/test (50/50)
- Fake: audio-driven, dfl, dffs only (excluding fo, fsgan)
- Random seed 42 for reproducibility
"""

import os
import glob
import random
import json

# Paths
TEST_BASE = "/home/elicer/liveness_detection/model1/test_balanced_npz"
REAL_BASE = os.path.join(TEST_BASE, "real")
FAKE_BASE = os.path.join(TEST_BASE, "fake")

# Fake directories to use (excluding fo and fsgan)
FAKE_DIRS = ["audio-driven", "dfl", "dffs"]

# Random seed
RANDOM_SEED = 42

# Output file
OUTPUT_FILE = "/home/elicer/liveness_detection/model1/data_split.json"


def collect_files(base_dir, subdirs=None):
    """Collect all npz files"""
    files = []
    if subdirs:
        for subdir in subdirs:
            pattern = os.path.join(base_dir, subdir, "**", "*.npz")
            files.extend(glob.glob(pattern, recursive=True))
    else:
        pattern = os.path.join(base_dir, "**", "*.npz")
        files.extend(glob.glob(pattern, recursive=True))
    return sorted(files)  # Sort for reproducibility


def main():
    print("="*80)
    print("Splitting test_balanced_npz into Stage2-Train / Final-Test (50/50)")
    print("="*80)

    # Collect files
    print("\n1. Collecting files...")
    fake_files = collect_files(FAKE_BASE, FAKE_DIRS)
    real_files = collect_files(REAL_BASE)

    print(f"   Fake files (audio-driven, dfl, dffs): {len(fake_files)}")
    print(f"   Real files: {len(real_files)}")

    # Set random seed
    random.seed(RANDOM_SEED)

    # Split fake files (50/50)
    random.shuffle(fake_files)
    n_fake_train = len(fake_files) // 2
    fake_train = fake_files[:n_fake_train]
    fake_test = fake_files[n_fake_train:]

    # Split real files (50/50)
    random.shuffle(real_files)
    n_real_train = len(real_files) // 2
    real_train = real_files[:n_real_train]
    real_test = real_files[n_real_train:]

    print("\n2. Split results:")
    print(f"   Stage2 Train - Real: {len(real_train)}, Fake: {len(fake_train)}")
    print(f"   Final Test   - Real: {len(real_test)}, Fake: {len(fake_test)}")

    # Save split
    split_data = {
        "random_seed": RANDOM_SEED,
        "fake_dirs_used": FAKE_DIRS,
        "fake_dirs_excluded": ["fo", "fsgan"],
        "stage2_train": {
            "real": real_train,
            "fake": fake_train
        },
        "final_test": {
            "real": real_test,
            "fake": fake_test
        },
        "counts": {
            "stage2_train_real": len(real_train),
            "stage2_train_fake": len(fake_train),
            "final_test_real": len(real_test),
            "final_test_fake": len(fake_test)
        }
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(split_data, f, indent=2)

    print(f"\n3. Saved split to: {OUTPUT_FILE}")
    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()
