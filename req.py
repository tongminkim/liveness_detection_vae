# ============================================
# One-Class TCN-VAE for Landmark Time-Series (POC)
# - Procrustes-based geometric normalization
# - Inputs: pos, vel, acc, slope(angle), slope-rate
# - Light TCN-VAE (C_h=48, C_z=12), KL + Recon
# - Best.pt selection (min val loss)
# - Balanced evaluation (anomaly score based)
# - Extra prints: dataset check, training start, batch-50 progress
# ============================================
import os, glob, sys, time, math, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix


# -------------------
# 0. Config
# -------------------
class CFG:
    data_live = "processed"
    data_fake = "processed_fake"
    save_dir = "runs/poc_vae"
    fps = 301
    T_fixed = 200
    K_lips = 42
    # feature switches
    use_vel = True
    use_acc = True
    use_slope = True
    use_srate = True
    batch_size = 64
    lr = 1e-3
    epochs = 10
    device = "mps"
    seed = 42
    pin_memory = True


# -------------------
# 1. Utils
# -------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_mkdir(p):
    os.makedirs(p, exist_ok=True)
    return p


def l1_loss(x, y):
    return F.l1_loss(x, y)


def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def central_diff(arr):
    """arr: [T, K, D] -> central diff along T, length-preserving"""
    d = np.empty_like(arr)
    d[1:-1] = (arr[2:] - arr[:-2]) / 2.0
    d[0] = arr[1] - arr[0]
    d[-1] = arr[-1] - arr[-2]
    return d


def second_diff(arr):
    """arr: [T, K, D] -> second diff along T, length-preserving"""
    dd = np.empty_like(arr)
    dd[1:-1] = arr[2:] - 2 * arr[1:-1] + arr[:-2]
    dd[0] = arr[1] - arr[0]
    dd[-1] = arr[-1] - arr[-2]
    return dd


def check_dataset_paths():
    live_files = glob.glob(os.path.join(CFG.data_live, "*.npz"))
    fake_files = glob.glob(os.path.join(CFG.data_fake, "*.npz"))

    if not os.path.exists(CFG.data_live) or not os.path.exists(CFG.data_fake):
        print("❌ 데이터 폴더 경로가 존재하지 않습니다.")
        sys.exit(1)
    if len(live_files) == 0:
        print(f"⚠️ Live 폴더에 .npz 없음: {CFG.data_live}")
        sys.exit(1)
    if len(fake_files) == 0:
        print(f"⚠️ Fake 폴더에 .npz 없음: {CFG.data_fake}")
        sys.exit(1)

    # ✅ 데이터가 있는 것이 확인되면 print 1회
    print(f"✅ 데이터 확인: live={len(live_files)}개, fake={len(fake_files)}개\n")
    return len(live_files), len(fake_files)


# ---- Procrustes (Umeyama) similarity transform ----
def umeyama_similarity(X, Y):
    """
    X, Y: [N,2] (float)  → find s,R,t s.t. Y ≈ s R X + t
    returns s, R(2x2), t(2,)
    """
    Xc = X.mean(axis=0)
    Yc = Y.mean(axis=0)
    X0 = X - Xc
    Y0 = Y - Yc
    var = (X0**2).sum() / X0.shape[0] + 1e-12
    U, S, Vt = np.linalg.svd((Y0.T @ X0) / X0.shape[0])
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    s = S.sum() / var
    t = Yc - s * (R @ Xc)
    return s, R, t


def apply_similarity(P, s, R, t):
    """P: [M,2]"""
    return (s * (P @ R.T)) + t


# -------------------
# 2. Dataset (with geometric normalization)
# -------------------
class LipDataset(Dataset):
    def __init__(self, root, T_fixed=200, K_lips=42, ref_frames=10):
        self.files = sorted(glob.glob(os.path.join(root, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No npz in {root}")
        self.T_fixed, self.K_lips, self.ref_frames = T_fixed, K_lips, ref_frames

        # Precompute reference mean shape (from first file)
        d0 = np.load(self.files[0])
        lips0 = np.concatenate([d0["lips_outer"], d0["lips_inner"]], axis=1)[
            :, : self.K_lips, :
        ]  # [T,K,2]
        h0, w0 = d0["size"]
        lips0[..., 0] /= w0 + 1e-8
        lips0[..., 1] /= h0 + 1e-8
        T0 = lips0.shape[0]
        if T0 > self.T_fixed:
            s = (T0 - self.T_fixed) // 2
            lips0 = lips0[s : s + self.T_fixed]
        elif T0 < self.T_fixed:
            lips0 = np.pad(lips0, ((0, self.T_fixed - T0), (0, 0), (0, 0)), mode="edge")
        ref = lips0[: min(self.ref_frames, self.T_fixed)]
        self.ref_shape = ref.mean(axis=0)  # [K,2]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        lips = np.concatenate([d["lips_outer"], d["lips_inner"]], axis=1)[
            :, : self.K_lips, :
        ]  # [T,K,2]

        # 1) normalize by image size
        h, w = d["size"]
        lips[..., 0] /= w + 1e-8
        lips[..., 1] /= h + 1e-8

        # 2) crop/pad to T_fixed
        T = lips.shape[0]
        if T > self.T_fixed:
            s = (T - self.T_fixed) // 2
            lips = lips[s : s + self.T_fixed]
        elif T < self.T_fixed:
            lips = np.pad(lips, ((0, self.T_fixed - T), (0, 0), (0, 0)), mode="edge")

        # 3) Procrustes-based similarity normalization per frame
        lips_norm = np.empty_like(lips)
        for t in range(CFG.T_fixed):
            X = lips[t]  # [K,2]
            Y = self.ref_shape  # [K,2]
            try:
                s, R, tt = umeyama_similarity(X, Y)
                lips_norm[t] = apply_similarity(X, s, R, tt)
            except np.linalg.LinAlgError:
                lips_norm[t] = X  # fallback

        # 4) build features
        feats = []
        feats.append(lips_norm)  # position

        if CFG.use_vel:
            vel = central_diff(lips_norm) * CFG.fps
            feats.append(vel)
        if CFG.use_acc:
            acc = second_diff(lips_norm) * (CFG.fps**2)
            feats.append(acc)

        if CFG.use_slope or CFG.use_srate:
            v = np.roll(lips_norm, -1, axis=1) - lips_norm  # [T,K,2]
            angles = np.arctan2(v[..., 1], v[..., 0])  # [T,K]
            angles_unwrap = np.unwrap(angles, axis=0)[..., None]  # [T,K,1]
            if CFG.use_slope:
                feats.append(angles_unwrap)
            if CFG.use_srate:
                ang_rate = central_diff(angles_unwrap) * CFG.fps
                feats.append(ang_rate)

        x_arr = np.concatenate(feats, axis=2)  # [T,K,Fdim]
        x = torch.tensor(x_arr).permute(2, 1, 0).reshape(-1, CFG.T_fixed)  # [C,T]
        return x.float()


def calc_C_in():
    # per landmark feature dim:
    fdim = 2  # pos
    if CFG.use_vel:
        fdim += 2
    if CFG.use_acc:
        fdim += 2
    if CFG.use_slope:
        fdim += 1
    if CFG.use_srate:
        fdim += 1
    return CFG.K_lips * fdim


# -------------------
# 3. Model (Light TCN-VAE: C_h=48, C_z=12)
# -------------------
class TCNBlock(nn.Module):
    def __init__(self, C_in, C_out, d):
        super().__init__()
        self.c = nn.Conv1d(C_in, C_out, kernel_size=3, padding=d, dilation=d)
        self.gn = nn.GroupNorm(1, C_out)
        self.act = nn.SiLU()
        self.res = nn.Conv1d(C_in, C_out, 1) if C_in != C_out else nn.Identity()

    def forward(self, x):
        y = self.act(self.gn(self.c(x)))
        return y + self.res(x)


class TCNEncoder(nn.Module):
    def __init__(self, C_in, C_h=48, C_z=12):
        super().__init__()
        self.inp = nn.Conv1d(C_in, C_h, 1)
        self.tcn = nn.Sequential(
            TCNBlock(C_h, C_h, d=1),
            TCNBlock(C_h, C_h, d=2),
            TCNBlock(C_h, C_h, d=4),
        )
        self.out = nn.Conv1d(C_h, C_z * 2, 1)

    def forward(self, x):
        h = self.tcn(self.inp(x))
        mu, logvar = torch.chunk(self.out(h), 2, dim=1)
        return mu, logvar


class TCNDecoder(nn.Module):
    def __init__(self, C_z=12, C_h=48, C_out=128):
        super().__init__()
        self.fc = nn.Conv1d(C_z, C_h, 1)
        self.tcn = nn.Sequential(
            TCNBlock(C_h, C_h, d=4),
            TCNBlock(C_h, C_h, d=2),
            TCNBlock(C_h, C_h, d=1),
        )
        self.out = nn.Conv1d(C_h, C_out, 1)

    def forward(self, z):
        h = self.tcn(self.fc(z))
        return self.out(h)


class SimpleVAE(nn.Module):
    def __init__(self, C_in, C_h=48, C_z=12):
        super().__init__()
        self.enc = TCNEncoder(C_in, C_h, C_z)
        self.dec = TCNDecoder(C_z, C_h, C_in)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparam(mu, logvar)
        xhat = self.dec(z)
        return xhat, mu, logvar


# -------------------
# 4. Train / Validate (best.pt)
# -------------------
def train_epoch(model, loader, opt, epoch):
    model.train()
    tot = rec = klv = 0.0
    st = time.time()
    nb = len(loader)

    # 🔊 에폭 시작 시점에 한 번 출력 (학습 시작 안내)
    if epoch == 1:
        print(
            f"🚀 Training started: epochs={CFG.epochs}, batches/epoch={nb}, device={CFG.device}"
        )

    for i, x in enumerate(loader):
        x = x.to(CFG.device)
        xhat, mu, logvar = model(x)
        rec_loss = l1_loss(xhat, x)
        kl = kl_loss(mu, logvar)
        loss = rec_loss + kl

        opt.zero_grad()
        loss.backward()
        opt.step()

        tot += loss.item()
        rec += rec_loss.item()
        klv += kl.item()

        # 🔊 배치 50개마다 진행 상황 + 남은 시간 출력
        if (i + 1) % 50 == 0 or (i + 1) == nb:
            elapsed = time.time() - st
            avg_b = elapsed / (i + 1)
            eta = (nb - (i + 1)) * avg_b
            print(
                f"[Ep {epoch:02d} | {i + 1:04d}/{nb}] "
                f"Total={tot / (i + 1):.4f} | Recon={rec / (i + 1):.4f} | KL={klv / (i + 1):.4f} "
                f"| Elapsed={elapsed / 60:.1f}m | ETA={eta / 60:.1f}m"
            )
    return tot / nb


@torch.no_grad()
def validate(model, loader):
    model.eval()
    tot = rec = klv = 0.0
    n = 0
    for x in loader:
        x = x.to(CFG.device)
        xhat, mu, logvar = model(x)
        rec_loss = l1_loss(xhat, x)
        kl = kl_loss(mu, logvar)
        loss = rec_loss + kl
        b = x.size(0)
        tot += loss.item() * b
        rec += rec_loss.item() * b
        klv += kl.item() * b
        n += b
    return (tot / max(1, n), rec / max(1, n), klv / max(1, n))


# -------------------
# 5. Scoring & Evaluation
# -------------------
@torch.no_grad()
def anomaly_scores(model, loader):
    """per-sample anomaly score = mean L1 + KL"""
    model.eval()
    out = []
    for x in loader:
        x = x.to(CFG.device)
        xhat, mu, logvar = model(x)
        rec = F.l1_loss(xhat, x, reduction="none").mean(dim=[1, 2])
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(dim=[1, 2])
        out.append((rec + kl).cpu().numpy())
    return np.concatenate(out)


def metrics_from_cm(cm):
    TN, FP, FN, TP = cm.ravel()
    acc = (TP + TN) / (TP + TN + FP + FN + 1e-12)
    tpr = TP / (TP + FN + 1e-12)
    fpr = FP / (FP + TN + 1e-12)
    prec = TP / (TP + FP + 1e-12)
    f1 = 2 * prec * tpr / (prec + tpr + 1e-12)
    return TN, FP, FN, TP, acc, tpr, fpr, prec, f1


def search_threshold_youden(live_scores, fake_scores, qmin=0.8, qmax=0.999, steps=400):
    y_true = np.concatenate([np.zeros_like(live_scores), np.ones_like(fake_scores)])
    all_sc = np.concatenate([live_scores, fake_scores])
    cand = np.quantile(live_scores, np.linspace(qmin, qmax, steps))
    best_th, best_val = None, -1
    for th in cand:
        y_pred = (all_sc > th).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        TN, FP, FN, TP = cm.ravel()
        TPR = TP / (TP + FN + 1e-12)
        FPR = FP / (FP + TN + 1e-12)
        val = TPR - FPR
        if val > best_val:
            best_val, best_th = val, th
    return float(best_th)


if __name__ == "__main__":
    # -------------------
    # 6. Main
    # -------------------
    seed_everything(CFG.seed)
    maybe_mkdir(CFG.save_dir)
    live_n, fake_n = check_dataset_paths()  # 🔊 여기서 데이터 확인 메시지 1회 출력

    live_ds = LipDataset(CFG.data_live, CFG.T_fixed, CFG.K_lips)
    fake_ds = LipDataset(CFG.data_fake, CFG.T_fixed, CFG.K_lips)

    tr_len = int(0.9 * len(live_ds))
    va_len = len(live_ds) - tr_len
    train_ds, val_ds = random_split(
        live_ds, [tr_len, va_len], generator=torch.Generator().manual_seed(CFG.seed)
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        pin_memory=CFG.pin_memory,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        pin_memory=CFG.pin_memory,
    )
    fake_dl = DataLoader(
        fake_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        pin_memory=CFG.pin_memory,
    )

    print(
        f"✅ Split: Train={len(train_ds)} | Val={len(val_ds)} | Fake={len(fake_ds)}\n"
    )

    C_in = CFG.K_lips * (
        2
        + (2 if CFG.use_vel else 0)
        + (2 if CFG.use_acc else 0)
        + (1 if CFG.use_slope else 0)
        + (1 if CFG.use_srate else 0)
    )
    model = SimpleVAE(C_in=C_in, C_h=48, C_z=12).to(CFG.device)
    opt = torch.optim.Adam(model.parameters(), lr=CFG.lr)

    best_val = float("inf")
    best_path = os.path.join(CFG.save_dir, "best.pt")

    for ep in range(1, CFG.epochs + 1):
        tr_loss = train_epoch(model, train_dl, opt, ep)
        va_total, va_rec, va_kl = validate(model, val_dl)
        print(
            f"Epoch {ep} ▶ Train={tr_loss:.4f} | ValTot={va_total:.4f} | ValRec={va_rec:.4f} | ValKL={va_kl:.4f}"
        )
        if va_total < best_val:
            best_val = va_total
            torch.save(model.state_dict(), best_path)
            print(f"🌟 New best saved at {best_path} (val={best_val:.4f})")

    print("\n[Load best.pt for evaluation]")
    model.load_state_dict(torch.load(best_path, map_location=CFG.device))
    model.to(CFG.device).eval()

    live_scores = anomaly_scores(model, val_dl)
    fake_scores = anomaly_scores(model, fake_dl)

    # Balanced eval
    N = min(len(live_scores), len(fake_scores))
    rng = np.random.default_rng(CFG.seed)
    li = rng.choice(len(live_scores), N, replace=False)
    fi = rng.choice(len(fake_scores), N, replace=False)
    live_bal, fake_bal = live_scores[li], fake_scores[fi]
    y_true = np.concatenate([np.zeros(N, int), np.ones(N, int)])

    # Threshold 1) Youden
    th_y = search_threshold_youden(live_bal, fake_bal)
    y_pred_y = np.concatenate(
        [(live_bal > th_y).astype(int), (fake_bal > th_y).astype(int)]
    )
    cm_y = confusion_matrix(y_true, y_pred_y, labels=[0, 1])
    TN, FP, FN, TP, acc, tpr, fpr, prec, f1 = metrics_from_cm(cm_y)
    print("\n[Balanced • Youden threshold]")
    print(cm_y)
    print(
        f"Acc={acc:.4f} | TPR={tpr:.4f} | FPR={fpr:.4f} | Prec={prec:.4f} | F1={f1:.4f} | thr={th_y:.6f}"
    )

    # Threshold 2) IQR (live_bal)
    Q1, Q3 = np.percentile(live_bal, [25, 75])
    iqr = Q3 - Q1
    th_iqr = Q3 + 1.5 * iqr
    y_pred_iqr = np.concatenate(
        [(live_bal > th_iqr).astype(int), (fake_bal > th_iqr).astype(int)]
    )
    cm_iqr = confusion_matrix(y_true, y_pred_iqr, labels=[0, 1])
    TN, FP, FN, TP, acc, tpr, fpr, prec, f1 = metrics_from_cm(cm_iqr)
    print("\n[Balanced • IQR threshold]")
    print(cm_iqr)
    print(
        f"Acc={acc:.4f} | TPR={tpr:.4f} | FPR={fpr:.4f} | Prec={prec:.4f} | F1={f1:.4f} | thr={th_iqr:.6f}"
    )
