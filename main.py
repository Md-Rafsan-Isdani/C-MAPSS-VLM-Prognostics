# %% [markdown]
# # VLM-Assisted Prognostic Decision Framework for RUL Prediction
# ### NASA C-MAPSS Turbofan Engine Degradation (FD001–FD004)
#
# This notebook implements a hybrid prognostic framework that combines numerical ML/DL
# remaining-useful-life (RUL) models with qualitative visual evidence from a vision-language model
# (VLM). The pipeline runs on all four C-MAPSS subsets and writes results to
# `/content/artifacts/<SUBSET>/`, with a cross-subset summary in `/content/artifacts/summary/`.
#
# **Pipeline**
# 1. **Numerical branch** — low-variance sensor removal, regime-wise normalization fit on training
#    data, piecewise-linear RUL capping, sliding windows, and an engine-level train/validation split.
#    Models: Linear Regression, Random Forest, XGBoost, BiLSTM, TCN, MS-TCN, MS-TCN+BiLSTM, Transformer.
# 2. **Visual branch** — per-engine sensor-trend figures rendered from the raw values of all 21
#    sensors and interpreted by Qwen2.5-VL-7B-Instruct, which returns structured qualitative evidence.
# 3. **Deterministic fusion** — the final RUL is computed in Python with weights proportional to
#    1/validation RMSE, MAD-based outlier exclusion, and a bounded (±`VLM_ADJUSTMENT_BOUND`)
#    adjustment from the VLM's qualitative RUL band.
# 4. **Evidence analysis** — Qwen2.5-7B-Instruct reviews the model predictions, VLM evidence, and
#    fusion result and returns structured commentary with an `accept` / `caution` / `reject`
#    recommendation. Its output schema contains no numeric RUL field.
#
# **Requirements.** A CUDA GPU (L4 or A100 recommended). Both language models are loaded once in
# 4-bit precision and reused across subsets. A full run over all 707 test engines takes several
# hours. With `SKIP_ALREADY_COMPLETED_SUBSETS = True`, completed subsets are loaded from disk on
# re-run, so an interrupted session can be resumed.


# %% [markdown]
# ## 0. Environment setup
#
# Installs `transformers`, `accelerate`, `bitsandbytes`, and `qwen-vl-utils`. The preinstalled
# `torch` and scientific stack (numpy, pandas, scipy, scikit-learn, etc.) are left unchanged to avoid
# binary incompatibilities; missing packages are installed without upgrading existing ones.
#
# If the numpy/scipy check fails, use *Runtime → Disconnect and delete runtime* and restart on a
# fresh VM.


# %%
# ------------------------------------------------------------------
# Early sanity check. A broken numpy/scipy ABI mismatch in THIS runtime (from numpy ever being
# force-upgraded in this same VM) crashes deep inside unrelated imports later with a confusing
# 10+ frame traceback. Catch it here, immediately, with a clear diagnosis instead.
#
# IMPORTANT: if this fires, "Runtime -> Restart session" will NOT fix it -- that only restarts
# the Python process, it does not undo packages already installed on this VM's disk. You need
# "Runtime -> Disconnect and delete runtime", then reconnect (a genuinely fresh VM).
# ------------------------------------------------------------------
try:
    import numpy as _np
    import scipy as _scipy  # noqa: F401 -- the import alone is what triggers the ABI check
    del _np, _scipy
    print("[TITLE] Environment check\nnumpy/scipy import OK — no ABI mismatch detected in this runtime.")
except AttributeError as _e:
    raise RuntimeError(
        "\n\nThis runtime's numpy/scipy installation is broken (an ABI mismatch between numpy "
        "and packages compiled against it, e.g. scipy/pandas/scikit-learn) — this happens once "
        "numpy has been force-upgraded in this VM, and it persists across 'Restart session' "
        "because that only restarts the Python process, not what's installed on disk.\n\n"
        ">>> Fix: Runtime -> Disconnect and delete runtime, then reconnect, then re-run this "
        "notebook from the top on the fresh VM. <<<\n"
    ) from _e

import subprocess, sys

def pip_install(*packages, quiet=True, upgrade=True):
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("-U")
    if quiet:
        cmd.append("-q")
    cmd.extend(packages)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"!! pip install failed for {packages}:")
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
    return result.returncode == 0

# These four are genuinely newer than what Colab ships (Qwen2.5-VL needs transformers>=4.49) —
# safe/necessary to upgrade explicitly. None of them pin/force a numpy version.
pip_install("transformers>=4.49.0", "accelerate>=1.0.0")
pip_install("bitsandbytes>=0.46.1", quiet=False)   # verbose: this is the package that most often fails silently
pip_install("qwen-vl-utils[decord]")

# The rest of the scientific stack: install ONLY if missing, NEVER with -U, so an already-present,
# Colab-tested numpy (and everything compiled against it) is never touched.
_stack = {"numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib", "seaborn": "seaborn",
          "scipy": "scipy", "sklearn": "scikit-learn", "xgboost": "xgboost",
          "statsmodels": "statsmodels", "tqdm": "tqdm"}
_missing = []
for _import_name, _pip_name in _stack.items():
    try:
        __import__(_import_name)
    except ImportError:
        _missing.append(_pip_name)
if _missing:
    print(f"Installing packages missing from this runtime (no -U — won't touch anything already "
          f"present): {_missing}")
    pip_install(*_missing, upgrade=False)
else:
    print("numpy/pandas/matplotlib/seaborn/scipy/scikit-learn/xgboost/statsmodels/tqdm are all "
          "already present in this runtime — left untouched.")

# Verify the quantization backend actually imports in THIS kernel.
import importlib

def _bnb_version():
    try:
        import bitsandbytes as bnb
        importlib.reload(bnb)
        return bnb.__version__
    except ImportError:
        return None

bnb_version = _bnb_version()
if bnb_version is None:
    print("bitsandbytes not importable after the first install — retrying with "
          "--force-reinstall --no-cache-dir ...")
    ok = pip_install("--force-reinstall", "--no-cache-dir", "bitsandbytes>=0.46.1", quiet=False)
    bnb_version = _bnb_version()

if bnb_version:
    print(f"bitsandbytes OK (v{bnb_version})")
else:
    print("bitsandbytes STILL not importable. Scroll up to the verbose pip output above for the "
          "actual error — that message determines the real fix, not a runtime restart.")

print("Setup complete.")


# %%
import os, io, re, json, math, glob, time, random, textwrap, zipfile, warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm.auto import tqdm
from IPython.display import Image, display

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

SEED = 191
random.seed(SEED)
np.random.seed(SEED)

import torch
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # Deterministic cuDNN kernels (a modest, one-time perf cost, not a reason to skip it per the
    # notebook's reproducibility requirement) -- combined with do_sample=False for every VLM/LLM
    # generation call, this makes the whole pipeline reproducible run-to-run on the same hardware.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if DEVICE == "cpu":
    print("WARNING: no GPU detected. Runtime -> Change runtime type -> GPU before running the "
          "VLM/LLM cells — two real 7B-class models in 4-bit need CUDA.")

def banner(title, char="="):
    # Used everywhere so every block of output is unambiguously titled.
    line = char * 78
    print(f"\n{line}\n{title}\n{line}")

banner("[TITLE] Runtime")
print("Device:", DEVICE, "| torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# %%
# ------------------------------------------------------------------
# GLOBAL CONFIG — the single place you change things
# ------------------------------------------------------------------
banner("[TITLE] Configuration")

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]     # processed automatically, in this order
SKIP_ALREADY_COMPLETED_SUBSETS = True              # resume-safe: skip a subset whose run_summary.json already exists

DATA_ROOT     = "/content/cmapss"
GRAPH_ROOT    = "/content/graphs"
ARTIFACT_ROOT = "/content/artifacts"
for d in [DATA_ROOT, GRAPH_ROOT, ARTIFACT_ROOT]:
    os.makedirs(d, exist_ok=True)

# Standard piecewise-linear RUL cap used across the RUL literature (Heimes 2008; Li et al. 2018)
RUL_CAP = 125

# Sliding-window length for the sequence models (cycles)
WINDOW_SIZE = 30
WINDOW_STRIDE = 1

# How many TRAIN engines (per subset) also get a rendered graph, purely for visual sanity-
# checking -- NOT passed through the VLM and NOT used in any metric. Set to 0 to skip, -1 for all.
N_TRAIN_ENGINES_FOR_VLM = 10

# Quick-validation mode: cap how many TEST engines per subset go through the VLM/fusion/eval path.
# None = full run (all test engines, as used for real results). Set to e.g. 15 to sanity-check a
# prompt/graph change cheaply (minutes, not hours) before committing to a full multi-hour re-run.
# Runs in this mode are clearly marked in the saved run_summary.json (quick_validation: true) and
# SHOULD NOT be reported as results -- they exist only to validate a change before the real run.
MAX_TEST_ENGINES_PER_SUBSET = None

USE_4BIT = True               # 4-bit quantization for the VLM / fusion LLM (recommended on L4/T4)
MAX_LLM_RETRIES = 3            # re-prompt the SAME real model this many times on invalid JSON before giving up

VLM_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
FUSION_LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"   # separate text-only agent; set to VLM_MODEL_ID to reuse one model's text tower

ML_DL_MODEL_LIST = ["LinearRegression", "RandomForest", "XGBoost",
                     "BiLSTM", "TCN", "MS-TCN", "MS-TCN+BiLSTM", "Transformer"]

CONFIG_SNAPSHOT = {
    "subsets": SUBSETS, "rul_cap": RUL_CAP, "window_size": WINDOW_SIZE, "window_stride": WINDOW_STRIDE,
    "n_train_engines_for_vlm": N_TRAIN_ENGINES_FOR_VLM,
    "max_test_engines_per_subset": MAX_TEST_ENGINES_PER_SUBSET, "use_4bit": USE_4BIT,
    "max_llm_retries": MAX_LLM_RETRIES, "vlm_model_id": VLM_MODEL_ID,
    "evidence_analysis_llm_model_id": FUSION_LLM_MODEL_ID,
    "ml_dl_model_list": ML_DL_MODEL_LIST, "seed": SEED,
    # --- methodology flags, recorded so a saved run is self-describing ---
    "final_rul_computed_by": "deterministic constrained_fusion() in Python (never LLM-generated)",
    "fusion_weighting": "w_i proportional to 1/validation_RMSE_i, MAD outlier down-weighting, "
                         "w_i >= 0 and sum(w_i) == 1",
    "vlm_sees_all_sensors": True,
    "terminal_phase_terminology": "terminal_phase_start_cycle (operational proxy; NOT ground-truth "
                                   "physical degradation onset)",
    "confidence_semantics": "deterministic reliability heuristic in [0,1]; NOT a calibrated probability",
    "deterministic_decoding": "do_sample=False for every VLM/LLM generation call",
    "torch_deterministic_cudnn": bool(torch.cuda.is_available()),
}
os.makedirs(os.path.join(ARTIFACT_ROOT, "summary"), exist_ok=True)
with open(os.path.join(ARTIFACT_ROOT, "summary", "config.json"), "w") as f:
    json.dump(CONFIG_SNAPSHOT, f, indent=2)

if MAX_TEST_ENGINES_PER_SUBSET is not None:
    print(f"!! QUICK-VALIDATION MODE: only the first {MAX_TEST_ENGINES_PER_SUBSET} test engines per "
          f"subset will be processed. Set MAX_TEST_ENGINES_PER_SUBSET = None above for a real run.")
print(json.dumps(CONFIG_SNAPSHOT, indent=2))


# %% [markdown]
# ## 1. Data ingestion
#
# Upload the NASA C-MAPSS archive (`CMAPSSData.zip`) containing `train_FD00X.txt`,
# `test_FD00X.txt`, and `RUL_FD00X.txt` for X = 1–4. Set `UPLOAD_ZIP = False` to load the archive
# from Google Drive instead.


# %%
banner("[TITLE] Data ingestion — upload & extract")

# --- Option A: direct upload ---
UPLOAD_ZIP = True   # set False if using Option B (Google Drive)

if UPLOAD_ZIP:
    from google.colab import files
    print("Select the C-MAPSS zip file (e.g. CMAPSSData.zip)...")
    uploaded = files.upload()
    ZIP_PATH = list(uploaded.keys())[0]
else:
    # --- Option B: Google Drive ---
    from google.colab import drive
    drive.mount("/content/drive")
    ZIP_PATH = "/content/drive/MyDrive/CMAPSSData.zip"   # <-- edit to your path

assert os.path.exists(ZIP_PATH), f"Zip not found at {ZIP_PATH}"

with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    zf.extractall(DATA_ROOT)

# The official archive sometimes nests files one directory deeper — flatten if so.
found_txt = glob.glob(os.path.join(DATA_ROOT, "**", "*.txt"), recursive=True)
for fp in found_txt:
    target = os.path.join(DATA_ROOT, os.path.basename(fp))
    if fp != target and not os.path.exists(target):
        os.rename(fp, target)

print(f"Extracted {len(found_txt)} files into {DATA_ROOT}")
found_names = sorted(os.path.basename(f) for f in found_txt)
print(found_names)

for _s in SUBSETS:
    for _kind in ["train", "test", "RUL"]:
        _fn = f"{_kind}_{_s}.txt"
        if _fn not in found_names:
            print(f"  !! WARNING: {_fn} not found in the extracted archive — {_s} will fail to load.")


# %%
# ------------------------------------------------------------------
# C-MAPSS schema + loader (subset-parameterized; no globals depend on a single subset)
# ------------------------------------------------------------------
BASE_COLS = ["unit", "cycle", "op1", "op2", "op3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
ALL_COLS = BASE_COLS + SENSOR_COLS

# The VLM branch must see ALL 21 original sensors, unfiltered -- it is a separate, minimally
# processed path from the ML/DL branch (which drops near-constant sensors per subset via
# drop_low_variance_sensors below, fit on that subset's training data). VLM_SENSORS is used
# EVERYWHERE a graph is rendered for the VLM; ACTIVE_SENSORS (computed per-subset later) is used
# ONLY for the numerical models. This separation is load-bearing: it is what makes it factually
# true, not just a design intention, that the VLM sees the complete raw sensor set rather than a
# version already filtered for the numerical branch.
VLM_SENSORS = SENSOR_COLS.copy()

# Number of operating regimes per subset (FD001/FD003: 1 regime; FD002/FD004: 6 regimes)
N_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
# Number of fault modes per subset (informational only)
N_FAULT_MODES = {"FD001": 1, "FD002": 1, "FD003": 2, "FD004": 2}

def load_cmapss(subset, root=DATA_ROOT):
    def _read(path):
        return pd.read_csv(path, sep=r"\s+", header=None, names=ALL_COLS, engine="python")

    train = _read(os.path.join(root, f"train_{subset}.txt"))
    test  = _read(os.path.join(root, f"test_{subset}.txt"))
    rul   = pd.read_csv(os.path.join(root, f"RUL_{subset}.txt"), sep=r"\s+", header=None,
                         names=["RUL"], engine="python")
    rul["unit"] = np.arange(1, len(rul) + 1)
    return train, test, rul

banner("[TITLE] Dataset overview — all four subsets")
_overview_rows = []
for _s in SUBSETS:
    _tr, _te, _ru = load_cmapss(_s)
    _overview_rows.append({"subset": _s, "train_engines": _tr.unit.nunique(),
                            "test_engines": _te.unit.nunique(), "train_rows": len(_tr),
                            "test_rows": len(_te), "regimes": N_REGIMES[_s],
                            "fault_modes": N_FAULT_MODES[_s]})
    del _tr, _te, _ru
overview_df = pd.DataFrame(_overview_rows).set_index("subset")
print(overview_df)
print(f"\nTotal test engines across all four subsets: {overview_df['test_engines'].sum()} "
      f"(this is what the VLM + fusion LLM will each run inference on, once per engine).")


# %% [markdown]
# ## 2. Model architectures
#
# The five neural models take input of shape `(batch, time, features)` and return one RUL value per
# sample. Linear Regression, Random Forest, and XGBoost are instantiated in Section 8. MS-TCN follows
# Farha & Gall (2019), adapted from per-frame action segmentation to sequence-to-one regression.


# %%
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]


class BiLSTMRUL(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                             bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.pad = pad
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.relu(self.conv1(x)[:, :, :-self.pad] if self.pad else self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out)[:, :, :-self.pad] if self.pad else self.conv2(out))
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNRUL(nn.Module):
    def __init__(self, n_features, channels=(64, 64, 64), kernel_size=3, dropout=0.2):
        super().__init__()
        layers, in_ch = [], n_features
        for i, ch in enumerate(channels):
            layers.append(TCNBlock(in_ch, ch, kernel_size, dilation=2 ** i, dropout=dropout))
            in_ch = ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(in_ch, 64), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, x):
        # x: (batch, time, features) -> conv1d wants (batch, features, time)
        out = self.tcn(x.transpose(1, 2))
        return self.head(out[:, :, -1]).squeeze(-1)


class MSTCNStage(nn.Module):
    # One stage of a Multi-Stage TCN (Farha & Gall, 2019): a stack of dilated residual conv
    # blocks that maps its input to a per-timestep OUTPUT of `out_ch` channels. What makes a
    # network "multi-stage" (vs. just a deeper single-stage TCN) is that subsequent stages
    # consume the previous stage's per-timestep OUTPUT, not its hidden features or the raw input
    # — each stage's job is to refine the previous stage's per-timestep estimate.
    def __init__(self, in_ch, hidden_ch, out_ch, kernel_size=3, num_layers=4, dropout=0.2):
        super().__init__()
        layers, ch = [], in_ch
        for i in range(num_layers):
            layers.append(TCNBlock(ch, hidden_ch, kernel_size, dilation=2 ** i, dropout=dropout))
            ch = hidden_ch
        self.net = nn.Sequential(*layers)
        self.out_proj = nn.Conv1d(hidden_ch, out_ch, 1)
    def forward(self, x):   # x: (batch, channels, time)
        h = self.net(x)
        return self.out_proj(h), h   # (per-timestep refined output, hidden features)


class MSTCN(nn.Module):
    # Multi-Stage TCN adapted from per-frame classification (its original action-segmentation
    # setting) to sequence-to-one RUL regression: stage 1 processes the raw windowed sensor
    # sequence, each following stage refines the PREVIOUS stage's per-timestep output, and the
    # final stage's last-timestep hidden feature feeds a regression head.
    def __init__(self, n_features, num_stages=3, hidden_ch=64, kernel_size=3, num_layers=4, dropout=0.2):
        super().__init__()
        self.stage1 = MSTCNStage(n_features, hidden_ch, hidden_ch, kernel_size, num_layers, dropout)
        self.refine_stages = nn.ModuleList([
            MSTCNStage(hidden_ch, hidden_ch, hidden_ch, kernel_size, num_layers, dropout)
            for _ in range(max(num_stages - 1, 0))
        ])
        self.head = nn.Sequential(nn.Linear(hidden_ch, 64), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, x):   # x: (batch, time, features)
        out, feat = self.stage1(x.transpose(1, 2))
        for stage in self.refine_stages:
            out, feat = stage(out)     # each stage refines the previous stage's OUTPUT
        return self.head(feat[:, :, -1]).squeeze(-1)


class MSTCNBiLSTM(nn.Module):
    # Hybrid: the same multi-stage temporal-refinement branch as MSTCN, concatenated with an
    # independent BiLSTM branch over the raw sequence, following the common RUL-literature
    # pattern of combining a convolutional/temporal feature extractor with a recurrent one before
    # the regression head.
    def __init__(self, n_features, num_stages=3, tcn_hidden=64, lstm_hidden=32,
                 kernel_size=3, num_layers=4, dropout=0.2):
        super().__init__()
        self.stage1 = MSTCNStage(n_features, tcn_hidden, tcn_hidden, kernel_size, num_layers, dropout)
        self.refine_stages = nn.ModuleList([
            MSTCNStage(tcn_hidden, tcn_hidden, tcn_hidden, kernel_size, num_layers, dropout)
            for _ in range(max(num_stages - 1, 0))
        ])
        self.lstm = nn.LSTM(n_features, lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(tcn_hidden + lstm_hidden * 2, 64), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, x):
        out, feat = self.stage1(x.transpose(1, 2))
        for stage in self.refine_stages:
            out, feat = stage(out)
        tcn_repr = feat[:, :, -1]
        lstm_out, _ = self.lstm(x)
        lstm_repr = lstm_out[:, -1, :]
        return self.head(torch.cat([tcn_repr, lstm_repr], dim=-1)).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerRUL(nn.Module):
    # Temporal / trend-guided transformer (TGT-style): a projection + positional encoding +
    # a standard transformer encoder + attention-pooled regression head.
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=3, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4,
                                                dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pool_attn = nn.Linear(d_model, 1)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, x):
        h = self.pos(self.proj(x))
        h = self.encoder(h)
        weights = torch.softmax(self.pool_attn(h), dim=1)
        pooled = (h * weights).sum(dim=1)
        return self.head(pooled).squeeze(-1)


def build_dl_models(n_features):
    # Fresh instances each call -> fresh weight init. Called once per subset.
    return {
        "BiLSTM": BiLSTMRUL(n_features),
        "TCN": TCNRUL(n_features),
        "MS-TCN": MSTCN(n_features),
        "MS-TCN+BiLSTM": MSTCNBiLSTM(n_features),
        "Transformer": TransformerRUL(n_features),
    }


def train_dl_model(model, name, train_loader, val_loader, epochs=60, lr=1e-3, patience=8, seed=SEED):
    # Reseed here (not just once globally) so each model's DataLoader shuffle order and any
    # stochastic op inside training is independently reproducible regardless of what ran before.
    torch.manual_seed(seed)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    loss_fn = nn.MSELoss()

    best_val, best_state, bad_epochs, history = float("inf"), None, 0, []

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_losses.append(loss_fn(model(xb), yb).item() * len(xb))
        val_rmse = math.sqrt(sum(val_losses) / len(val_loader.dataset))
        sched.step(val_rmse ** 2)
        history.append(val_rmse)

        if val_rmse < best_val - 1e-3:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    print(f"  {name:18s} best val RMSE = {best_val:6.2f}  (stopped at epoch {epoch + 1})")
    return model, best_val, history


def predict_test(model, test_loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds)


def verify_reproducibility(subset, dl_models_trained, test_loader, run_vlm_fn=None,
                            test_graph_paths=None, n_vlm_check=2):
    """
    Runs a bounded, real determinism check and reports max/mean numerical differences -- this is
    the notebook's answer to "can I run the same configuration twice and get the same result."

    1. DL models (cheap, no GPU generation involved): re-run inference twice through each already-
       trained model in eval() mode and diff the outputs. With dropout disabled by .eval() and no
       other stochastic ops at inference time, this should be exactly 0.0 -- if it isn't, that's a
       real reproducibility bug worth knowing about, which is exactly why this check exists rather
       than just asserting it.
    2. VLM (bounded cost: only n_vlm_check engines, not all of them, since a real generation call
       is expensive): re-run the real VLM on a couple of already-rendered images and diff the
       parsed confidence value + trend/qualitative_rul string agreement. Requires run_vlm_fn and
       test_graph_paths; skipped (not faked) if either is unavailable.
    """
    banner(f"[TITLE] {subset} — Reproducibility verification")
    report = {"subset": subset, "dl_model_determinism": {}, "vlm_determinism": None}

    for name, model in dl_models_trained.items():
        preds1 = predict_test(model, test_loader)
        preds2 = predict_test(model, test_loader)
        max_diff = float(np.max(np.abs(preds1 - preds2)))
        mean_diff = float(np.mean(np.abs(preds1 - preds2)))
        report["dl_model_determinism"][name] = {"max_abs_diff": max_diff, "mean_abs_diff": mean_diff}
        print(f"  {name:18s} max|diff|={max_diff:.6f}  mean|diff|={mean_diff:.6f}  "
              f"({'deterministic' if max_diff == 0.0 else 'NOT bit-identical -- investigate'})")

    if run_vlm_fn is not None and test_graph_paths and n_vlm_check > 0:
        sample_units = list(test_graph_paths.keys())[:n_vlm_check]
        vlm_checks = []
        for u in sample_units:
            r1, _ = run_vlm_fn(test_graph_paths[u], u, "test")
            r2, _ = run_vlm_fn(test_graph_paths[u], u, "test")
            same_trend = r1.get("degradation_trend") == r2.get("degradation_trend")
            same_band = r1.get("qualitative_rul") == r2.get("qualitative_rul")
            same_regime = r1.get("degradation_regime") == r2.get("degradation_regime")
            same_quality = r1.get("evidence_quality") == r2.get("evidence_quality")
            same_tp = r1.get("terminal_phase_start_cycle") == r2.get("terminal_phase_start_cycle")
            same_sensors = (r1.get("abnormal_sensors") or []) == (r2.get("abnormal_sensors") or [])
            conf1, conf2 = r1.get("confidence"), r2.get("confidence")
            conf_diff = abs((conf1 or 0) - (conf2 or 0)) if conf1 is not None and conf2 is not None else None
            _fields = {"same_trend": same_trend, "same_qualitative_rul": same_band,
                        "same_degradation_regime": same_regime, "same_evidence_quality": same_quality,
                        "same_terminal_phase_start": same_tp, "same_abnormal_sensors": same_sensors}
            vlm_checks.append({"unit": int(u), **_fields, "confidence_diff": conf_diff,
                                "all_structured_fields_identical": all(_fields.values())})
            print(f"  VLM re-run, unit {u}: "
                  + ", ".join(f"{k.replace('same_','')}={'OK' if v else 'DIFFERS'}" for k, v in _fields.items())
                  + f", confidence diff={_fmt(conf_diff) if conf_diff is not None else 'n/a'}")
            if not all(_fields.values()):
                print(f"    !! structured fields differed across identical re-runs of unit {u} -- "
                      f"greedy decoding should make this deterministic; record it rather than ignore it.")
        report["vlm_determinism"] = vlm_checks
    else:
        print("  VLM determinism check skipped for this call (no run_vlm_fn/test_graph_paths passed).")

    return report

print("Model classes defined: BiLSTMRUL, TCNRUL, MSTCN, MSTCNBiLSTM, TransformerRUL "
      "(+ build_dl_models / train_dl_model / predict_test / verify_reproducibility helpers)")


# %% [markdown]
# ## 3. Preprocessing (numerical branch)
#
# 1. Remove sensors with near-zero variance in each subset's training set.
# 2. Apply min-max scaling fit on training data only. For the multi-regime subsets (FD002, FD004),
#    operating conditions are clustered with KMeans and each regime is scaled separately.
# 3. Apply a piecewise-linear RUL cap (`RUL_CAP = 125`).
# 4. Derive test-set RUL from `RUL_FD00X.txt`.
# 5. Build sliding-window sequences for the neural models and window summary statistics
#    (mean, std, last value, slope) for the tabular baselines.
#
# This preprocessing applies only to the numerical branch; the VLM receives raw sensor values.


# %%
def drop_low_variance_sensors(train_df, sensor_cols=SENSOR_COLS, threshold=1e-5):
    variances = train_df[sensor_cols].var()
    keep = variances[variances > threshold].index.tolist()
    dropped = [c for c in sensor_cols if c not in keep]
    return keep, dropped


from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

def fit_regime_normalizers(train_df, sensor_cols, n_regimes, seed=SEED):
    # Fit on TRAIN ONLY. Returns (kmeans_or_None, {regime_id: fitted MinMaxScaler}).
    if n_regimes <= 1:
        return None, {0: MinMaxScaler().fit(train_df[sensor_cols])}
    km = KMeans(n_clusters=n_regimes, random_state=seed, n_init=10)
    km.fit(train_df[["op1", "op2", "op3"]])
    labels = km.predict(train_df[["op1", "op2", "op3"]])
    scalers = {r: MinMaxScaler().fit(train_df.loc[labels == r, sensor_cols]) for r in range(n_regimes)}
    return km, scalers

def apply_regime_normalization(df, kmeans, scalers, sensor_cols):
    out = df.copy()
    if kmeans is None:
        out[sensor_cols] = scalers[0].transform(out[sensor_cols])
        return out
    labels = kmeans.predict(out[["op1", "op2", "op3"]])
    normed = np.zeros((len(out), len(sensor_cols)))
    for r, scaler in scalers.items():
        mask = labels == r
        if mask.sum() == 0:
            continue
        normed[mask] = scaler.transform(out.loc[mask, sensor_cols])
    out[sensor_cols] = normed
    return out


def add_piecewise_rul(df, rul_cap):
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["RUL"] = np.minimum(max_cycle - out["cycle"], rul_cap)
    return out

def add_test_rul(df, rul_table, rul_cap):
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    end_rul = out["unit"].map(rul_table.set_index("unit")["RUL"])
    out["RUL"] = np.minimum((max_cycle - out["cycle"]) + end_rul, rul_cap)
    return out


def create_sequences(df, sensor_cols, window, stride, is_train):
    # Sliding-window sequences per engine. Short trajectories (< window) are left-padded by
    # repeating the first observed row (standard trick, e.g. Zheng et al. 2017 / Li et al. 2018).
    X, y, meta = [], [], []
    for unit, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        arr, ruls, cycles = g[sensor_cols].values, g["RUL"].values, g["cycle"].values
        n = len(g)
        if n < window:
            pad = np.repeat(arr[:1], window - n, axis=0)
            arr = np.vstack([pad, arr])
            ruls = np.concatenate([np.repeat(ruls[0], window - n), ruls])
            cycles = np.concatenate([np.repeat(cycles[0], window - n), cycles])
            n = window
        if is_train:
            for start in range(0, n - window + 1, stride):
                X.append(arr[start:start + window])
                y.append(ruls[start + window - 1])
                meta.append((unit, cycles[start + window - 1]))
        else:
            X.append(arr[-window:])   # only the final window per test engine (what we evaluate on)
            y.append(ruls[-1])
            meta.append((unit, cycles[-1]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), meta

def create_tabular_features(df, sensor_cols, window, stride, is_train):
    # Summary-statistic features per window (mean/std/last/slope) for the ML baselines.
    Xs, y, meta = create_sequences(df, sensor_cols, window, stride, is_train)
    n, w, s = Xs.shape
    t = np.arange(w)
    feats = []
    for i in range(n):
        win = Xs[i]
        mean, std, last = win.mean(axis=0), win.std(axis=0), win[-1]
        slope = np.array([np.polyfit(t, win[:, j], 1)[0] for j in range(s)])
        feats.append(np.concatenate([mean, std, last, slope]))
    return np.array(feats, dtype=np.float32), y, meta

print("Preprocessing functions defined: drop_low_variance_sensors, fit/apply_regime_normalizers, "
      "add_piecewise_rul, add_test_rul, create_sequences, create_tabular_features")


# %% [markdown]
# ## 4. Graph generation (visual branch)
#
# Each engine is rendered as one figure containing a subplot for each of the 21 sensors (raw value
# vs. cycle), an all-sensor overlay panel, and a within-regime detrended panel for multi-regime
# engines. Each subplot shows the raw signal and a rolling-mean trend line. All rescaling is for
# display only and does not affect the numerical models.


# %%
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

def generate_engine_graph(unit_df, unit_id, split, sensor_cols, out_dir, dpi=150):
    unit_df = unit_df.sort_values("cycle")
    n_sensors = len(sensor_cols)
    ncols = 4
    n_extra_panels = 2   # all-sensor overlay + within-regime detrended overlay
    nrows = math.ceil((n_sensors + n_extra_panels) / ncols)

    cycles = unit_df["cycle"].values
    n_cycles = len(cycles)
    # Adaptive smoothing window: ~10% of the trajectory length, clamped to a sane range so short
    # test trajectories still get some smoothing and long ones don't over-smooth away real drift.
    smooth_window = int(np.clip(n_cycles // 10, 5, 25))

    def _smoothed(values):
        return pd.Series(values, dtype=float).rolling(
            smooth_window, center=True, min_periods=1).mean().values

    def _style_x_axis(ax):
        # Major gridlines (labeled, ~6 round numbers) give orientation; minor gridlines (unlabeled,
        # 4 per major interval) give intermediate reference points so an onset estimate can land
        # BETWEEN two round numbers instead of only ever matching a labeled tick -- without this,
        # a model reading the chart has nothing to anchor a non-round answer to.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.tick_params(labelsize=7)
        ax.grid(True, which="major", axis="x", alpha=0.35, linewidth=0.6)
        ax.grid(True, which="minor", axis="x", alpha=0.15, linewidth=0.4, linestyle=":")

    # --- DISPLAY-ONLY regime handling -------------------------------------------------
    # Detect whether this engine's operating settings form discrete clusters (FD002/FD004 have 6
    # operating conditions; FD001/FD003 have 1). If they do, the raw traces split into parallel
    # bands that visually dominate any degradation trend. We therefore ALSO plot a within-regime
    # z-scored trace, which removes the between-band offset for VIEWING ONLY. Nothing is dropped:
    # the raw line and its smoothed trend remain on every panel exactly as before, and none of
    # this touches the numerical branch (which does its own regime normalisation separately).
    _op_cols = [c for c in ("op1", "op2", "op3") if c in unit_df.columns]
    _regime_id = None
    if _op_cols:
        _ops = unit_df[_op_cols].round(0).astype(int)
        _regime_id = _ops.astype(str).agg("|".join, axis=1).values
        if len(set(_regime_id)) <= 1:
            _regime_id = None     # single regime -> nothing to correct for

    def _regime_detrended(values):
        # z-score within each operating regime, so the bands overlay instead of stacking
        v = np.asarray(values, dtype=float)
        if _regime_id is None:
            return None
        out = np.full_like(v, np.nan)
        for r in set(_regime_id):
            m = _regime_id == r
            seg = v[m]
            sd = seg.std()
            out[m] = (seg - seg.mean()) / sd if sd > 1e-9 else 0.0
        return out

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.6, nrows * 3.3))
    axes = axes.flatten()

    for i, sensor in enumerate(sensor_cols):
        ax = axes[i]
        raw = unit_df[sensor].values.astype(float)
        ax.plot(cycles, raw, linewidth=0.7, color="#a8c4e0", alpha=0.85, zorder=1, label="raw")
        ax.plot(cycles, _smoothed(raw), linewidth=1.9, color="#1f3d7a", zorder=2, label="smoothed trend")
        ax.set_title(sensor, fontsize=10)
        ax.set_xlabel("cycle", fontsize=8)
        _style_x_axis(ax)
    axes[0].legend(fontsize=6, loc="best")   # explain the two line styles once, not on every panel

    ax = axes[n_sensors]
    for sensor in sensor_cols:
        smoothed = _smoothed(unit_df[sensor].values.astype(float))
        rng = smoothed.max() - smoothed.min()
        v_norm = (smoothed - smoothed.min()) / rng if rng > 1e-9 else np.zeros_like(smoothed)
        ax.plot(cycles, v_norm, linewidth=1.1, alpha=0.65)
    ax.set_title("All sensors overlay (smoothed trend, display-only rescale)", fontsize=9)
    ax.set_xlabel("cycle", fontsize=8)
    _style_x_axis(ax)

    # --- extra panel: within-regime detrended overlay (display-only) ---
    ax = axes[n_sensors + 1]
    if _regime_id is not None:
        for sensor in sensor_cols:
            det = _regime_detrended(unit_df[sensor].values)
            if det is None:
                continue
            ax.plot(cycles, _smoothed(det), linewidth=1.0, alpha=0.7)
        ax.set_title(f"Within-regime detrended overlay ({len(set(_regime_id))} operating regimes "
                     f"detected; display-only)", fontsize=8)
        ax.set_ylabel("z-score within regime", fontsize=7)
    else:
        ax.text(0.5, 0.5, "Single operating regime detected\n(no regime detrending needed)",
                ha="center", va="center", fontsize=9, transform=ax.transAxes)
        ax.set_title("Within-regime detrended overlay", fontsize=8)
    ax.set_xlabel("cycle", fontsize=8)
    _style_x_axis(ax)

    for j in range(n_sensors + n_extra_panels, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{split} engine unit {unit_id} — sensor trends over {n_cycles} operating cycles "
                 f"(light = raw, bold = smoothed trend; gridlines mark cycle numbers; "
                 f"final panel shows within-regime detrended trends)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(out_dir, f"{split}_unit{unit_id}.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path

print("generate_engine_graph() defined (smoothed-trend overlay + major/minor cycle gridlines).")


# %% [markdown]
# ## 5. VLM branch
#
# Qwen2.5-VL-7B-Instruct reads each engine figure and returns a JSON object with the fields defined
# in `VLM_SCHEMA_DESCRIPTION`: degradation trend and regime, terminal-phase start cycle, abnormal
# sensors, qualitative RUL band, evidence quality, confidence, and a short explanation.
#
# Responses are validated against the schema and the set of plotted sensor names. Invalid responses
# are re-prompted up to `MAX_LLM_RETRIES` times and are otherwise flagged `_parse_failed`. Decoding
# is greedy (`do_sample=False`).
#
# `terminal_phase_start_cycle` is an operational estimate of where a terminal degradation phase
# begins; C-MAPSS provides no ground-truth onset label.


# %%
VLM_SCHEMA_ALLOWED = {
    "degradation_trend": {"increasing", "decreasing", "stable", "mixed", None},
    "degradation_regime": {"stable", "degrading", "rapid_degrading", None},
    "qualitative_rul": {"high", "medium", "low", "critical", None},
    "evidence_quality": {"clear", "weak", "unreadable", None},
}
VLM_SCHEMA_DESCRIPTION = {
    "engine_id": "integer, the engine unit id",
    "degradation_trend": "one of: increasing | decreasing | stable | mixed, or null if not evident",
    "degradation_regime": "one of: stable | degrading | rapid_degrading, or null if not evident. "
                          "'stable' = bold lines flat within their early-life band; 'degrading' = a "
                          "sustained but gradual departure; 'rapid_degrading' = a steep or "
                          "accelerating departure in the most recent portion of the record.",
    "stable_region_cycles": "[start,end] cycle range that looks stable/healthy, or null",
    "unstable_region_cycles": "[start,end] cycle range with visibly unstable/degrading behavior, or null",
    "terminal_phase_start_cycle": "integer cycle where the BOLD smoothed trend visibly departs "
                                  "from its early-life baseline, based on the curve itself -- or "
                                  "null if the bold line stays flat throughout. This is an "
                                  "operational judgment about where a terminal/critical phase "
                                  "appears to begin, not a claim about the true physical onset of "
                                  "damage, which cannot be read off a sensor chart.",
    "regime_changes_detected": "true/false, or null if you cannot tell from the graph",
    "abnormal_sensors": "list of the sensors whose BOLD line drift is clearly larger than most "
                        "other sensors in this image -- a comparison across the sensors shown, "
                        "not a fixed count. Return [] if no sensor clearly stands out.",
    "sensor_relationships": "short string describing how sensors move together or diverge, or null",
    "approaching_failure": "true/false, or null if you cannot tell",
    "qualitative_rul": "one of: high | medium | low | critical, based on the trend, "
                       "terminal_phase_start_cycle, and abnormal_sensors you reported -- see the "
                       "mapping given in the instructions below",
    "confidence": "float 0.0-1.0, your confidence in this specific assessment given what the "
                  "graph actually shows",
    "explanation": "1-3 sentences citing the SPECIFIC visual evidence (which sensors, which cycle "
                   "range) behind your assessment -- not a generic statement",
    "evidence_quality": "one of: clear | weak | unreadable. How legible this particular figure is "
                        "for degradation assessment: 'clear' = trends are plainly visible; 'weak' = "
                        "some drift but noisy/ambiguous; 'unreadable' = regime switching or noise "
                        "dominates so no trend judgment is supportable. Report 'unreadable' "
                        "honestly when that is the case -- it is a valid and useful answer.",
}
VLM_SCHEMA_EXAMPLE = {
    "engine_id": 7, "degradation_trend": "increasing", "degradation_regime": "rapid_degrading",
    "stable_region_cycles": [1, 123],
    "unstable_region_cycles": [124, 187], "terminal_phase_start_cycle": 124,
    "regime_changes_detected": False, "abnormal_sensors": ["s7", "s12", "s15"],
    "sensor_relationships": "s7 and s12 rise together while s15 falls, consistent with a single fault mode.",
    "approaching_failure": True, "qualitative_rul": "low", "confidence": 0.78,
    "explanation": "s7/s12/s15 show a clear monotonic drift starting around cycle 124, clearly "
                   "larger than any other sensor in this image, consistent with progressive "
                   "degradation approaching end-of-life.",
    "evidence_quality": "clear",
}
VLM_SCHEMA_EXAMPLE_INSUFFICIENT_EVIDENCE = {
    "engine_id": 41, "degradation_trend": "stable", "degradation_regime": "stable",
    "stable_region_cycles": [1, 45],
    "unstable_region_cycles": None, "terminal_phase_start_cycle": None,
    "regime_changes_detected": False, "abnormal_sensors": [],
    "sensor_relationships": None, "approaching_failure": False, "qualitative_rul": "high",
    "confidence": 0.4, "explanation": "The bold smoothed line stays flat within its early-life "
                                       "band in every subplot across all 45 visible cycles.",
    "evidence_quality": "weak",
}
# Third worked example: a multi-regime figure where regime switching dominates. Added because a
# real run produced 'unreadable'-type figures on FD002/FD004 (6 operating regimes) yet the VLM
# still emitted a confident 'high' band on ~85% of engines -- this example makes the honest
# abstention path concrete instead of leaving 'high' as the path of least resistance.
VLM_SCHEMA_EXAMPLE_UNREADABLE = {
    "engine_id": 88, "degradation_trend": None, "degradation_regime": None,
    "stable_region_cycles": None, "unstable_region_cycles": None,
    "terminal_phase_start_cycle": None, "regime_changes_detected": True,
    "abnormal_sensors": [], "sensor_relationships": None, "approaching_failure": None,
    "qualitative_rul": None, "confidence": 0.2,
    "explanation": "Sensor values switch between several discrete operating levels throughout the "
                   "record; the resulting banding dominates every subplot, so no degradation trend "
                   "can be separated from the regime changes.",
    "evidence_quality": "unreadable",
}

VLM_SYSTEM_PROMPT = (
    "You are a senior reliability engineer specializing in turbofan engine prognostics. "
    "You are shown a figure with one trend subplot per sensor for a single engine (operating "
    "cycle on the x-axis, raw sensor value on the y-axis), plus an overlay panel. In every "
    "subplot, the LIGHT thin line is the raw sensor value and the BOLD dark line is a smoothed "
    "rolling-average trend of that same sensor -- use the bold line to judge direction and "
    "magnitude of drift, and the light line only to gauge how noisy the signal is. Vertical "
    "gridlines mark reference cycle numbers for orientation. Analyze the trends the way an "
    "experienced engineer would when judging degradation and proximity to failure, and report "
    "only what the bold line in front of you actually supports.\n\n"
    "IMPORTANT CONTEXT about what you are looking at:\n"
    "- This record is TRUNCATED at an arbitrary point before failure. The engine does NOT fail at "
    "the right edge of the plot. So a chart that ends while still trending is common and expected, "
    "and the absence of a dramatic end-of-life collapse is NOT evidence that the engine is "
    "healthy. Judge the degradation STATE from the shape and slope of the visible record, not from "
    "how close the right edge looks to failure.\n"
    "- Some engines operate under several DISCRETE OPERATING CONDITIONS. When that happens the raw "
    "traces separate into parallel bands or jump between levels, and this banding can completely "
    "hide a real degradation trend. The FINAL panel of the figure is a 'within-regime detrended "
    "overlay': each sensor is z-scored inside its own operating regime, which removes the "
    "between-band offset so that any underlying drift becomes visible. When the per-sensor panels "
    "look like banding, READ THAT PANEL -- a trend visible there is real evidence even though the "
    "raw panels look like noise. If the detrended panel is also featureless, say so via "
    "evidence_quality = 'unreadable' and return nulls -- do not guess a trend through the banding, "
    "and equally do not report 'stable'/'high' just because no trend is visible. 'I cannot read "
    "this' and 'this engine is healthy' are DIFFERENT answers and must not be conflated.\n\n"
    "Field-by-field guidance:\n"
    "- Sensor noise in the thin raw line is normal and is not itself evidence of degradation -- "
    "base your trend judgment on the bold smoothed line.\n"
    "- degradation_regime: 'stable' if the bold lines stay within their early-life band; "
    "'degrading' if there is a sustained but gradual departure; 'rapid_degrading' if the departure "
    "is steep or visibly accelerating toward the end of the record.\n"
    "- terminal_phase_start_cycle: your best estimate, read directly from the curve, of the cycle "
    "where the bold line departs from its early-life baseline. This is an operational estimate of "
    "where a terminal/critical phase appears to begin, not a claim about the true physical onset "
    "of damage (which cannot be determined from a sensor chart alone). Report null if the bold "
    "line does not show a departure.\n"
    "- abnormal_sensors: name the sensors whose bold-line drift is clearly larger than the others "
    "shown, as a comparison across this specific image -- not against a fixed threshold or a "
    "fixed count. Return [] if nothing clearly stands out.\n"
    "- qualitative_rul: set this from the trend / terminal-phase-start / abnormal-sensor evidence "
    "you just reported, using this mapping -- a stable trend with no abnormal sensors maps to "
    "'high'; a clear terminal-phase start with several abnormal sensors and a large unstable "
    "region maps to 'low' or 'critical' depending on how far past that start the trajectory runs; "
    "anything else (weak or single-sensor drift, an ambiguous or very recent terminal-phase start) "
    "maps to 'medium'. If evidence_quality is 'unreadable', qualitative_rul MUST be null -- an "
    "unreadable figure supports no band at all, and 'high' is NOT the fallback for 'I could not "
    "see a trend'.\n"
    "- terminal_phase_start_cycle: report a cycle whenever the bold line shows a visible transition "
    "toward faster degradation, even if the record ends before failure. This field is about a "
    "visible transition in the curve, not about proximity to the right edge of the plot.\n"
    "- confidence: reflect how clear the visual evidence actually is for THIS engine -- a flat, "
    "featureless chart and a chart with several sensors clearly drifting can both be reported with "
    "high confidence in their respective (different) conclusions; an ambiguous chart should get a "
    "lower confidence.\n"
    "- Only report null/[] when the bold line genuinely does not support a field -- do not invent "
    "a trend, sensor, or cycle number that is not visible, but also do not withhold an answer the "
    "chart actually supports.\n\n"
    "Respond with ONLY a single JSON object, no prose, no markdown code fences, no text before or "
    "after it. Use exactly these keys (allowed values/types in parentheses):\n"
    + json.dumps(VLM_SCHEMA_DESCRIPTION, indent=2)
    + "\n\nExample of a well-formed response with clear evidence:\n"
    + json.dumps(VLM_SCHEMA_EXAMPLE, indent=2)
    + "\n\nExample of a well-formed response on a genuinely flat engine (nulls/[] are a normal, "
      "valid answer here, not a failure):\n"
    + json.dumps(VLM_SCHEMA_EXAMPLE_INSUFFICIENT_EVIDENCE, indent=2)
    + "\n\nExample of a well-formed response on a figure dominated by operating-regime banding, "
      "where abstaining is the correct answer:\n"
    + json.dumps(VLM_SCHEMA_EXAMPLE_UNREADABLE, indent=2)
)

def build_vlm_user_prompt(unit_id, split):
    return (f"This is {split} engine unit {unit_id}. The record is truncated before failure. Study "
            f"the bold smoothed trend line in every subplot, then return the JSON object described "
            f"in the system prompt, reporting only what this specific chart supports. If "
            f"operating-regime banding makes the figure unreadable, say so rather than defaulting "
            f"to a healthy verdict.")

def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)   # light repair: trailing commas
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None

def validate_vlm_output(parsed, valid_sensor_names=VLM_SENSORS):
    if parsed is None:
        return ["Response was not valid JSON."]
    errors = [f"Missing required key '{k}'." for k in VLM_SCHEMA_DESCRIPTION if k not in parsed]
    for key, allowed in VLM_SCHEMA_ALLOWED.items():
        if key in parsed and parsed[key] not in allowed:
            errors.append(f"Key '{key}' must be one of {sorted(a for a in allowed if a)} or null, "
                           f"got {parsed.get(key)!r}.")
    if "confidence" in parsed and parsed["confidence"] is not None:
        try:
            c = float(parsed["confidence"])
            if not (0.0 <= c <= 1.0):
                errors.append("Key 'confidence' must be a float between 0.0 and 1.0 (or null).")
        except (TypeError, ValueError):
            errors.append("Key 'confidence' must be a number or null.")
    # Validate any reported sensor names against the actual sensor set the VLM was shown -- this
    # catches hallucinated sensor names (e.g. a sensor that doesn't exist, or a typo) before they
    # ever reach the fusion layer or a saved artifact.
    abnormal = parsed.get("abnormal_sensors")
    if abnormal:
        if not isinstance(abnormal, list):
            errors.append("Key 'abnormal_sensors' must be a list (use [] if none).")
        else:
            unknown = [s for s in abnormal if s not in valid_sensor_names]
            if unknown:
                errors.append(f"Key 'abnormal_sensors' contains sensor name(s) not present in "
                               f"this figure: {unknown}. Only use names from: {valid_sensor_names}.")
    return errors


# %%
banner("[TITLE] Loading the real VLM (Qwen2.5-VL-7B-Instruct) — once, reused for all subsets")

vlm_model, vlm_processor = None, None

def load_real_vlm():
    global vlm_model, vlm_processor
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    quant_cfg = None
    if USE_4BIT:
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID, torch_dtype="auto", device_map="auto", quantization_config=quant_cfg)
    vlm_model.eval()
    # min/max_pixels controls how many visual tokens the ViT spends on the image -- our composite
    # multi-panel grids are dense (now with an added smoothed-trend line and gridlines, at higher
    # dpi), so raise the ceiling further above the library default so that added detail survives
    # resizing instead of being thrown away (per the official model card's documented
    # min_pixels/max_pixels usage pattern). This does cost some extra latency per VLM call.
    vlm_processor = AutoProcessor.from_pretrained(
        VLM_MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=2048 * 28 * 28)
    assert next(vlm_model.parameters()).is_cuda or DEVICE == "cpu", "VLM did not load onto CUDA"
    print(f"Loaded {VLM_MODEL_ID} (4bit={USE_4BIT}) on {vlm_model.device}")

load_real_vlm()


# %%
def _generate_vlm_once(image_path, unit_id, split, prior_errors=None, max_new_tokens=512):
    from qwen_vl_utils import process_vision_info
    user_text = build_vlm_user_prompt(unit_id, split)
    if prior_errors:
        user_text += ("\n\nYour previous response was invalid for these reasons:\n- "
                       + "\n- ".join(prior_errors) + "\nReturn a corrected, complete JSON object only.")
    messages = [
        {"role": "system", "content": VLM_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": user_text},
        ]},
    ]
    text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vlm_processor(text=[text], images=image_inputs, videos=video_inputs,
                            padding=True, return_tensors="pt").to(vlm_model.device)
    with torch.no_grad():
        generated = vlm_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    return vlm_processor.batch_decode(trimmed, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)[0]

def run_vlm(image_path, unit_id, split):
    # Always the real model. On invalid/non-conformant JSON, re-prompt the SAME model with the
    # validation errors up to MAX_LLM_RETRIES times. No mock/synthetic fallback at any point.
    errors, last_raw, parsed = None, "", None
    for attempt in range(MAX_LLM_RETRIES + 1):
        raw_text = _generate_vlm_once(image_path, unit_id, split, prior_errors=errors)
        last_raw = raw_text
        parsed = extract_json(raw_text)
        errors = validate_vlm_output(parsed)
        if not errors:
            parsed["engine_id"] = int(parsed.get("engine_id") or unit_id)
            return parsed, raw_text
    partial = parsed if isinstance(parsed, dict) else {}
    partial["engine_id"] = int(unit_id)
    partial["_parse_failed"] = True
    partial["_validation_errors"] = errors
    return partial, last_raw

print("VLM prompt/schema/retry logic ready: run_vlm(image_path, unit_id, split)")


# %% [markdown]
# ## 6. LLM evidence analysis
#
# For each engine, Qwen2.5-7B-Instruct receives the individual model predictions, their validation
# RMSEs, the VLM output, and the computed fusion result. It returns `agreement_assessment`,
# `vlm_numeric_consistency`, `reliability_reasoning`, `recommended_action`, and `explanation`. It
# does not produce or modify the numeric RUL.


# %%
FUSION_ALLOWED_KEYS = {"agreement_assessment", "vlm_numeric_consistency", "reliability_reasoning",
                        "recommended_action", "explanation"}
FUSION_ALLOWED_VALUES = {"recommended_action": {"accept", "caution", "reject"}}

FUSION_SYSTEM_PROMPT = (
    "You are an evidence-analysis agent in a turbofan-engine prognostics system. You do NOT "
    "compute or output a numeric Remaining-Useful-Life (RUL) value -- that number has already "
    "been computed deterministically by a fixed formula (inverse-validation-RMSE weighting with "
    "outlier down-weighting, optionally nudged by the VLM's qualitative evidence) and is given to "
    "you below for you to comment on, not to change.\n\n"
    "You receive: (1) numeric RUL predictions from several independently trained ML/DL models, "
    "each with its own held-out validation RMSE (lower = historically more reliable); (2) a "
    "structured visual assessment from a vision-language model that read the engine's raw sensor "
    "trend charts (some fields may be null if the VLM judged the evidence insufficient -- treat "
    "null as 'no visual signal', not as a value of zero); and (3) the deterministic fusion result "
    "already computed from (1) and (2).\n\n"
    "Analyze this evidence and return ONLY a single JSON object, no prose, no markdown fences, "
    "with exactly these keys:\n"
    "- agreement_assessment (string): do the individual ML/DL models agree with each other?\n"
    "- vlm_numeric_consistency (string): are the numeric predictions consistent with the VLM's "
    "visual degradation evidence, or do they conflict?\n"
    "- reliability_reasoning (string): given the validation RMSEs and the visual evidence, which "
    "model(s)' predictions look most/least trustworthy for THIS engine, and why?\n"
    "- recommended_action (one of: 'accept' | 'caution' | 'reject'): 'accept' if the evidence "
    "sources agree and the deterministic result looks reliable; 'caution' if there is some "
    "disagreement or the VLM evidence is weak/null; 'reject' if the evidence sources clearly "
    "conflict or model predictions are wildly inconsistent with each other.\n"
    "- explanation (string): 1-3 sentences summarizing your overall assessment."
)

def build_fusion_user_prompt(unit_id, preds_by_model, val_rmse, vlm_obs, constrained_result, prior_errors=None):
    lines = [f"Engine unit: {unit_id}", "", "ML/DL model predictions (RUL in cycles):"]
    for name, pred in preds_by_model.items():
        lines.append(f"  - {name}: predicted RUL = {pred:.1f}, validation RMSE = {val_rmse[name]:.2f}")
    lines += ["", "VLM visual assessment:", json.dumps(vlm_obs, indent=2) if vlm_obs else "  (none available)"]
    lines += ["", "Deterministic fusion result already computed by Python (for you to analyze, not to change):",
              json.dumps({"final_rul": round(constrained_result["final_rul"], 1),
                          "weights": {k: round(v, 3) for k, v in constrained_result["weights"].items()},
                          "down_weighted_models": constrained_result["down_weighted_models"],
                          "model_agreement": round(constrained_result["agreement"], 3)}, indent=2)]
    if prior_errors:
        lines.append("\nYour previous response was invalid for these reasons:\n- " + "\n- ".join(prior_errors))
        lines.append("Return a corrected, complete JSON object only.")
    return "\n".join(lines)

def validate_fusion_output(parsed):
    if parsed is None:
        return ["Response was not valid JSON."]
    errors = [f"Missing required key '{k}'." for k in FUSION_ALLOWED_KEYS if k not in parsed]
    for key, allowed in FUSION_ALLOWED_VALUES.items():
        if key in parsed and parsed[key] not in allowed:
            errors.append(f"Key '{key}' must be one of {sorted(allowed)}, got {parsed.get(key)!r}.")
    return errors


# %%
banner("[TITLE] Loading the real evidence-analysis LLM (Qwen2.5-7B-Instruct) — once, reused for all subsets")

fusion_llm_model, fusion_llm_tokenizer = None, None

def load_real_fusion_llm():
    global fusion_llm_model, fusion_llm_tokenizer
    if FUSION_LLM_MODEL_ID == VLM_MODEL_ID and vlm_model is not None:
        fusion_llm_model = vlm_model
        fusion_llm_tokenizer = vlm_processor.tokenizer
        print("Reusing the VLM's text tower as the evidence-analysis agent (no separate model loaded).")
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    quant_cfg = None
    if USE_4BIT:
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    fusion_llm_tokenizer = AutoTokenizer.from_pretrained(FUSION_LLM_MODEL_ID)
    fusion_llm_model = AutoModelForCausalLM.from_pretrained(
        FUSION_LLM_MODEL_ID, torch_dtype="auto", device_map="auto", quantization_config=quant_cfg)
    fusion_llm_model.eval()
    assert next(fusion_llm_model.parameters()).is_cuda or DEVICE == "cpu", "Evidence-analysis LLM did not load onto CUDA"
    print(f"Loaded evidence-analysis LLM {FUSION_LLM_MODEL_ID} (4bit={USE_4BIT}) on {fusion_llm_model.device}")

load_real_fusion_llm()


# %%
def _generate_fusion_once(unit_id, preds_by_model, val_rmse, vlm_obs, constrained_result,
                           prior_errors=None, max_new_tokens=400):
    messages = [
        {"role": "system", "content": FUSION_SYSTEM_PROMPT},
        {"role": "user", "content": build_fusion_user_prompt(
            unit_id, preds_by_model, val_rmse, vlm_obs, constrained_result, prior_errors)},
    ]
    text = fusion_llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = fusion_llm_tokenizer(text, return_tensors="pt").to(fusion_llm_model.device)
    with torch.no_grad():
        out = fusion_llm_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return fusion_llm_tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

def run_llm_evidence_analysis(unit_id, preds_by_model, val_rmse, vlm_obs, constrained_result):
    # Always the real model, with retry-on-validation-error. No mock/synthetic fallback. This
    # function returns TEXT ANALYSIS ONLY -- it has no numeric RUL field to parse or use, by
    # construction (see FUSION_ALLOWED_KEYS above), so there is nothing here that could silently
    # override the deterministic fusion result even by accident.
    errors, last_raw, parsed = None, "", None
    for attempt in range(MAX_LLM_RETRIES + 1):
        raw = _generate_fusion_once(unit_id, preds_by_model, val_rmse, vlm_obs, constrained_result,
                                     prior_errors=errors)
        last_raw = raw
        parsed = extract_json(raw)
        errors = validate_fusion_output(parsed)
        if not errors:
            return parsed, raw
    partial = parsed if isinstance(parsed, dict) else {}
    partial["_parse_failed"] = True
    partial["_validation_errors"] = errors
    return partial, last_raw

print("Evidence-analysis prompt/schema/retry logic ready: "
      "run_llm_evidence_analysis(unit_id, preds_by_model, val_rmse, vlm_obs, constrained_result)")


# %% [markdown]
# ## 7. Metrics and fusion
#
# **Metrics:** RMSE, MAE, NASA PHM08 score, and Pearson correlation.
#
# **Fusion methods compared**
# - *Simple average* of all model predictions.
# - *Weighted average* with weights proportional to 1/validation RMSE.
# - *Constrained fusion (numeric only)* — inverse-RMSE weights with MAD-based outlier exclusion.
# - *Constrained fusion + VLM (proposed)* — as above, with each model's weight adjusted by at most
#   ±`VLM_ADJUSTMENT_BOUND` according to its consistency with the VLM's qualitative RUL band.
#
# **Confidence score:** a fixed weighted combination of model agreement (0.5), VLM band consistency
# (0.3), and VLM self-reported confidence (0.2). It is a reliability heuristic, not a calibrated
# probability.
#
# Fusion weights depend only on validation RMSE and VLM output; test labels are used only for
# evaluation.


# %%
from sklearn.metrics import mean_squared_error, mean_absolute_error

def nasa_score(y_true, y_pred):
    # Asymmetric NASA PHM08 scoring function: penalizes late (over-)predictions more heavily
    # than early (under-)predictions, matching the real cost asymmetry of missed maintenance.
    d = np.asarray(y_pred) - np.asarray(y_true)
    s = np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)
    return float(np.sum(s))

def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    corr = stats.pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan
    return {"RMSE": rmse, "MAE": mae, "NASA_score": nasa_score(y_true, y_pred), "Pearson_r": corr}


# Mapping from the VLM's qualitative_rul category to a numeric RUL range. This is a FIXED SCORING
# CONVENTION used only to check whether a numeric prediction falls inside the VLM's stated band
# (band_consistency below) and to bound the VLM's adjustment in constrained_fusion -- it is not a
# claim that the VLM performs numeric regression, and these ranges are not independently validated.
QUAL_BAND_TO_RANGE = {"critical": (0, 20), "low": (0, 50), "medium": (30, 90), "high": (60, 200)}

def band_consistency(numeric_pred, qual_band):
    if qual_band is None:
        return 0.5   # VLM explicitly abstained -- neutral, not a penalty
    lo, hi = QUAL_BAND_TO_RANGE.get(qual_band, (0, 200))
    if lo <= numeric_pred <= hi:
        return 1.0
    dist = min(abs(numeric_pred - lo), abs(numeric_pred - hi))
    return float(max(0.0, 1.0 - dist / 50.0))

def simple_average_fusion(preds_by_model):
    return float(np.mean(list(preds_by_model.values())))

def weighted_average_fusion(preds_by_model, val_rmse):
    weights = np.array([1.0 / max(val_rmse[m], 1e-6) for m in preds_by_model])
    vals = np.array(list(preds_by_model.values()))
    return float(np.average(vals, weights=weights))


# How much the VLM's qualitative evidence is allowed to move a model's fusion weight, as a
# fraction of that weight -- e.g. 0.15 means a model whose own prediction is maximally consistent
# with the VLM's stated band gets up to +15% weight, and a maximally inconsistent one gets up to
# -15%, before renormalization. Fixed and documented, never tuned against test results -- the VLM
# can nudge the fusion; by construction it cannot dominate or override it.
VLM_ADJUSTMENT_BOUND = 0.15
CONFIG_SNAPSHOT["vlm_adjustment_bound"] = VLM_ADJUSTMENT_BOUND
with open(os.path.join(ARTIFACT_ROOT, "summary", "config.json"), "w") as _f:
    json.dump(CONFIG_SNAPSHOT, _f, indent=2)

def constrained_fusion(preds_by_model, val_rmse, vlm_obs=None, vlm_adjustment_bound=VLM_ADJUSTMENT_BOUND):
    """
    Deterministic, fully reproducible fusion of ML/DL predictions. Given the same inputs, this
    ALWAYS returns the same output -- no sampling, no LLM call, no free-form generation anywhere.

    w_i is proportional to 1 / validation_RMSE_i (models with lower held-out validation error get
    more weight), with models flagged as outliers by a Median-Absolute-Deviation rule excluded
    (weight 0). If vlm_obs is given, weights get one additional BOUNDED adjustment
    (VLM_ADJUSTMENT_BOUND) based on how consistent each model's OWN prediction is with the VLM's
    qualitative_rul band -- this is the only way visual evidence enters the numeric result, and it
    is capped so it can only nudge, never override, the validation-driven weighting.

    Weights always satisfy w_i >= 0 and sum(w_i) == 1. final_rul = sum(w_i * pred_i).
    Pass vlm_obs=None for the VLM-free variant ("numerical models only" in the ablation table).
    """
    names = list(preds_by_model.keys())
    vals = np.array([preds_by_model[n] for n in names])

    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) + 1e-6
    z = np.abs(vals - med) / (1.4826 * mad)
    keep_mask = (z < 3.0).astype(float)                      # MAD-based outlier down-weight (0/1)
    if keep_mask.sum() == 0:
        keep_mask = np.ones_like(keep_mask)

    base_w = np.array([1.0 / max(val_rmse[n], 1e-6) for n in names]) * keep_mask

    qual_band = vlm_obs.get("qualitative_rul") if vlm_obs else None
    if vlm_obs is not None and qual_band is not None:
        # Per-model consistency with the VLM's band. NOTE: band_consistency() saturates at 0.0
        # once a prediction is far outside the band, so if EVERY model is far outside (the case
        # where the VLM most strongly disagrees with the numerical branch) the raw scores are all
        # identical and the adjustment would cancel out completely in the renormalization below --
        # i.e. the VLM's evidence would silently have no effect precisely when it dissents most.
        # To avoid that, rank models by their signed distance to the VLM's band instead, and spread
        # the adjustment across that ranking. When the raw scores DO differ, they are used directly.
        per_model_consistency = np.array([band_consistency(preds_by_model[n], qual_band) for n in names])
        if len(names) > 1 and float(np.ptp(per_model_consistency)) < 1e-9:
            lo, hi = QUAL_BAND_TO_RANGE.get(qual_band, (0, 200))
            band_mid = 0.5 * (lo + hi)
            dist = np.array([abs(preds_by_model[n] - band_mid) for n in names])
            if float(np.ptp(dist)) > 1e-9:
                # nearest-to-band -> 1.0, farthest -> 0.0 (relative, not absolute)
                per_model_consistency = 1.0 - (dist - dist.min()) / (dist.max() - dist.min())
            # if distances are also all identical, the models are interchangeable w.r.t. the VLM
            # and leaving the adjustment neutral is the correct behavior.
        adjustment = 1.0 + vlm_adjustment_bound * (2.0 * per_model_consistency - 1.0)  # -> [1-bound, 1+bound]
        weights_raw = base_w * adjustment
    else:
        weights_raw = base_w

    weights = weights_raw / weights_raw.sum()
    fused = float(np.sum(vals * weights))
    down_weighted = [n for n, k in zip(names, keep_mask) if k == 0]
    agreement = float(1.0 - min(vals.std() / (abs(med) + 1e-6), 1.0))
    consistency = band_consistency(fused, qual_band) if vlm_obs is not None else None

    explanation = (
        f"Deterministic fusion over {len(names)} models "
        f"({'with' if vlm_obs is not None else 'without'} VLM evidence). "
        f"{len(down_weighted)} model(s) down-weighted as MAD outliers ({down_weighted or 'none'}). "
        f"Model agreement score={agreement:.2f}."
        + (f" VLM qualitative band='{qual_band}', fused RUL={fused:.1f}, "
           f"band consistency={consistency:.2f}." if vlm_obs is not None else "")
    )
    return {"final_rul": fused, "weights": dict(zip(names, weights.tolist())),
            "down_weighted_models": down_weighted, "agreement": agreement,
            "vlm_consistency": consistency, "used_vlm_evidence": vlm_obs is not None,
            "explanation": explanation}


def compute_confidence_score(agreement, vlm_consistency, vlm_confidence,
                              w_agreement=0.5, w_consistency=0.3, w_vlm_conf=0.2):
    """
    A deterministic RELIABILITY HEURISTIC in [0,1] -- NOT a calibrated probability. Combines,
    with fixed and documented weights (never tuned against test-set outcomes): (1) numerical model
    agreement, (2) consistency between the fused numeric prediction and the VLM's qualitative
    band, and (3) the VLM's own self-reported evidence confidence. See Section 9's reliability
    analysis for how well this score empirically tracks accuracy -- a distinct and weaker claim
    than formal probability calibration.
    """
    vlm_consistency = 0.5 if vlm_consistency is None else vlm_consistency
    vlm_confidence = 0.5 if vlm_confidence is None else vlm_confidence
    score = w_agreement * agreement + w_consistency * vlm_consistency + w_vlm_conf * vlm_confidence
    return float(np.clip(score, 0.0, 1.0))


def reliability_table(df, pred_col, conf_col, n_bins=5, error_threshold=15.0):
    # TERMINOLOGY NOTE: this reports whether a stated confidence score empirically tracks
    # accuracy (a reliability-diagram-style analysis) -- NOT formal probability calibration (e.g.
    # Guo et al. 2017's definition, which requires the score to BE a probability of correctness
    # under a proper scoring rule); we do not claim that here, see compute_confidence_score above.
    # Evaluated on the TEST set, which is standard practice for reporting a fixed method's
    # behavior (same as reporting test RMSE) -- not leakage, since nothing here is fit, tuned, or
    # selected using these numbers; the confidence formula and its weights are fixed in code
    # before any test-set result is computed.
    d = df[[pred_col, conf_col, "true_rul"]].dropna().copy()
    d["abs_err"] = (d[pred_col] - d["true_rul"]).abs()
    d["correct"] = (d["abs_err"] <= error_threshold).astype(int)
    d["bin"] = pd.cut(d[conf_col], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    table = d.groupby("bin", observed=False).agg(mean_confidence=(conf_col, "mean"),
                                                   empirical_accuracy=("correct", "mean"),
                                                   n=("correct", "size"))
    reliability_gap = float((table["n"] / table["n"].sum() *
                              (table["mean_confidence"] - table["empirical_accuracy"]).abs()).sum())
    return table, reliability_gap


DEGRADATION_LEVEL_MAP = {"critical": "CRITICAL", "low": "HIGH", "medium": "MEDIUM", "high": "LOW", None: "UNKNOWN"}

def _fmt(x, spec=".2f", default="n/a"):
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return default

def sensor_arrow(unit_id, sensor, raw_df):
    # Direction is computed from the actual raw sensor data (not asked of the VLM/LLM) -- the VLM
    # tells us WHICH sensors are abnormal, the data tells us the real direction of movement.
    g = raw_df[raw_df.unit == unit_id].sort_values("cycle")
    if sensor not in g.columns or len(g) < 2:
        return "→"
    v = g[sensor].values.astype(float)
    c = g["cycle"].values.astype(float)
    if np.std(v) < 1e-9:
        return "→"
    slope = np.polyfit(c, v, 1)[0]
    return "↑" if slope > 1e-6 else ("↓" if slope < -1e-6 else "→")

print("Metric/fusion-logic functions defined: nasa_score, regression_metrics, band_consistency, "
      "simple/weighted_average_fusion, constrained_fusion, compute_confidence_score, "
      "reliability_table, sensor_arrow")


# %% [markdown]
# ## 8. Per-subset pipeline
#
# `run_subset_pipeline(subset)` runs preprocessing → graph rendering → VLM inference → model
# training → reproducibility check → fusion and evidence analysis → evaluation. All subset-specific
# state is local to the function.


# %%
def run_subset_pipeline(subset):
    t_subset_start = time.time()
    banner(f"[TITLE] {subset} — PIPELINE START", char="#")

    # Quick-validation runs write to a physically separate directory so they can never be mistaken
    # for -- or accidentally satisfy the resume/skip check for -- a real full run of this subset.
    _dir_suffix = "" if MAX_TEST_ENGINES_PER_SUBSET is None else "_quickcheck"
    subset_graph_dir = os.path.join(GRAPH_ROOT, subset + _dir_suffix)
    subset_artifact_dir = os.path.join(ARTIFACT_ROOT, subset + _dir_suffix)
    os.makedirs(subset_graph_dir, exist_ok=True)
    os.makedirs(subset_artifact_dir, exist_ok=True)

    # ---------------- Data loading & preprocessing ----------------
    banner(f"[TITLE] {subset} — Data loading & preprocessing (ML/DL branch)")
    train_df, test_df, rul_df = load_cmapss(subset)
    print(f"{subset}: train {train_df.shape}, test {test_df.shape}, "
          f"{train_df.unit.nunique()} train engines, {test_df.unit.nunique()} test engines, "
          f"regimes={N_REGIMES[subset]}, fault_modes={N_FAULT_MODES[subset]}")

    ACTIVE_SENSORS, DROPPED_SENSORS = drop_low_variance_sensors(train_df)
    print(f"Keeping {len(ACTIVE_SENSORS)} sensors, dropping near-constant "
          f"(train-set variance <= 1e-5): {DROPPED_SENSORS}")

    torch.manual_seed(SEED)
    kmeans_model, regime_scalers = fit_regime_normalizers(train_df, ACTIVE_SENSORS, N_REGIMES[subset], seed=SEED)
    train_norm = apply_regime_normalization(train_df, kmeans_model, regime_scalers, ACTIVE_SENSORS)
    test_norm = apply_regime_normalization(test_df, kmeans_model, regime_scalers, ACTIVE_SENSORS)
    print("Normalizer fit on TRAINING data only; ranges after applying to train (should be ~[0,1] per regime):")
    print(train_norm[ACTIVE_SENSORS].describe().loc[["min", "max"]])

    train_norm = add_piecewise_rul(train_norm, RUL_CAP)
    test_norm = add_test_rul(test_norm, rul_df, RUL_CAP)

    # RUL-cap transparency: both the TRAINING target and the TEST evaluation ground truth are
    # capped at RUL_CAP (standard C-MAPSS convention, Heimes 2008 / Li et al. 2018, used for
    # comparability with published baselines) -- report how many test engines this actually
    # affects so the capping is a visible, deliberate choice rather than a silent side effect.
    _last_test_rows = test_norm.sort_values(["unit", "cycle"]).groupby("unit").tail(1)
    _n_capped_test = int((_last_test_rows["RUL"] >= RUL_CAP).sum())
    print(f"RUL_CAP={RUL_CAP}: {_n_capped_test}/{len(_last_test_rows)} test engines have a true "
          f"remaining life at or above the cap and are evaluated against the capped value "
          f"({RUL_CAP}), not their unbounded true RUL.")

    X_train_seq, y_train_seq, meta_train_seq = create_sequences(train_norm, ACTIVE_SENSORS, WINDOW_SIZE, WINDOW_STRIDE, True)
    X_test_seq, y_test_seq, meta_test_seq = create_sequences(test_norm, ACTIVE_SENSORS, WINDOW_SIZE, WINDOW_STRIDE, False)
    X_train_tab, y_train_tab, _ = create_tabular_features(train_norm, ACTIVE_SENSORS, WINDOW_SIZE, WINDOW_STRIDE, True)
    X_test_tab, y_test_tab, _ = create_tabular_features(test_norm, ACTIVE_SENSORS, WINDOW_SIZE, WINDOW_STRIDE, False)
    test_units_order = [m[0] for m in meta_test_seq]
    if MAX_TEST_ENGINES_PER_SUBSET is not None:
        n_keep = min(MAX_TEST_ENGINES_PER_SUBSET, len(test_units_order))
        X_test_seq, y_test_seq = X_test_seq[:n_keep], y_test_seq[:n_keep]
        X_test_tab, y_test_tab = X_test_tab[:n_keep], y_test_tab[:n_keep]
        test_units_order = test_units_order[:n_keep]
        print(f"!! QUICK-VALIDATION MODE: capped {subset} to the first {n_keep} test engines "
              f"(of {test_norm.unit.nunique()} available). Not a real result.")
    y_true_by_unit = dict(zip(test_units_order, y_test_seq))
    print(f"Sequence shapes: train {X_train_seq.shape}, test {X_test_seq.shape}")
    print(f"Tabular feature shapes: train {X_train_tab.shape}, test {X_test_tab.shape}")

    # ---------------- Graph generation (VLM's separate raw-data path) ----------------
    banner(f"[TITLE] {subset} — Graph generation (VLM's separate, minimally processed path)")
    test_graph_paths = {}
    for unit_id in tqdm(test_units_order, desc=f"{subset}: rendering test-engine graphs"):
        raw_g = test_df[test_df.unit == unit_id]   # raw, un-normalized values for the VLM
        test_graph_paths[unit_id] = generate_engine_graph(raw_g, unit_id, "test", VLM_SENSORS, subset_graph_dir)

    train_units = sorted(train_norm.unit.unique())
    sample_units = train_units if N_TRAIN_ENGINES_FOR_VLM in (-1, None) else \
        random.sample(train_units, min(N_TRAIN_ENGINES_FOR_VLM, len(train_units)))
    train_graph_paths = {}
    for unit_id in tqdm(sample_units, desc=f"{subset}: rendering sampled train-engine graphs (sanity-check only)"):
        raw_g = train_df[train_df.unit == unit_id]
        train_graph_paths[unit_id] = generate_engine_graph(raw_g, unit_id, "train", VLM_SENSORS, subset_graph_dir)
    print(f"Rendered {len(test_graph_paths)} test graphs + {len(train_graph_paths)} train graphs -> {subset_graph_dir}")

    sample_unit = sorted(test_graph_paths.keys())[0]
    print(f"[TITLE] {subset} — preview: test unit {sample_unit}")
    display(Image(filename=test_graph_paths[sample_unit]))

    # ---------------- VLM pass ----------------
    banner(f"[TITLE] {subset} — VLM pass over {len(test_graph_paths)} test engines (real Qwen2.5-VL-7B-Instruct)")
    vlm_results, vlm_raw_text = {}, {}
    for i, (unit_id, path) in enumerate(tqdm(test_graph_paths.items(), desc=f"{subset}: VLM pass")):
        parsed, raw_text = run_vlm(path, unit_id, "test")
        vlm_results[unit_id] = parsed
        vlm_raw_text[unit_id] = raw_text
        if i == 0:
            print(f"--- {subset} first engine (unit {unit_id}) — full structured VLM output ---")
            print(json.dumps(parsed, indent=2))
            print("-" * 60)
        status = "PARSE_FAILED" if parsed.get("_parse_failed") else "ok"
        print(f"  [{subset}] unit {unit_id:>4}: trend={str(parsed.get('degradation_trend')):<12} "
              f"term_phase_start={str(parsed.get('terminal_phase_start_cycle')):<6} "
              f"qual_rul={str(parsed.get('qualitative_rul')):<10} "
              f"conf={str(parsed.get('confidence')):<6} [{status}]")

    n_vlm_failed = sum(bool(v.get("_parse_failed")) for v in vlm_results.values())
    vlm_df = pd.DataFrame(vlm_results.values())
    vlm_df.to_json(os.path.join(subset_artifact_dir, "vlm_outputs.json"), orient="records", indent=2)
    with open(os.path.join(subset_artifact_dir, "vlm_raw_text.json"), "w") as f:
        json.dump({str(k): v for k, v in vlm_raw_text.items()}, f, indent=2)
    print(f"\n[TITLE] {subset} — VLM branch complete: {len(vlm_results)} test engines "
          f"({n_vlm_failed} failed schema validation after {MAX_LLM_RETRIES} retries).")
    display(vlm_df.head(10))

    # ---------------- ML/DL branch ----------------
    banner(f"[TITLE] {subset} — ML/DL branch training (8 models)")
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_groups = np.array([m[0] for m in meta_train_seq])
    tr_idx, val_idx = next(gss.split(X_train_seq, y_train_seq, groups=train_groups))
    print(f"{subset}: train windows={len(tr_idx)}, val windows={len(val_idx)} (split by engine unit, "
          f"{len(set(train_groups[tr_idx]))}/{len(set(train_groups[val_idx]))} units — no engine "
          f"appears in both splits)")

    model_val_rmse, model_test_preds = {}, {}
    Xtr_tab, ytr_tab = X_train_tab[tr_idx], y_train_seq[tr_idx]
    Xval_tab, yval_tab = X_train_tab[val_idx], y_train_seq[val_idx]

    def _fit_eval_tabular(name, model):
        model.fit(Xtr_tab, ytr_tab)
        val_rmse = math.sqrt(mean_squared_error(yval_tab, model.predict(Xval_tab)))
        test_pred = model.predict(X_test_tab)
        model_val_rmse[name] = val_rmse
        model_test_preds[name] = dict(zip(test_units_order, test_pred))
        print(f"  {name:18s} val RMSE = {val_rmse:6.2f}")

    print(f"{subset}: training tabular baselines (LR / RandomForest / XGBoost)...")
    _fit_eval_tabular("LinearRegression", LinearRegression())
    _fit_eval_tabular("RandomForest", RandomForestRegressor(
        n_estimators=300, max_depth=12, random_state=SEED, n_jobs=-1))
    _fit_eval_tabular("XGBoost", xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=SEED, n_jobs=-1))

    n_features = X_train_seq.shape[-1]
    train_loader = DataLoader(SeqDataset(X_train_seq[tr_idx], y_train_seq[tr_idx]), batch_size=128, shuffle=True)
    val_loader = DataLoader(SeqDataset(X_train_seq[val_idx], y_train_seq[val_idx]), batch_size=256, shuffle=False)
    test_loader = DataLoader(SeqDataset(X_test_seq, y_test_seq), batch_size=256, shuffle=False)

    torch.manual_seed(SEED)
    dl_models = build_dl_models(n_features)
    dl_histories = {}
    print(f"\n{subset}: training {len(dl_models)} deep sequence models (BiLSTM, TCN, MS-TCN, "
          f"MS-TCN+BiLSTM, Transformer)...")
    for name, m in dl_models.items():
        trained, val_rmse, hist = train_dl_model(m, name, train_loader, val_loader, seed=SEED)
        model_val_rmse[name] = val_rmse
        dl_histories[name] = hist
        model_test_preds[name] = dict(zip(test_units_order, predict_test(trained, test_loader)))

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, hist in dl_histories.items():
        ax.plot(hist, label=name)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation RMSE"); ax.legend()
    ax.set_title(f"{subset}: DL model training curves")
    plt.show()

    ml_dl_model_names = list(model_test_preds.keys())
    print(f"\n[TITLE] {subset} — validation RMSE by model (lower = better):")
    for k, v in sorted(model_val_rmse.items(), key=lambda kv: kv[1]):
        print(f"  {k:18s} {v:6.2f}")

    # ---------------- Reproducibility check ----------------
    # dl_models holds the SAME (now-trained) model objects that model_test_preds was built from
    # (train_dl_model mutates and returns the model it was given, in place), so this genuinely
    # re-exercises the trained models, not fresh/untrained copies.
    repro_report = verify_reproducibility(subset, dl_models, test_loader,
                                           run_vlm_fn=run_vlm, test_graph_paths=test_graph_paths,
                                           n_vlm_check=2)

    # ---------------- Fusion ----------------
    banner(f"[TITLE] {subset} — Fusion over {len(test_units_order)} test engines "
           f"(deterministic numeric fusion + real Qwen2.5-7B-Instruct evidence analysis)")
    fusion_records = []
    for i, unit_id in enumerate(tqdm(test_units_order, desc=f"{subset}: fusion pass")):
        preds_by_model = {m: float(model_test_preds[m][unit_id]) for m in ml_dl_model_names}
        vlm_obs = vlm_results.get(unit_id)
        true_rul = float(y_true_by_unit[unit_id])

        avg = simple_average_fusion(preds_by_model)
        wavg = weighted_average_fusion(preds_by_model, model_val_rmse)
        # Numeric-only deterministic fusion ("numerical models only" in the ablation table) ...
        constrained = constrained_fusion(preds_by_model, model_val_rmse, vlm_obs=None)
        # ... and the same deterministic mechanism with the VLM's bounded adjustment applied --
        # this second one is the proposed framework's numeric answer.
        constrained_vlm = constrained_fusion(preds_by_model, model_val_rmse, vlm_obs=vlm_obs)
        confidence_score = compute_confidence_score(
            constrained_vlm["agreement"], constrained_vlm["vlm_consistency"],
            vlm_obs.get("confidence") if vlm_obs else None)

        # --- VLM/numeric conflict record (diagnostic only; changes no numeric result) ---
        # Recorded rather than resolved: where the visual evidence disagrees with the numerical
        # consensus we log the disagreement instead of forcing either side to yield.
        _vlm_band = vlm_obs.get("qualitative_rul") if vlm_obs else None
        _band_lo, _band_hi = QUAL_BAND_TO_RANGE.get(_vlm_band, (None, None))
        _numeric_ref = constrained["final_rul"]        # VLM-free number, so this is an honest comparison
        if _vlm_band is None:
            _conflict_status = "vlm_abstained"
            _conflict_margin = None
        elif _band_lo <= _numeric_ref <= _band_hi:
            _conflict_status = "agree"
            _conflict_margin = 0.0
        else:
            _conflict_status = "conflict"
            _conflict_margin = float(min(abs(_numeric_ref - _band_lo), abs(_numeric_ref - _band_hi)))

        # The real LLM analyzes the evidence and the already-computed deterministic result; it
        # does not (and structurally cannot, see FUSION_ALLOWED_KEYS) return a numeric RUL.
        llm_evidence, llm_raw = run_llm_evidence_analysis(
            unit_id, preds_by_model, model_val_rmse, vlm_obs, constrained_vlm)

        if i == 0:
            print(f"--- {subset} first engine (unit {unit_id}) — deterministic fusion result ---")
            print(json.dumps({k: v for k, v in constrained_vlm.items()}, indent=2))
            print(f"--- {subset} first engine (unit {unit_id}) — LLM evidence analysis ---")
            print(json.dumps(llm_evidence, indent=2))
            print("-" * 60)
        status = "PARSE_FAILED" if llm_evidence.get("_parse_failed") else "ok"
        print(f"  [{subset}] unit {unit_id:>4}: true={true_rul:6.1f}  "
              f"ConstrainedFusion+VLM={constrained_vlm['final_rul']:6.1f}  "
              f"confidence_score={confidence_score:.2f}  "
              f"action={llm_evidence.get('recommended_action', 'n/a'):<8}  [{status}]")

        fusion_records.append({
            "unit": unit_id, "true_rul": true_rul,
            **{f"pred_{m}": preds_by_model[m] for m in ml_dl_model_names},
            "pred_SimpleAvg": avg, "pred_WeightedAvg": wavg,
            "pred_ConstrainedFusion": constrained["final_rul"],
            "pred_ConstrainedFusionVLM": constrained_vlm["final_rul"],
            "confidence_score": confidence_score,
            "weights_ConstrainedFusionVLM": json.dumps({k: round(v, 4) for k, v in constrained_vlm["weights"].items()}),
            "down_weighted_ConstrainedFusion": constrained["down_weighted_models"],
            "down_weighted_ConstrainedFusionVLM": constrained_vlm["down_weighted_models"],
            "agreement_ConstrainedFusionVLM": constrained_vlm["agreement"],
            "vlm_consistency_ConstrainedFusionVLM": constrained_vlm["vlm_consistency"],
            "explanation_ConstrainedFusion": constrained["explanation"],
            "explanation_ConstrainedFusionVLM": constrained_vlm["explanation"],
            "llm_parse_failed": bool(llm_evidence.get("_parse_failed", False)) if llm_evidence else True,
            "llm_agreement_assessment": llm_evidence.get("agreement_assessment", "") if llm_evidence else "",
            "llm_vlm_numeric_consistency": llm_evidence.get("vlm_numeric_consistency", "") if llm_evidence else "",
            "llm_reliability_reasoning": llm_evidence.get("reliability_reasoning", "") if llm_evidence else "",
            "llm_recommended_action": llm_evidence.get("recommended_action") if llm_evidence else None,
            "llm_explanation": llm_evidence.get("explanation", "") if llm_evidence else "",
            "vlm_qualitative_rul": vlm_obs.get("qualitative_rul") if vlm_obs else None,
            "vlm_degradation_regime": vlm_obs.get("degradation_regime") if vlm_obs else None,
            "vlm_evidence_quality": vlm_obs.get("evidence_quality") if vlm_obs else None,
            "vlm_confidence": vlm_obs.get("confidence") if vlm_obs else None,
            "vlm_terminal_phase_start_cycle": vlm_obs.get("terminal_phase_start_cycle") if vlm_obs else None,
            "vlm_abnormal_sensor_count": len(vlm_obs.get("abnormal_sensors") or []) if vlm_obs else 0,
            "vlm_numeric_conflict_status": _conflict_status,
            "vlm_numeric_conflict_margin": _conflict_margin,
            "vlm_parse_failed": bool(vlm_obs.get("_parse_failed", False)) if vlm_obs else True,
            "_llm_raw_text": llm_raw,
        })

    fusion_df = pd.DataFrame(fusion_records)
    # raw LLM evidence-analysis text is saved as its own artifact (matching how raw VLM text is
    # saved), not left only inside the main results table.
    with open(os.path.join(subset_artifact_dir, "fusion_llm_raw_text.json"), "w") as f:
        json.dump({str(r["unit"]): r["_llm_raw_text"] for r in fusion_records}, f, indent=2)
    fusion_df = fusion_df.drop(columns=["_llm_raw_text"])

    for m in ml_dl_model_names + ["SimpleAvg", "WeightedAvg", "ConstrainedFusion", "ConstrainedFusionVLM"]:
        col = f"pred_{m}"
        if col in fusion_df.columns:
            fusion_df[f"abs_err_{m}"] = (fusion_df[col] - fusion_df["true_rul"]).abs()
    fusion_df.to_csv(os.path.join(subset_artifact_dir, "fusion_results.csv"), index=False)

    n_llm_failed = int(fusion_df["llm_parse_failed"].sum())
    print(f"\n[TITLE] {subset} — fusion complete: {len(fusion_df)} test engines "
          f"({n_llm_failed} LLM evidence-analysis responses failed schema validation after "
          f"{MAX_LLM_RETRIES} retries -- this affects only the qualitative analysis fields, "
          f"never the numeric RUL, which is always computed deterministically).")
    display(fusion_df[["unit", "true_rul", "pred_SimpleAvg", "pred_WeightedAvg", "pred_ConstrainedFusion",
                        "pred_ConstrainedFusionVLM", "confidence_score", "llm_recommended_action"]].head(10))

    return _evaluate_and_save_subset(
        subset, subset_artifact_dir, t_subset_start, train_df, test_df, test_norm, ACTIVE_SENSORS,
        DROPPED_SENSORS, ml_dl_model_names, model_val_rmse, fusion_df, vlm_df, vlm_results,
        test_graph_paths, n_vlm_failed, n_llm_failed, repro_report,
    )

print("run_subset_pipeline(subset) defined (data -> graphs -> VLM -> ML/DL -> fusion; "
      "evaluation/saving delegated to _evaluate_and_save_subset defined next).")


# %% [markdown]
# ## 9. Evaluation and artifacts
#
# For each subset:
# - Accuracy metrics for all 12 methods (8 models + 4 ensemble/fusion methods)
# - Ablation: best single model → SimpleAvg → WeightedAvg → constrained fusion → + VLM evidence
# - Confidence–accuracy reliability analysis
# - VLM terminal-phase estimate vs. an RUL ≤ 30 operational proxy
# - Outlier-detection rate and distribution of LLM recommendations
# - Wilcoxon signed-rank and paired t-tests against the best single model
# - Case reports for the best, median, and worst engines
# - VLM effect diagnostics (helped / hurt / no effect; agreement with the numeric consensus)
#
# Results are saved to `/content/artifacts/<SUBSET>/`.


# %%
def _evaluate_and_save_subset(subset, subset_artifact_dir, t_subset_start, train_df, test_df,
                               test_norm, ACTIVE_SENSORS, DROPPED_SENSORS, ml_dl_model_names,
                               model_val_rmse, fusion_df, vlm_df, vlm_results, test_graph_paths,
                               n_vlm_failed, n_llm_failed, repro_report):
    banner(f"[TITLE] {subset} — Evaluation")

    # ---------------- 9.1 Accuracy metrics, all methods ----------------
    PROPOSED_METHOD = "ConstrainedFusion+VLM (proposed)"
    method_pred_cols = {**{m: f"pred_{m}" for m in ml_dl_model_names},
                         "SimpleAvg_Ensemble": "pred_SimpleAvg",
                         "WeightedAvg_Ensemble": "pred_WeightedAvg",
                         "ConstrainedFusion (numeric only)": "pred_ConstrainedFusion",
                         PROPOSED_METHOD: "pred_ConstrainedFusionVLM"}
    metrics_rows = []
    for method, col in method_pred_cols.items():
        mrow = regression_metrics(fusion_df["true_rul"], fusion_df[col])
        mrow["Method"] = method
        metrics_rows.append(mrow)
    metrics_table = pd.DataFrame(metrics_rows).set_index("Method")[["RMSE", "MAE", "NASA_score", "Pearson_r"]]
    metrics_table = metrics_table.sort_values("RMSE")
    print(f"[TITLE] {subset} — accuracy metrics by method "
          f"({len(ml_dl_model_names)} individual models + 4 ensemble/fusion methods)")
    display(metrics_table.round(3))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, metric in zip(axes, ["RMSE", "MAE", "NASA_score"]):
        metrics_table[metric].sort_values().plot(kind="barh", ax=ax, color="#4c72b0")
        ax.set_title(metric); ax.set_xlabel(metric)
    fig.suptitle(f"{subset}: accuracy comparison across all methods (lower = better)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

    # ---------------- 9.2 Ablation study ----------------
    # Isolates each ingredient's contribution. Reported honestly: if adding the VLM or the
    # constrained fusion does not help on this subset, the delta column shows that as a negative
    # number rather than being hidden or reframed.
    best_single_model = metrics_table.loc[
        [m for m in metrics_table.index if m in ml_dl_model_names], "RMSE"].idxmin()
    ablation_stages = [
        (f"Best individual model ({best_single_model})", f"pred_{best_single_model}"),
        ("SimpleAvg ensemble", "pred_SimpleAvg"),
        ("WeightedAvg ensemble (val-RMSE weights)", "pred_WeightedAvg"),
        ("Constrained fusion, numerical models only", "pred_ConstrainedFusion"),
        ("Constrained fusion + VLM evidence (full framework)", "pred_ConstrainedFusionVLM"),
    ]
    ablation_rows = []
    _baseline_rmse = None
    for stage_name, col in ablation_stages:
        m = regression_metrics(fusion_df["true_rul"], fusion_df[col])
        if _baseline_rmse is None:
            _baseline_rmse = m["RMSE"]
        ablation_rows.append({
            "Stage": stage_name, "RMSE": m["RMSE"], "MAE": m["MAE"],
            "NASA_score": m["NASA_score"], "Pearson_r": m["Pearson_r"],
            "RMSE_vs_best_single": m["RMSE"] - _baseline_rmse,
        })
    ablation_table = pd.DataFrame(ablation_rows).set_index("Stage")
    print(f"\n[TITLE] {subset} — ablation study "
          f"(RMSE_vs_best_single: negative = better than the best individual model)")
    display(ablation_table.round(3))

    # ---------------- 9.3 Confidence-reliability analysis ----------------
    # NOT probability calibration -- see compute_confidence_score / reliability_table in Section 7.
    conf_table, conf_reliability_gap = reliability_table(fusion_df, "pred_ConstrainedFusionVLM",
                                                          "confidence_score")
    print(f"\n[TITLE] {subset} — confidence-reliability analysis "
          f"(does the deterministic confidence SCORE track empirical accuracy? "
          f"NOT a calibrated probability)")
    print(f"  Mean |stated confidence - empirical accuracy| across bins "
          f"(reliability gap, |err|<=15 cycles): {conf_reliability_gap:.3f}")
    display(conf_table.round(3))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="score == empirical accuracy")
    ax.plot(conf_table["mean_confidence"], conf_table["empirical_accuracy"],
            marker="o", label="ConstrainedFusion+VLM confidence score")
    ax.set_xlabel("stated confidence score (not a probability)")
    ax.set_ylabel("empirical accuracy (|err| <= 15 cycles)")
    ax.set_title(f"{subset}: confidence-reliability diagram")
    ax.legend(); plt.show()

    # ---------------- 9.4 Terminal-phase-start detection ----------------
    # TERMINAL_RUL_THRESHOLD identifies a terminal/critical RUL region and is used as an
    # OPERATIONAL PROXY, not as ground-truth physical degradation onset. C-MAPSS provides no
    # onset label, and no such label is fabricated here: this compares the VLM's visual
    # terminal-phase-start estimate against a threshold-defined reference, and both are
    # operational definitions rather than physical ground truth.
    TERMINAL_RUL_THRESHOLD = 30
    def _proxy_terminal_phase_start(unit_id):
        g = test_norm[test_norm.unit == unit_id].sort_values("cycle")
        hit = g[g["RUL"] <= TERMINAL_RUL_THRESHOLD]
        return int(hit["cycle"].iloc[0]) if len(hit) else None

    tp_rows = [{"unit": u, "proxy_terminal_phase_start": _proxy_terminal_phase_start(u),
                "vlm_terminal_phase_start": vlm_results.get(u, {}).get("terminal_phase_start_cycle")}
               for u in fusion_df["unit"]]
    terminal_phase_df = pd.DataFrame(tp_rows)
    tp_detected = terminal_phase_df.dropna().copy()
    if len(tp_detected):
        tp_detected["abs_error_cycles"] = (
            tp_detected["proxy_terminal_phase_start"] - tp_detected["vlm_terminal_phase_start"]).abs()

    print(f"\n[TITLE] {subset} — terminal-phase-start detection "
          f"(VLM visual estimate vs. RUL<={TERMINAL_RUL_THRESHOLD} operational proxy; "
          f"neither is ground-truth physical onset)")
    print(f"  VLM gave a non-null terminal_phase_start_cycle for "
          f"{len(tp_detected)}/{len(terminal_phase_df)} engines.")
    if len(tp_detected):
        _signed = (tp_detected["proxy_terminal_phase_start"] - tp_detected["vlm_terminal_phase_start"]).mean()
        print(f"  Mean |difference| vs. proxy: {tp_detected['abs_error_cycles'].mean():.1f} cycles "
              f"(signed mean: {_signed:+.1f}; positive = VLM flags the phase EARLIER than the "
              f"RUL<={TERMINAL_RUL_THRESHOLD} proxy, which is expected if it detects drift before "
              f"the terminal band and is not by itself an error)")
        print(f"  Within +/-20 cycles of the proxy: {(tp_detected['abs_error_cycles'] <= 20).mean()*100:.1f}%")

        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(tp_detected["proxy_terminal_phase_start"], tp_detected["vlm_terminal_phase_start"], alpha=0.6)
        lims = [0, max(tp_detected["proxy_terminal_phase_start"].max(),
                       tp_detected["vlm_terminal_phase_start"].max()) + 10]
        ax.plot(lims, lims, "k--", alpha=0.4, label="exact agreement with proxy")
        ax.set_xlabel(f"operational proxy: first cycle with RUL <= {TERMINAL_RUL_THRESHOLD}")
        ax.set_ylabel("VLM-estimated terminal_phase_start_cycle")
        ax.set_title(f"{subset}: VLM terminal-phase estimate vs. operational proxy")
        ax.legend(); plt.show()
    else:
        print("  No engines had both a proxy value and a non-null VLM estimate — skipping the scatter plot.")

    # ---------------- 9.5 Disagreement detection ----------------
    def _worst_model_per_unit(row):
        errs = {m: abs(row[f"pred_{m}"] - row["true_rul"]) for m in ml_dl_model_names}
        return max(errs, key=errs.get)

    fusion_df["actual_worst_model"] = fusion_df.apply(_worst_model_per_unit, axis=1)
    fusion_df["fusion_flagged_worst_correctly"] = fusion_df.apply(
        lambda r: r["actual_worst_model"] in r["down_weighted_ConstrainedFusionVLM"], axis=1)
    _action_counts = fusion_df["llm_recommended_action"].value_counts(dropna=False).to_dict()
    print(f"\n[TITLE] {subset} — disagreement detection")
    print(f"  Deterministic MAD rule down-weighted the engine's actual worst model in "
          f"{fusion_df['fusion_flagged_worst_correctly'].mean()*100:.1f}% of engines.")
    print(f"  (Not a failure by itself: when all models agree closely, no model IS an outlier, so "
          f"down-weighting nothing is the correct behavior. Cross-reference "
          f"'agreement_ConstrainedFusionVLM' before interpreting this number.)")
    print(f"  LLM evidence-analysis recommended_action distribution: {_action_counts}")

    # ---------------- 9.6 Statistical significance ----------------
    print(f"\n[TITLE] {subset} — statistical significance vs. best single model ({best_single_model})")
    sig_rows = []
    for fusion_col, fusion_name in [("pred_ConstrainedFusion", "ConstrainedFusion (numeric only)"),
                                     ("pred_ConstrainedFusionVLM", PROPOSED_METHOD)]:
        err_best = (fusion_df[f"pred_{best_single_model}"] - fusion_df["true_rul"]).abs()
        err_fusion = (fusion_df[fusion_col] - fusion_df["true_rul"]).abs()
        # Wilcoxon requires at least one non-zero difference; identical arrays raise.
        if float(np.max(np.abs(err_best.values - err_fusion.values))) < 1e-12:
            w_p, t_p = float("nan"), float("nan")
            print(f"  {fusion_name} vs {best_single_model}: predictions identical — no test performed.")
        else:
            _, w_p = stats.wilcoxon(err_best, err_fusion)
            _, t_p = stats.ttest_rel(err_best, err_fusion)
            direction = "better" if err_fusion.mean() < err_best.mean() else "worse"
            print(f"  {fusion_name} vs {best_single_model}: mean|err| {err_fusion.mean():.2f} vs "
                  f"{err_best.mean():.2f} ({direction}) | Wilcoxon p={w_p:.4f} | paired t-test p={t_p:.4f}")
        sig_rows.append({"method": fusion_name, "vs": best_single_model,
                          "wilcoxon_p": float(w_p), "ttest_p": float(t_p),
                          "mean_abs_err_fusion": float(err_fusion.mean()),
                          "mean_abs_err_baseline": float(err_best.mean())})
    sig_df = pd.DataFrame(sig_rows)

    # ---------------- 9.7 Per-engine narrative report ----------------
    def render_engine_report(unit_id, show_plot=True):
        row = fusion_df[fusion_df.unit == unit_id].iloc[0]
        vlm_obs = vlm_results.get(unit_id, {}) or {}

        deg_level = DEGRADATION_LEVEL_MAP.get(vlm_obs.get("qualitative_rul"), "UNKNOWN")
        tp_start = vlm_obs.get("terminal_phase_start_cycle")
        trend_word = vlm_obs.get("degradation_trend")
        if tp_start and trend_word:
            trend_desc = f"{str(trend_word).capitalize()} trend; terminal phase estimated from cycle {tp_start}"
        elif trend_word:
            trend_desc = f"{str(trend_word).capitalize()} trend; no terminal-phase start identified"
        else:
            trend_desc = "Insufficient visual evidence for a trend judgment (VLM returned null)"
        abnormal = vlm_obs.get("abnormal_sensors") or []
        sensor_line = (", ".join(f"{s} {sensor_arrow(unit_id, s, test_df)}" for s in abnormal)
                       if abnormal else "none flagged")
        rul_lo, rul_hi = QUAL_BAND_TO_RANGE.get(vlm_obs.get("qualitative_rul"), (0, 200))

        print("=" * 68)
        print(f"[TITLE] {subset} — ENGINE UNIT {unit_id} case report")
        print("=" * 68)

        print("\n[1] VLM VISUAL EVIDENCE (qualitative only — no numeric RUL from the VLM):")
        print(f"  Degradation level:        {deg_level}")
        print(f"  Trend:                    {trend_desc}")
        print(f"  Flagged sensors:          {sensor_line}")
        print(f"  Degradation regime:       {vlm_obs.get('degradation_regime') or 'n/a'}")
        print(f"  Qualitative RUL band:     {vlm_obs.get('qualitative_rul') or 'n/a'} "
              f"(scoring-convention range {rul_lo}-{rul_hi} cycles)")
        print(f"  Figure evidence quality:  {vlm_obs.get('evidence_quality') or 'n/a'}")
        print(f"  VLM evidence confidence:  {_fmt(vlm_obs.get('confidence'))} "
              f"(qualitative indicator, NOT a calibrated probability)")
        print(f"  vs. numerical consensus:  {row.get('vlm_numeric_conflict_status') or 'n/a'}"
              + (f" (band is {_fmt(row.get('vlm_numeric_conflict_margin'), '.1f')} cycles from the "
                 f"numeric estimate)" if row.get('vlm_numeric_conflict_status') == 'conflict' else ""))

        print("\n[2] INDIVIDUAL ML/DL MODEL PREDICTIONS (cycles):")
        model_preds = {}
        for name in ml_dl_model_names:
            pred = row.get(f"pred_{name}", np.nan)
            model_preds[name] = float(pred) if pred == pred else np.nan
            print(f"  {name:18s} {_fmt(pred, '.0f'):>6}   (val RMSE {model_val_rmse.get(name, float('nan')):.2f})")

        print("\n[3] DETERMINISTIC CONSTRAINED FUSION (Python computes this number, not an LLM):")
        try:
            _weights = json.loads(row.get("weights_ConstrainedFusionVLM") or "{}")
        except (TypeError, ValueError):
            _weights = {}
        _top_w = sorted(_weights.items(), key=lambda kv: kv[1], reverse=True)[:4]
        print(f"  Final RUL:                {_fmt(row.get('pred_ConstrainedFusionVLM'), '.1f')} cycles "
              f"(true RUL: {_fmt(row.get('true_rul'), '.0f')})")
        print(f"  Numeric-only variant:     {_fmt(row.get('pred_ConstrainedFusion'), '.1f')} cycles "
              f"(same mechanism, VLM evidence withheld)")
        print(f"  Highest fusion weights:   " +
              ", ".join(f"{k}={v:.3f}" for k, v in _top_w) if _top_w else "  (weights unavailable)")
        print(f"  Down-weighted as outlier: {row.get('down_weighted_ConstrainedFusionVLM') or 'none'}")
        print(f"  Model agreement:          {_fmt(row.get('agreement_ConstrainedFusionVLM'))}")
        print(f"  VLM band consistency:     {_fmt(row.get('vlm_consistency_ConstrainedFusionVLM'))}")
        print(f"  Confidence SCORE:         {_fmt(row.get('confidence_score'))} "
              f"(deterministic heuristic, not a calibrated probability)")

        print("\n[4] LLM EVIDENCE ANALYSIS (commentary only — cannot change the number above):")
        print(f"  Recommended action:       {row.get('llm_recommended_action') or 'n/a'}")
        print(f"  Model agreement:          {row.get('llm_agreement_assessment') or 'n/a'}")
        print(f"  Visual/numeric consistency: {row.get('llm_vlm_numeric_consistency') or 'n/a'}")
        print(f"  Reliability reasoning:    {row.get('llm_reliability_reasoning') or 'n/a'}")
        print(f"  Summary:                  {row.get('llm_explanation') or 'n/a'}")

        if show_plot:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            names = list(model_preds.keys())
            vals = [model_preds[n] for n in names]
            axes[0].bar(names, vals, color="#4c72b0")
            axes[0].axhline(row["true_rul"], color="black", linestyle="--",
                             label=f"True RUL = {row['true_rul']:.0f}")
            if row.get("pred_ConstrainedFusionVLM") == row.get("pred_ConstrainedFusionVLM"):
                axes[0].scatter([len(names) - 0.5], [row["pred_ConstrainedFusionVLM"]], color="red",
                                 zorder=5, s=110,
                                 label=f"Constrained fusion+VLM = {row['pred_ConstrainedFusionVLM']:.0f}")
            axes[0].set_ylabel("Predicted RUL (cycles)")
            axes[0].set_title(f"{subset} unit {unit_id}: individual models vs. deterministic fusion")
            axes[0].tick_params(axis="x", rotation=30)
            axes[0].legend()

            g = test_df[test_df.unit == unit_id].sort_values("cycle")
            sensors_to_plot = abnormal if abnormal else VLM_SENSORS[:4]
            for s in sensors_to_plot:
                if s in g.columns:
                    axes[1].plot(g["cycle"], g[s], label=s, linewidth=1.5)
            if tp_start:
                axes[1].axvline(tp_start, color="red", linestyle="--",
                                 label=f"VLM terminal-phase start = {tp_start}")
            axes[1].set_xlabel("cycle"); axes[1].set_ylabel("raw sensor value")
            axes[1].set_title(f"{subset} unit {unit_id}: VLM-flagged sensors (raw values)")
            axes[1].legend(fontsize=8)
            fig.tight_layout(); plt.show()

        return {"unit": unit_id, "vlm": vlm_obs, "model_preds": model_preds,
                "final_rul": row.get("pred_ConstrainedFusionVLM"),
                "confidence_score": row.get("confidence_score"),
                "llm_recommended_action": row.get("llm_recommended_action")}

    _abs_err = (fusion_df["pred_ConstrainedFusionVLM"] - fusion_df["true_rul"]).abs()
    best_case = fusion_df.loc[_abs_err.idxmin()]
    worst_case = fusion_df.loc[_abs_err.idxmax()]
    median_idx = (_abs_err - _abs_err.median()).abs().idxmin()
    median_case = fusion_df.loc[median_idx]

    print(f"\n[TITLE] {subset} — narrative case reports: BEST / TYPICAL(median-error) / WORST engine "
          f"(selected by the proposed method's absolute error)")
    for uid in [int(best_case["unit"]), int(median_case["unit"]), int(worst_case["unit"])]:
        render_engine_report(uid)
        print()

    # ---------------- 9.9 VLM evidence analysis: agreement, conflict, help/hurt -------------
    # Diagnostic only. Compares the VLM-free number (pred_ConstrainedFusion) against the
    # VLM-informed number (pred_ConstrainedFusionVLM) per engine to show WHERE the visual
    # evidence helped, hurt, or did nothing -- reported honestly, including when the net effect
    # is ~zero. No predictions are recomputed here and no method is re-selected using these
    # numbers; this block only summarises results already fixed above.
    banner(f"[TITLE] {subset} — VLM evidence diagnostics (agreement / conflict / help-hurt)")

    _err_no_vlm = (fusion_df["pred_ConstrainedFusion"] - fusion_df["true_rul"]).abs()
    _err_vlm = (fusion_df["pred_ConstrainedFusionVLM"] - fusion_df["true_rul"]).abs()
    fusion_df["vlm_delta_abs_err"] = _err_vlm - _err_no_vlm        # negative => VLM helped
    _EPS = 1e-9
    fusion_df["vlm_effect"] = np.where(fusion_df["vlm_delta_abs_err"] < -_EPS, "helped",
                               np.where(fusion_df["vlm_delta_abs_err"] > _EPS, "hurt", "no_effect"))

    _effect_counts = fusion_df["vlm_effect"].value_counts().to_dict()
    _n = len(fusion_df)
    print(f"Per-engine effect of adding VLM evidence to the deterministic fusion (n={_n}):")
    for k in ["helped", "hurt", "no_effect"]:
        c = _effect_counts.get(k, 0)
        print(f"  {k:10s} {c:4d}  ({c/_n*100:5.1f}%)")
    print(f"  mean change in |error| when VLM evidence is added: "
          f"{fusion_df['vlm_delta_abs_err'].mean():+.4f} cycles "
          f"(negative = VLM improved accuracy; a value near 0 means no measurable effect)")
    _helped = fusion_df.loc[fusion_df["vlm_effect"] == "helped", "vlm_delta_abs_err"]
    _hurt = fusion_df.loc[fusion_df["vlm_effect"] == "hurt", "vlm_delta_abs_err"]
    if len(_helped):
        print(f"  when it helped: mean {_helped.mean():+.3f}, best {_helped.min():+.3f} cycles")
    if len(_hurt):
        print(f"  when it hurt:   mean {_hurt.mean():+.3f}, worst {_hurt.max():+.3f} cycles")

    print(f"\n[TITLE] {subset} — VLM vs. numerical-consensus agreement")
    _conflict_counts = fusion_df["vlm_numeric_conflict_status"].value_counts().to_dict()
    for k, v in _conflict_counts.items():
        print(f"  {str(k):14s} {v:4d}  ({v/_n*100:5.1f}%)")
    _conf_rows = fusion_df[fusion_df["vlm_numeric_conflict_status"] == "conflict"]
    if len(_conf_rows):
        print(f"  among conflicts: mean distance from the VLM band to the numeric estimate = "
              f"{_conf_rows['vlm_numeric_conflict_margin'].mean():.1f} cycles")
        _conf_err = (_conf_rows["pred_ConstrainedFusion"] - _conf_rows["true_rul"]).abs().mean()
        _agree_rows = fusion_df[fusion_df["vlm_numeric_conflict_status"] == "agree"]
        if len(_agree_rows):
            _agree_err = (_agree_rows["pred_ConstrainedFusion"] - _agree_rows["true_rul"]).abs().mean()
            print(f"  numeric |error| on conflict engines = {_conf_err:.2f} vs "
                  f"{_agree_err:.2f} on agreement engines "
                  f"({'conflict flags harder engines' if _conf_err > _agree_err else 'no such pattern'})")

    print(f"\n[TITLE] {subset} — VLM evidence-quality and regime breakdown")
    for col in ["vlm_evidence_quality", "vlm_degradation_regime", "vlm_qualitative_rul"]:
        if col in fusion_df.columns:
            print(f"  {col}: {fusion_df[col].value_counts(dropna=False).to_dict()}")
    _tp_rate = fusion_df["vlm_terminal_phase_start_cycle"].notna().mean()
    print(f"  terminal_phase_start_cycle reported for {_tp_rate*100:.1f}% of engines")
    if "vlm_evidence_quality" in fusion_df.columns:
        _unreadable = (fusion_df["vlm_evidence_quality"] == "unreadable").mean()
        print(f"  figures the VLM judged unreadable: {_unreadable*100:.1f}% "
              f"(high values on multi-regime subsets are an honest finding, not a bug)")

    # worked examples: clearest help and clearest hurt on THIS subset
    vlm_example_units = {}
    if (fusion_df["vlm_effect"] == "helped").any():
        vlm_example_units["most_helped"] = int(fusion_df.loc[fusion_df["vlm_delta_abs_err"].idxmin(), "unit"])
    if (fusion_df["vlm_effect"] == "hurt").any():
        vlm_example_units["most_hurt"] = int(fusion_df.loc[fusion_df["vlm_delta_abs_err"].idxmax(), "unit"])
    if len(_conf_rows):
        vlm_example_units["largest_conflict"] = int(
            _conf_rows.loc[_conf_rows["vlm_numeric_conflict_margin"].idxmax(), "unit"])
    print(f"\n[TITLE] {subset} — worked VLM examples: {vlm_example_units}")
    for label, uid in vlm_example_units.items():
        r = fusion_df[fusion_df.unit == uid].iloc[0]
        print(f"  [{label}] unit {uid}: true={r['true_rul']:.0f}, "
              f"numeric-only={r['pred_ConstrainedFusion']:.1f}, +VLM={r['pred_ConstrainedFusionVLM']:.1f}, "
              f"delta|err|={r['vlm_delta_abs_err']:+.3f}, VLM band={r['vlm_qualitative_rul']}, "
              f"regime={r['vlm_degradation_regime']}, quality={r['vlm_evidence_quality']}, "
              f"status={r['vlm_numeric_conflict_status']}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    _order = ["helped", "no_effect", "hurt"]
    _vals = [_effect_counts.get(k, 0) for k in _order]
    axes[0].bar(_order, _vals, color=["#2a9d4a", "#999999", "#c0392b"])
    axes[0].set_ylabel("engines"); axes[0].set_title(f"{subset}: per-engine effect of VLM evidence")
    for i, v in enumerate(_vals):
        axes[0].text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    axes[1].hist(fusion_df["vlm_delta_abs_err"], bins=30, color="#4c72b0")
    axes[1].axvline(0, color="black", linestyle="--", label="no effect")
    axes[1].set_xlabel("change in |error| when VLM evidence added (negative = better)")
    axes[1].set_ylabel("engines"); axes[1].set_title(f"{subset}: distribution of VLM effect")
    axes[1].legend()
    fig.tight_layout(); plt.show()

    vlm_diagnostics = {
        "n_engines": int(_n),
        "effect_counts": {k: int(v) for k, v in _effect_counts.items()},
        "mean_delta_abs_err": float(fusion_df["vlm_delta_abs_err"].mean()),
        "conflict_counts": {str(k): int(v) for k, v in _conflict_counts.items()},
        "terminal_phase_report_rate": float(_tp_rate),
        "evidence_quality_counts": (fusion_df["vlm_evidence_quality"].value_counts(dropna=False)
                                     .rename(index=str).to_dict()
                                     if "vlm_evidence_quality" in fusion_df.columns else {}),
        "degradation_regime_counts": (fusion_df["vlm_degradation_regime"].value_counts(dropna=False)
                                       .rename(index=str).to_dict()
                                       if "vlm_degradation_regime" in fusion_df.columns else {}),
        "worked_examples": vlm_example_units,
    }

    # ---------------- 9.8 Save artifacts ----------------
    banner(f"[TITLE] {subset} — saving artifacts")
    metrics_table.to_csv(os.path.join(subset_artifact_dir, "metrics_table.csv"))
    ablation_table.to_csv(os.path.join(subset_artifact_dir, "ablation_table.csv"))
    conf_table.to_csv(os.path.join(subset_artifact_dir, "confidence_reliability_analysis.csv"))
    terminal_phase_df.to_csv(os.path.join(subset_artifact_dir, "terminal_phase_eval.csv"), index=False)
    sig_df.to_csv(os.path.join(subset_artifact_dir, "significance_tests.csv"), index=False)
    fusion_df.to_csv(os.path.join(subset_artifact_dir, "fusion_results_full.csv"), index=False)
    pd.DataFrame([{"model": k, "validation_rmse": v} for k, v in model_val_rmse.items()]).to_csv(
        os.path.join(subset_artifact_dir, "model_validation_rmse.csv"), index=False)
    with open(os.path.join(subset_artifact_dir, "reproducibility_check.json"), "w") as f:
        json.dump(repro_report, f, indent=2)
    with open(os.path.join(subset_artifact_dir, "vlm_diagnostics.json"), "w") as f:
        json.dump(vlm_diagnostics, f, indent=2, default=str)

    summary = {
        "subset": subset, "quick_validation": MAX_TEST_ENGINES_PER_SUBSET is not None,
        "n_test_engines": len(fusion_df), "n_train_engines": int(train_df.unit.nunique()),
        "vlm_sensors_all": VLM_SENSORS,
        "ml_dl_active_sensors": ACTIVE_SENSORS, "ml_dl_dropped_sensors": DROPPED_SENSORS,
        "rul_cap": RUL_CAP, "window_size": WINDOW_SIZE, "seed": SEED,
        "vlm_model": VLM_MODEL_ID, "evidence_analysis_llm_model": FUSION_LLM_MODEL_ID,
        "final_rul_source": "deterministic constrained_fusion() in Python; no LLM-generated RUL",
        "vlm_adjustment_bound": VLM_ADJUSTMENT_BOUND,
        "model_validation_rmse": {k: float(v) for k, v in model_val_rmse.items()},
        "n_vlm_schema_failures": int(n_vlm_failed),
        "n_llm_evidence_analysis_schema_failures": int(n_llm_failed),
        "best_single_model": best_single_model,
        "proposed_method": PROPOSED_METHOD,
        "metrics_table": metrics_table.round(3).to_dict(orient="index"),
        "ablation_table": ablation_table.round(3).to_dict(orient="index"),
        "confidence_reliability_gap": conf_reliability_gap,
        "terminal_phase_threshold_rul": TERMINAL_RUL_THRESHOLD,
        "terminal_phase_detected_count": int(len(tp_detected)),
        "terminal_phase_mean_abs_diff_cycles": (float(tp_detected["abs_error_cycles"].mean())
                                                 if len(tp_detected) else None),
        "disagreement_detection_rate_deterministic": float(fusion_df["fusion_flagged_worst_correctly"].mean()),
        "llm_recommended_action_counts": {str(k): int(v) for k, v in _action_counts.items()},
        "significance_tests": sig_rows,
        "vlm_diagnostics": vlm_diagnostics,
        "reproducibility_check": repro_report,
        "elapsed_seconds": time.time() - t_subset_start,
    }
    with open(os.path.join(subset_artifact_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"{subset}: all artifacts saved under {subset_artifact_dir}/")
    print(f"  metrics_table.csv, ablation_table.csv, confidence_reliability_analysis.csv,")
    print(f"  terminal_phase_eval.csv, significance_tests.csv, fusion_results_full.csv,")
    print(f"  model_validation_rmse.csv, vlm_outputs.json, vlm_raw_text.json,")
    print(f"  fusion_llm_raw_text.json, reproducibility_check.json, run_summary.json")
    print(f"{subset}: pipeline finished in {summary['elapsed_seconds']/60:.1f} minutes.")
    banner(f"[TITLE] {subset} — PIPELINE COMPLETE", char="#")

    return {"subset": subset, "metrics_table": metrics_table, "ablation_table": ablation_table,
            "fusion_df": fusion_df, "vlm_df": vlm_df, "terminal_phase_df": terminal_phase_df,
            "sig_df": sig_df, "summary": summary, "repro_report": repro_report,
            "vlm_diagnostics": vlm_diagnostics,
            "render_engine_report": render_engine_report}

print("_evaluate_and_save_subset(...) defined (called automatically at the end of run_subset_pipeline).")


# %% [markdown]
# ## 10. Run all subsets
#
# Runs FD001–FD004 in sequence. With `SKIP_ALREADY_COMPLETED_SUBSETS = True`, subsets that already
# have a `run_summary.json` are loaded from disk instead of being re-run.


# %%
banner("[TITLE] FULL EXPERIMENT — running all subsets", char="#")
print(f"Subsets to process: {SUBSETS}")
print(f"Total test engines across all subsets: {int(overview_df['test_engines'].sum())} "
      f"(each gets one real VLM call and one real fusion-LLM call).")
print(f"SKIP_ALREADY_COMPLETED_SUBSETS = {SKIP_ALREADY_COMPLETED_SUBSETS}")

t_all_start = time.time()
all_results = {}

for subset in SUBSETS:
    _summary_path = os.path.join(ARTIFACT_ROOT, subset, "run_summary.json")
    _fusion_path = os.path.join(ARTIFACT_ROOT, subset, "fusion_results_full.csv")
    if SKIP_ALREADY_COMPLETED_SUBSETS and os.path.exists(_summary_path) and os.path.exists(_fusion_path):
        banner(f"[TITLE] {subset} — SKIPPED (run_summary.json already exists)", char="#")
        print(f"Loading existing results for {subset} from {os.path.dirname(_summary_path)} "
              f"instead of re-running. Set SKIP_ALREADY_COMPLETED_SUBSETS = False in the config "
              f"cell (Section 0) and re-run this cell to force a clean re-run of {subset}.")
        with open(_summary_path) as f:
            _existing_summary = json.load(f)
        _fusion_df = pd.read_csv(_fusion_path)
        _metrics_table = pd.DataFrame(_existing_summary["metrics_table"]).T
        _metrics_table.index.name = "Method"
        _ablation_table = pd.DataFrame(_existing_summary.get("ablation_table", {})).T
        if len(_ablation_table):
            _ablation_table.index.name = "Stage"
        all_results[subset] = {"subset": subset, "metrics_table": _metrics_table,
                                "ablation_table": _ablation_table, "fusion_df": _fusion_df,
                                "summary": _existing_summary,
                                "repro_report": _existing_summary.get("reproducibility_check")}
        continue
    all_results[subset] = run_subset_pipeline(subset)

t_all_elapsed = time.time() - t_all_start
banner("[TITLE] FULL EXPERIMENT — all subsets complete", char="#")
print(f"Total elapsed this run: {t_all_elapsed/60:.1f} minutes across {len(SUBSETS)} subsets "
      f"(subsets loaded from existing artifacts above cost ~0 of that time).")


# %% [markdown]
# ## 11. Cross-subset summary
#
# Combines metrics across all subsets and reports two deltas per subset (positive = improvement):
# - `proposed_vs_best_single`: proposed framework vs. the best individual model
# - `vlm_evidence_contribution`: constrained fusion with vs. without VLM evidence
#
# Combined ablation and reproducibility tables are written to `/content/artifacts/summary/`.


# %%
banner("[TITLE] CROSS-SUBSET SUMMARY", char="#")

summary_rows = []
for subset, res in all_results.items():
    mt = res["metrics_table"]
    for method in mt.index:
        row = mt.loc[method]
        summary_rows.append({"subset": subset, "method": method, "RMSE": float(row["RMSE"]),
                              "MAE": float(row["MAE"]), "NASA_score": float(row["NASA_score"]),
                              "Pearson_r": float(row["Pearson_r"])})
summary_all_df = pd.DataFrame(summary_rows)
summary_dir = os.path.join(ARTIFACT_ROOT, "summary")
os.makedirs(summary_dir, exist_ok=True)
summary_all_df.to_csv(os.path.join(summary_dir, "metrics_all_subsets.csv"), index=False)

print("[TITLE] RMSE by method x subset (full combined table)")
display(summary_all_df.pivot(index="method", columns="subset", values="RMSE").round(2))

fig, ax = plt.subplots(figsize=(12, 6))
key_methods = ML_DL_MODEL_LIST + ["SimpleAvg_Ensemble", "WeightedAvg_Ensemble",
                                   "ConstrainedFusion (numeric only)",
                                   "ConstrainedFusion+VLM (proposed)"]
pivot_rmse = summary_all_df.pivot(index="method", columns="subset", values="RMSE")
pivot_rmse = pivot_rmse.reindex([m for m in key_methods if m in pivot_rmse.index])
pivot_rmse.plot(kind="bar", ax=ax)
ax.set_ylabel("RMSE (cycles)")
ax.set_title("RMSE by method across all four C-MAPSS subsets (lower = better)")
ax.legend(title="Subset")
plt.xticks(rotation=45, ha="right")
fig.tight_layout()
plt.show()

PROPOSED_METHOD_NAME = "ConstrainedFusion+VLM (proposed)"
NUMERIC_ONLY_NAME = "ConstrainedFusion (numeric only)"

edge_rows = []
for subset, res in all_results.items():
    mt = res["metrics_table"]
    candidates = [m for m in mt.index if m in ML_DL_MODEL_LIST]
    best_single = mt.loc[candidates, "RMSE"].idxmin()
    proposed_rmse = float(mt.loc[PROPOSED_METHOD_NAME, "RMSE"])
    numeric_only_rmse = float(mt.loc[NUMERIC_ONLY_NAME, "RMSE"]) if NUMERIC_ONLY_NAME in mt.index else float("nan")
    edge_rows.append({
        "subset": subset, "best_single_model": best_single,
        "best_single_RMSE": float(mt.loc[best_single, "RMSE"]),
        "constrained_numeric_only_RMSE": numeric_only_rmse,
        "proposed_RMSE": proposed_rmse,
        # positive = proposed method beats that reference on this subset
        "proposed_vs_best_single": float(mt.loc[best_single, "RMSE"]) - proposed_rmse,
        "vlm_evidence_contribution": numeric_only_rmse - proposed_rmse,
    })
edge_df = pd.DataFrame(edge_rows)
print("\n[TITLE] Cross-subset robustness: does the proposed framework's edge hold up?")
print("(positive 'proposed_vs_best_single' = proposed beats the best individual model on that")
print(" subset; positive 'vlm_evidence_contribution' = adding VLM evidence to the deterministic")
print(" fusion helped. Negative values are reported as-is, not reframed.)")
display(edge_df.round(3))
edge_df.to_csv(os.path.join(summary_dir, "proposed_method_edge_by_subset.csv"), index=False)

# ---- combined ablation across subsets ----
_abl_frames = []
for subset, res in all_results.items():
    at = res.get("ablation_table")
    if at is not None and len(at):
        _t = at.copy()
        _t["subset"] = subset
        _abl_frames.append(_t.reset_index())
if _abl_frames:
    ablation_all_df = pd.concat(_abl_frames, ignore_index=True)
    ablation_all_df.to_csv(os.path.join(summary_dir, "ablation_all_subsets.csv"), index=False)
    print("\n[TITLE] Ablation RMSE by stage x subset (combined)")
    display(ablation_all_df.pivot(index="Stage", columns="subset", values="RMSE").round(3))

# ---- reproducibility roll-up ----
_repro_rows = []
for subset, res in all_results.items():
    rep = res.get("repro_report") or {}
    for mname, d in (rep.get("dl_model_determinism") or {}).items():
        _repro_rows.append({"subset": subset, "model": mname,
                             "max_abs_diff": d.get("max_abs_diff"),
                             "mean_abs_diff": d.get("mean_abs_diff")})
if _repro_rows:
    repro_all_df = pd.DataFrame(_repro_rows)
    repro_all_df.to_csv(os.path.join(summary_dir, "reproducibility_all_subsets.csv"), index=False)
    _worst = float(repro_all_df["max_abs_diff"].max())
    print(f"\n[TITLE] Reproducibility: largest re-run prediction difference across all subsets/models "
          f"= {_worst:.8f}")
    print("  (0.0 means repeated inference on the same trained model is bit-identical, as expected "
          "with eval() mode + fixed seeds + deterministic cuDNN.)")

with open(os.path.join(summary_dir, "config.json"), "w") as f:
    json.dump(CONFIG_SNAPSHOT, f, indent=2)

print(f"\n[TITLE] Artifact locations")
print(f"Cross-subset summary saved under {summary_dir}/")
print(f"Per-subset artifacts (predictions, VLM/LLM JSON, graphs, metrics, plots, run_summary.json) "
      f"saved under {ARTIFACT_ROOT}/<SUBSET>/ for SUBSET in {SUBSETS}")


# %% [markdown]
# ## 12. Limitations
#
# - **Terminal-phase evaluation.** C-MAPSS has no ground-truth degradation-onset label. The VLM's
#   `terminal_phase_start_cycle` is compared against an RUL ≤ 30 proxy; both are operational
#   definitions, and the signed difference is reported alongside the absolute difference.
# - **Confidence score.** The score is a fixed heuristic in [0, 1] and is not calibrated. Section 9.3
#   reports how well it tracks empirical accuracy.
# - **RUL capping.** `RUL_CAP` is applied to both training targets and test references, following
#   standard C-MAPSS practice. Predictions are not clipped.
# - **Statistical power.** Test-set sizes differ across subsets (100, 259, 100, and 248 engines).
#   Small RMSE differences and single p-values should be interpreted with caution.
# - **MS-TCN.** The architecture is adapted from action segmentation to sequence-to-one regression
#   with a single MSE loss.
# - **Explanations.** The case reports and LLM recommendations have not been validated against
#   expert judgment.
# - **Reproducibility.** Fixed seeds, greedy decoding, and deterministic cuDNN ensure run-to-run
#   determinism on the same hardware and library versions. Results are specific to the 4-bit
#   quantization configuration.
#
# ## References
#
# - Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation modeling for aircraft
#   engine run-to-failure simulation. *PHM 2008*.
# - Heimes, F. O. (2008). Recurrent neural networks for remaining useful life estimation. *PHM 2008*.
# - Li, X., Ding, Q., & Sun, J.-Q. (2018). Remaining useful life estimation in prognostics using deep
#   convolution neural networks. *Reliability Engineering & System Safety*.
# - Zheng, S., Ristovski, K., Farahat, A., & Gupta, C. (2017). Long short-term memory network for
#   remaining useful life estimation. *IEEE ICPHM*.
# - Farha, Y. A., & Gall, J. (2019). MS-TCN: Multi-stage temporal convolutional network for action
#   segmentation. *CVPR*.
# - Bai, S. et al. (2025). Qwen2.5-VL technical report. arXiv:2502.13923.
# - Chen, Z. et al. (2025). Harnessing multiple large language models: A survey on LLM ensemble.
#   arXiv:2502.18036.
# - Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural
#   networks. *ICML*.
