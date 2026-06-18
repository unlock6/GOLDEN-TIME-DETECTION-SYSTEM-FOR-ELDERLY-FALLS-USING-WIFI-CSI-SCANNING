"""
4-클래스 실제 데이터 전체 학습 + 합성 테스트 데이터 예측

실제 데이터 (새 폴더):
  - fall     → 라벨 1  (위험)
  - lie_down → 라벨 1  (위험)
  - static   → 라벨 0  (정상)
  - stand    → 라벨 0  (정상)

실행: python full_analysis.py
"""

from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── 경로 ──────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"C:\Users\주현준\OneDrive\Desktop\새 폴더")
OUTPUT_DIR = Path(r"C:\Users\주현준\OneDrive\Desktop\files\artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 신호 파라미터 ──────────────────────────────────────────────────────────────
WINDOW_SAMPLES = 100
WINDOW_STRIDE  = 20
N_SUB          = 64      # 128 I/Q → 64 진폭
RANDOM_SEED    = 42
N_SYNTH_TEST   = 30      # 상태별 합성 테스트 샘플 수

DANGER = {"fall", "lie_down"}
LABEL_MAP = {"static": 0, "stand": 0, "fall": 1, "lie_down": 1}
CLASS_NAMES = ["정상(static+stand)", "위험(fall+lie_down)"]

_PAT = re.compile(r"\[([0-9,\s\-]+)\]")


# ══════════════════════════════════════════════════════════════════════════════
# 1. 데이터 로딩
# ══════════════════════════════════════════════════════════════════════════════

def _amp(raw: str) -> np.ndarray | None:
    m = _PAT.search(raw)
    if not m:
        return None
    try:
        v = np.array(ast.literal_eval("[" + m.group(1) + "]"), dtype=np.float32)
        return np.sqrt(v[0::2]**2 + v[1::2]**2)[:N_SUB] if len(v) % 2 == 0 else v[:N_SUB]
    except Exception:
        return None


def load_csv_windows(path: Path) -> tuple[np.ndarray, str]:
    df = pd.read_csv(path, encoding="cp949")
    cls = str(df["label"].iloc[0]).strip().lower()
    frames = [a for raw in df["raw_data"] if (a := _amp(str(raw))) is not None]
    if len(frames) < WINDOW_SAMPLES:
        return np.empty((0, WINDOW_SAMPLES, N_SUB), np.float32), cls
    sig = np.array(frames, np.float32)
    wins = [sig[s:s+WINDOW_SAMPLES] for s in range(0, len(sig)-WINDOW_SAMPLES+1, WINDOW_STRIDE)]
    return (np.array(wins, np.float32) if wins else np.empty((0, WINDOW_SAMPLES, N_SUB), np.float32)), cls


def load_all_data() -> tuple[np.ndarray, np.ndarray, dict]:
    """RX1 파일만 사용해 실제 데이터를 로드한다."""
    files = sorted(f for f in DATA_DIR.glob("*.csv") if "rx1" in f.name.lower())
    if not files:
        raise RuntimeError(f"CSV 파일 없음: {DATA_DIR}")

    Xs, ys, stats = [], [], {}
    for f in files:
        W, cls = load_csv_windows(f)
        if W.shape[0] == 0 or cls not in LABEL_MAP:
            continue
        lbl = LABEL_MAP[cls]
        Xs.append(W); ys.append(np.full(len(W), lbl, np.int64))
        stats[cls] = stats.get(cls, 0) + len(W)

    if not Xs:
        raise RuntimeError("유효한 데이터 없음")
    return np.concatenate(Xs), np.concatenate(ys), stats


# ══════════════════════════════════════════════════════════════════════════════
# 2. 특징 추출
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(X: np.ndarray) -> np.ndarray:
    feats = []
    for w in X:                       # w: (T, S)
        d1 = np.diff(w, n=1, axis=0)
        d2 = np.diff(w, n=2, axis=0)
        mu  = w.mean(0); sig = w.std(0) + 1e-8
        kurt = ((((w - mu)/sig)**4).mean(0))
        e_t  = (w**2).sum(1)          # 시간별 에너지
        feat = np.concatenate([
            mu, sig, w.max(0),                     # 진폭 통계
            np.abs(d1).mean(0), np.abs(d1).max(0), # 1차 차분
            np.abs(d2).mean(0),                    # 2차 차분
            kurt,                                  # 첨도
            (w**2).sum(0),                         # 서브캐리어별 에너지
            [e_t.std(), e_t.max(), e_t.mean()],    # 시간 에너지 통계
        ])
        feats.append(feat)
    return np.array(feats, np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 합성 테스트 데이터 생성
# ══════════════════════════════════════════════════════════════════════════════

def make_synthetic_test(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    4가지 상태의 현실적인 합성 CSI 윈도우를 생성한다.

    static   : 낮은 진폭, 미세 노이즈
    stand    : 중간 진폭, 호흡 주기
    lie_down : 중간 진폭 + 바닥 다중경로 → 전체적으로 높은 에너지, 완만한 변화
    fall     : 높은 초기 진폭 급변 → 짧은 transient 후 안정화
    """
    T, S = WINDOW_SAMPLES, N_SUB
    t = np.linspace(0, 1, T, dtype=np.float32)
    states, true_labels, state_tags = [], [], []

    for _ in range(n):
        base = rng.uniform(5, 15, S).astype(np.float32)

        # static
        w = base + rng.normal(0, 0.4, (T, S)).astype(np.float32)
        states.append(np.clip(w, 0, None)); true_labels.append(0); state_tags.append("static")

        # stand
        breath = (3.0 * np.sin(2*np.pi*0.3*t))[:, None]
        w = rng.uniform(10, 28, S) + breath + rng.normal(0, 1.0, (T, S)).astype(np.float32)
        states.append(np.clip(w, 0, None)); true_labels.append(0); state_tags.append("stand")

        # lie_down
        base_ld = rng.uniform(15, 35, S).astype(np.float32)
        slow = (4.0 * np.sin(2*np.pi*0.15*t))[:, None]
        w = base_ld + slow + rng.normal(0, 1.5, (T, S)).astype(np.float32)
        states.append(np.clip(w, 0, None)); true_labels.append(1); state_tags.append("lie_down")

        # fall: 급격한 진폭 spike → 감쇠
        peak_t = rng.integers(10, 30)
        envelope = np.zeros(T, np.float32)
        for tau in range(T):
            if tau < peak_t:
                envelope[tau] = tau / peak_t
            else:
                envelope[tau] = np.exp(-(tau - peak_t) / 15.0)
        base_f = rng.uniform(20, 50, S).astype(np.float32)
        w = base_f * envelope[:, None] + rng.normal(0, 2.0, (T, S)).astype(np.float32)
        states.append(np.clip(w, 0, None)); true_labels.append(1); state_tags.append("fall")

    X = np.array(states, np.float32)
    y = np.array(true_labels, np.int64)
    return X, y, state_tags


# ══════════════════════════════════════════════════════════════════════════════
# 4. 학습 / 평가
# ══════════════════════════════════════════════════════════════════════════════

def build_pipes(seed):
    return {
        "LogisticRegression": Pipeline([("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))]),
        "RandomForest":       Pipeline([("sc", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                           random_state=seed, n_jobs=-1))]),
        "GradientBoosting":   Pipeline([("sc", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                               learning_rate=0.05, random_state=seed))]),
    }


def train_and_eval(X_feat, y, seed):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_feat, y, test_size=0.2, stratify=y, random_state=seed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    results = []

    print("\n" + "="*62)
    print("  모델 학습 및 평가 (실제 데이터)")
    print("="*62)

    for name, pipe in build_pipes(seed).items():
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        y_prob = pipe.predict_proba(X_te)[:, 1]

        acc = accuracy_score(y_te, y_pred)
        f1  = f1_score(y_te, y_pred, average="weighted")
        auc = roc_auc_score(y_te, y_prob)
        cm  = confusion_matrix(y_te, y_pred)

        # 5-fold CV
        cv_a, cv_f, cv_u = [], [], []
        for tr, te in skf.split(X_feat, y):
            p = build_pipes(seed)[name]
            p.fit(X_feat[tr], y[tr])
            yp = p.predict(X_feat[te])
            yb = p.predict_proba(X_feat[te])[:, 1]
            cv_a.append(accuracy_score(y[te], yp))
            cv_f.append(f1_score(y[te], yp, average="weighted"))
            cv_u.append(roc_auc_score(y[te], yb))

        print(f"\n[{name}]")
        print(f"  Accuracy : {acc:.4f}   F1: {f1:.4f}   AUC: {auc:.4f}")
        print(f"  5-fold CV Acc: {np.mean(cv_a):.4f} +/- {np.std(cv_a):.4f}")
        print(f"  5-fold CV AUC: {np.mean(cv_u):.4f} +/- {np.std(cv_u):.4f}")
        print("  Classification Report:")
        print(classification_report(y_te, y_pred,
              target_names=CLASS_NAMES, digits=4))
        print("  Confusion Matrix:")
        print(cm)

        results.append(dict(name=name, acc=acc, f1=f1, auc=auc, cm=cm,
                            y_te=y_te, y_pred=y_pred, y_prob=y_prob,
                            pipe=pipe,
                            cv_acc=(np.mean(cv_a), np.std(cv_a)),
                            cv_auc=(np.mean(cv_u), np.std(cv_u))))
    return results, X_te, y_te


# ══════════════════════════════════════════════════════════════════════════════
# 5. 합성 테스트 예측
# ══════════════════════════════════════════════════════════════════════════════

def predict_synthetic(results, X_synth_feat, y_synth, tags):
    best = max(results, key=lambda r: r["auc"])
    pipe = best["pipe"]

    y_pred_synth = pipe.predict(X_synth_feat)
    y_prob_synth = pipe.predict_proba(X_synth_feat)[:, 1]

    acc_synth = accuracy_score(y_synth, y_pred_synth)

    print("\n" + "="*62)
    print(f"  합성 테스트 데이터 예측 (최고 모델: {best['name']})")
    print("="*62)
    print(f"  총 {len(y_synth)}개 샘플 | 정확도: {acc_synth:.4f}")
    print()
    print(f"  {'#':>3}  {'실제 상태':<12} {'예측 라벨':<12} {'위험 확률':>8}  {'결과'}")
    print("  " + "-"*52)

    correct_by_state = {}
    total_by_state   = {}
    for i, (tag, yt, yp, yb) in enumerate(zip(tags, y_synth, y_pred_synth, y_prob_synth)):
        pred_name = "위험(1)" if yp == 1 else "정상(0)"
        ok = "O" if yp == yt else "X"
        print(f"  {i+1:>3}  {tag:<12} {pred_name:<12} {yb:>8.4f}  [{ok}]")
        total_by_state[tag] = total_by_state.get(tag, 0) + 1
        if yp == yt:
            correct_by_state[tag] = correct_by_state.get(tag, 0) + 1

    print()
    print("  상태별 정확도:")
    for state in ["static", "stand", "lie_down", "fall"]:
        tot = total_by_state.get(state, 0)
        cor = correct_by_state.get(state, 0)
        bar = "#" * cor + "-" * (tot - cor)
        print(f"    {state:<12}: {cor}/{tot}  [{bar}]  "
              f"({cor/tot*100:.0f}%)" if tot > 0 else f"    {state:<12}: 데이터 없음")

    print(f"\n  전체 합성 테스트 정확도: {acc_synth*100:.2f}%")
    return y_pred_synth, y_prob_synth, acc_synth


# ══════════════════════════════════════════════════════════════════════════════
# 6. 시각화
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, name):
    p = OUTPUT_DIR / name
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [저장] {p.name}")


def plot_class_dist(stats, save=True):
    labels = list(stats.keys())
    counts = list(stats.values())
    colors = ["steelblue" if LABEL_MAP.get(l, 0) == 0 else "tomato" for l in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, counts, color=colors, width=0.5, edgecolor="white")
    for b, c in zip(bars, counts):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+30,
                f"{c:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("클래스별 윈도우 수 (실제 데이터)", fontsize=13, fontweight="bold")
    ax.set_ylabel("윈도우 수"); ax.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="steelblue", label="라벨 0 (정상)"),
                       Patch(color="tomato",    label="라벨 1 (위험)")])
    if save: _save(fig, "A1_class_dist.png")


def plot_signal_per_state(X_all, y_all, stats, save=True):
    state_order = ["static", "stand", "lie_down", "fall"]
    state_label = {"static": 0, "stand": 0, "lie_down": 1, "fall": 1}

    # 각 상태의 첫 번째 윈도우를 찾기 위해 파일별로 다시 로드
    samples = {}
    files = sorted(f for f in DATA_DIR.glob("*.csv") if "rx1" in f.name.lower())
    for f in files:
        df = pd.read_csv(f, encoding="cp949")
        cls = str(df["label"].iloc[0]).strip().lower()
        if cls in state_order and cls not in samples:
            frames = [a for raw in df["raw_data"] if (a := _amp(str(raw))) is not None]
            if len(frames) >= WINDOW_SAMPLES:
                samples[cls] = np.array(frames[:WINDOW_SAMPLES], np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    colors_map = {"static": "steelblue", "stand": "seagreen",
                  "lie_down": "orange", "fall": "tomato"}
    for ax, st in zip(axes.flatten(), state_order):
        if st not in samples:
            ax.set_title(f"{st} (데이터 없음)"); continue
        w = samples[st]
        for ch in range(min(12, w.shape[1])):
            ax.plot(w[:, ch], alpha=0.5, lw=0.8, color=colors_map[st])
        lbl = state_label[st]
        ax.set_title(f"{st}  [라벨 {lbl}]", fontsize=11, fontweight="bold",
                     color=colors_map[st])
        ax.set_xlabel("프레임"); ax.set_ylabel("CSI 진폭"); ax.grid(alpha=0.3)
    plt.suptitle("상태별 실제 CSI 신호 (서브캐리어 0~11)", fontsize=12)
    plt.tight_layout()
    if save: _save(fig, "A2_signals.png")


def plot_model_comparison(results, save=True):
    names = [r["name"] for r in results]
    accs  = [r["acc"]  for r in results]
    f1s   = [r["f1"]   for r in results]
    aucs  = [r["auc"]  for r in results]
    x = np.arange(len(names)); w = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x-w, accs, w, label="Accuracy",  color="steelblue")
    b2 = ax.bar(x,   f1s,  w, label="F1-score",  color="seagreen")
    b3 = ax.bar(x+w, aucs, w, label="ROC-AUC",   color="coral")
    for bars in (b1, b2, b3):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=10)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
    ax.set_title("모델별 성능 비교 (실제 데이터, 홀드아웃 20%)", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    if save: _save(fig, "A3_model_comparison.png")


def plot_confusion(results, save=True):
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
    if len(results) == 1: axes = [axes]
    for ax, r in zip(axes, results):
        ConfusionMatrixDisplay(r["cm"], display_labels=["정상(0)", "위험(1)"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(r["name"], fontsize=10, fontweight="bold")
    plt.suptitle("혼동 행렬 (실제 데이터)", fontsize=12)
    plt.tight_layout()
    if save: _save(fig, "A4_confusion.png")


def plot_synth_result(tags, y_synth, y_pred_synth, y_prob_synth, acc_synth, save=True):
    state_order = ["static", "stand", "lie_down", "fall"]
    colors_state = {"static": "steelblue", "stand": "seagreen",
                    "lie_down": "orange", "fall": "tomato"}

    n = len(y_synth)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 왼쪽: 샘플별 위험 확률 산점도
    ax = axes[0]
    for i, (tag, yt, yb) in enumerate(zip(tags, y_synth, y_prob_synth)):
        ok = yb >= 0.5  # 예측 라벨
        marker = "o" if ok == bool(yt) else "X"
        ax.scatter(i, yb, color=colors_state[tag], marker=marker, s=70, zorder=3)
    ax.axhline(0.5, ls="--", color="gray", lw=1)
    ax.set_xlabel("샘플 번호"); ax.set_ylabel("위험 확률 (라벨 1)")
    ax.set_title(f"합성 테스트 예측 확률  (전체 정확도: {acc_synth*100:.1f}%)",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
    from matplotlib.lines import Line2D
    legend_elems = [Line2D([0],[0], marker="o", color="w",
                           markerfacecolor=colors_state[s], markersize=9, label=s)
                    for s in state_order]
    legend_elems += [Line2D([0],[0], marker="X", color="gray",
                             markersize=9, label="오분류")]
    ax.legend(handles=legend_elems, fontsize=8, loc="upper left")

    # 오른쪽: 상태별 정확도 막대
    ax2 = axes[1]
    state_acc = {}
    for tag, yt, yp in zip(tags, y_synth, y_pred_synth):
        state_acc.setdefault(tag, []).append(int(yt == yp))
    state_means = {s: np.mean(v) for s, v in state_acc.items()}
    bars = ax2.bar(state_order,
                   [state_means.get(s, 0) for s in state_order],
                   color=[colors_state[s] for s in state_order], width=0.5)
    for b, s in zip(bars, state_order):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.02,
                 f"{b.get_height()*100:.0f}%",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1.2); ax2.set_ylabel("정확도")
    ax2.set_title("상태별 분류 정확도 (합성 테스트)", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    ax2.axhline(1.0, ls="--", color="gray", lw=1)

    plt.tight_layout()
    if save: _save(fig, "A5_synth_result.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7. 메인
# ══════════════════════════════════════════════════════════════════════════════

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    # ── 1. 실제 데이터 로드 ────────────────────────────────────────────────
    print("\n" + "="*62)
    print("  1단계: 실제 데이터 로드")
    print("="*62)
    X_all, y_all, stats = load_all_data()

    print("\n  [데이터 요약]")
    print(f"  {'상태':<12} {'라벨':>6} {'윈도우 수':>10}")
    print("  " + "-"*32)
    for cls, cnt in sorted(stats.items()):
        print(f"  {cls:<12} {LABEL_MAP[cls]:>6}      {cnt:>8,}")
    n0 = int((y_all==0).sum()); n1 = int((y_all==1).sum())
    print("  " + "-"*32)
    print(f"  {'라벨 0 합계':<12}        {n0:>8,}  ({n0/len(y_all)*100:.1f}%)")
    print(f"  {'라벨 1 합계':<12}        {n1:>8,}  ({n1/len(y_all)*100:.1f}%)")
    print(f"  {'전체':<12}        {len(y_all):>8,}")

    # ── 2. 특징 추출 ───────────────────────────────────────────────────────
    print("\n  2단계: 특징 추출 중...")
    X_feat = extract_features(X_all)
    print(f"  특징 행렬: {X_feat.shape}")

    # ── 3. 시각화 ─────────────────────────────────────────────────────────
    print("\n  3단계: 시각화 생성")
    plot_class_dist(stats)
    plot_signal_per_state(X_all, y_all, stats)

    # ── 4. 학습 + 평가 ─────────────────────────────────────────────────────
    results, X_te, y_te = train_and_eval(X_feat, y_all, RANDOM_SEED)
    plot_model_comparison(results)
    plot_confusion(results)

    # ── 5. 합성 테스트 데이터 생성 + 예측 ────────────────────────────────
    print("\n  5단계: 합성 테스트 데이터 생성 및 예측")
    X_synth, y_synth, tags = make_synthetic_test(N_SYNTH_TEST, rng)
    X_synth_feat = extract_features(X_synth)

    y_pred_synth, y_prob_synth, acc_synth = predict_synthetic(
        results, X_synth_feat, y_synth, tags)

    plot_synth_result(tags, y_synth, y_pred_synth, y_prob_synth, acc_synth)

    # ── 6. 최종 요약 ──────────────────────────────────────────────────────
    best = max(results, key=lambda r: r["auc"])
    print("\n" + "="*62)
    print("  최종 요약")
    print("="*62)
    print(f"  {'모델':<22} {'Acc':>7} {'F1':>7} {'AUC':>7} {'CV Acc':>12} {'CV AUC':>12}")
    print("  " + "-"*62)
    for r in results:
        print(f"  {r['name']:<22} {r['acc']:>7.4f} {r['f1']:>7.4f} {r['auc']:>7.4f} "
              f"  {r['cv_acc'][0]:.4f}+/-{r['cv_acc'][1]:.4f}  "
              f"{r['cv_auc'][0]:.4f}+/-{r['cv_auc'][1]:.4f}")
    print("="*62)
    print(f"\n  최고 모델: {best['name']}  (AUC={best['auc']:.4f})")
    print(f"  합성 테스트 정확도: {acc_synth*100:.2f}%  ({int(acc_synth*len(y_synth))}/{len(y_synth)} 정답)")
    print(f"\n  저장 경로: {OUTPUT_DIR}")
    print("="*62)


if __name__ == "__main__":
    main()
