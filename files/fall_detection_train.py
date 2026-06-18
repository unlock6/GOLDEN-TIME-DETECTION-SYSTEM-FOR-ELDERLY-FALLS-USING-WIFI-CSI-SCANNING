"""
CSI 낙상 감지 이진 분류 학습 및 통계 분석

실제 수집 데이터:
  - fall      → 라벨 1 (낙상/위험)
  - lie_down  → 라벨 1 (낙상/위험)
합성 생성 데이터:
  - static    → 라벨 0 (정상)
  - stand     → 라벨 0 (정상)

실행: python fall_detection_train.py
"""

from __future__ import annotations

import ast
import os
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경에서도 저장 가능
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
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"C:\Users\주현준\OneDrive\Desktop\새 폴더")
OUTPUT_DIR = Path(r"C:\Users\주현준\OneDrive\Desktop\files\artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 신호 파라미터 ──────────────────────────────────────────────────────────────
WINDOW_SAMPLES = 100   # 윈도우당 프레임 수
WINDOW_STRIDE  = 20    # 슬라이딩 보폭
N_SUBCARRIERS  = 64    # 128 I/Q → 64 서브캐리어 진폭
RANDOM_SEED    = 42

_CSI_PATTERN = re.compile(r"\[([0-9,\s\-]+)\]")


# ══════════════════════════════════════════════════════════════════════════════
# 1. 데이터 로딩
# ══════════════════════════════════════════════════════════════════════════════

def extract_csi_amplitude(raw_data: str) -> np.ndarray | None:
    """raw_data 문자열 → CSI I/Q 쌍 파싱 → 진폭 배열 반환."""
    if not isinstance(raw_data, str):
        return None
    match = _CSI_PATTERN.search(raw_data)
    if not match:
        return None
    try:
        vals = ast.literal_eval("[" + match.group(1) + "]")
        arr  = np.array(vals, dtype=np.float32)
        if len(arr) % 2 == 0:
            I = arr[0::2]
            Q = arr[1::2]
            return np.sqrt(I**2 + Q**2)
        return arr
    except Exception:
        return None


def _file_to_windows(file_path: Path) -> np.ndarray:
    """CSV 파일 → (n_windows, WINDOW_SAMPLES, N_SUBCARRIERS) ndarray."""
    df = pd.read_csv(file_path, encoding="cp949")
    if "raw_data" not in df.columns:
        return np.empty((0, WINDOW_SAMPLES, N_SUBCARRIERS), dtype=np.float32)

    frames = []
    for raw in df["raw_data"]:
        amp = extract_csi_amplitude(str(raw))
        if amp is not None and len(amp) >= N_SUBCARRIERS:
            frames.append(amp[:N_SUBCARRIERS])

    if len(frames) < WINDOW_SAMPLES:
        return np.empty((0, WINDOW_SAMPLES, N_SUBCARRIERS), dtype=np.float32)

    signal = np.array(frames, dtype=np.float32)
    windows = []
    for start in range(0, len(signal) - WINDOW_SAMPLES + 1, WINDOW_STRIDE):
        windows.append(signal[start : start + WINDOW_SAMPLES])

    return np.array(windows, dtype=np.float32) if windows else np.empty((0, WINDOW_SAMPLES, N_SUBCARRIERS), dtype=np.float32)


def load_real_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    새 폴더의 fall / lie_down CSV 파일을 로드한다.
    RX1 파일만 사용 (RX2는 중복 안테나 — 앙상블 용도로 별도 활용 가능).
    """
    class_patterns = {
        "fall":      (1, "rx1", "fall"),
        "lie_down":  (1, "rx1", "lie_down"),
    }

    all_X, all_y = [], []
    stats: dict[str, int] = {}

    for cls_name, (label, rx, keyword) in class_patterns.items():
        files = sorted(data_dir.glob(f"*{rx}*{keyword}*.csv"))
        cls_windows = []
        for f in files:
            w = _file_to_windows(f)
            if w.shape[0] > 0:
                cls_windows.append(w)

        if not cls_windows:
            print(f"  [경고] '{cls_name}' 파일 없음: {data_dir}")
            stats[cls_name] = 0
            continue

        X_cls = np.concatenate(cls_windows, axis=0)
        y_cls = np.full(len(X_cls), label, dtype=np.int64)
        all_X.append(X_cls)
        all_y.append(y_cls)
        stats[cls_name] = len(X_cls)
        print(f"  [{cls_name}] 파일 {len(files)}개, 윈도우 {len(X_cls):,}개 (라벨={label})")

    if not all_X:
        raise RuntimeError("실제 데이터가 없습니다.")

    return np.concatenate(all_X), np.concatenate(all_y), stats


# ══════════════════════════════════════════════════════════════════════════════
# 2. 합성 데이터 생성 (static / stand)
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic(n_windows: int, state: str, rng: np.random.Generator) -> np.ndarray:
    """
    state별 합성 CSI 진폭 윈도우를 생성한다.

    static : 거의 움직임 없음 — 낮은 진폭, 아주 작은 노이즈
    stand  : 미세 체중 이동 + 호흡 — 중간 진폭, 완만한 주기 변화
    """
    T, N = WINDOW_SAMPLES, N_SUBCARRIERS
    t = np.linspace(0, 1, T, dtype=np.float32)

    samples = []
    for _ in range(n_windows):
        carrier_base = rng.uniform(5, 15, size=N).astype(np.float32)

        if state == "static":
            noise = rng.normal(0, 0.3, size=(T, N)).astype(np.float32)
            window = carrier_base[None, :] + noise

        elif state == "stand":
            # 호흡 (~0.3 Hz) + 미세 체중 이동 (~0.1 Hz)
            breath = 2.5 * np.sin(2 * np.pi * 0.3 * t)[:, None]
            sway   = 1.2 * np.sin(2 * np.pi * 0.1 * t + rng.uniform(0, 2*np.pi))[:, None]
            noise  = rng.normal(0, 0.8, size=(T, N)).astype(np.float32)
            carrier_base = rng.uniform(10, 25, size=N).astype(np.float32)
            window = carrier_base[None, :] + breath + sway + noise

        else:
            raise ValueError(f"Unknown state: {state}")

        samples.append(np.clip(window, 0, None).astype(np.float32))

    return np.array(samples, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 특징 추출
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(X: np.ndarray) -> np.ndarray:
    """
    (N, T, S) 윈도우 배열 → (N, F) 특징 행렬.

    추출 특징 (서브캐리어별 → 평균으로 집약):
      - 시간축 평균 / 표준편차 / 최대값
      - 1차 차분 절댓값의 평균 / 최대값   (동적 변화량)
      - 2차 차분 절댓값의 평균              (가속도)
      - 첨도(kurtosis) 평균                 (충격성 이벤트)
      - 에너지 (제곱합)
    """
    feats = []
    for window in X:  # (T, S)
        mean_t = np.mean(window, axis=0)
        std_t  = np.std(window,  axis=0)
        max_t  = np.max(window,  axis=0)

        d1     = np.diff(window, n=1, axis=0)
        d2     = np.diff(window, n=2, axis=0)

        mean_d1 = np.mean(np.abs(d1), axis=0)
        max_d1  = np.max(np.abs(d1),  axis=0)
        mean_d2 = np.mean(np.abs(d2), axis=0)

        # 첨도
        mu   = mean_t
        sig  = std_t + 1e-8
        kurt = np.mean(((window - mu) / sig) ** 4, axis=0)

        energy = np.sum(window ** 2, axis=0)

        # 채널 간 상관 (전체 에너지의 변동성)
        chan_energy = np.sum(window ** 2, axis=1)   # (T,)
        energy_std  = np.std(chan_energy)
        energy_max  = np.max(chan_energy)

        feat = np.concatenate([
            mean_t, std_t, max_t,
            mean_d1, max_d1, mean_d2,
            kurt, energy,
            [energy_std, energy_max],
        ])
        feats.append(feat)

    return np.array(feats, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 4. 통계 및 시각화
# ══════════════════════════════════════════════════════════════════════════════

def print_dataset_stats(y: np.ndarray, stats_real: dict, n_synth: dict):
    """데이터셋 구성 통계를 출력한다."""
    print("\n" + "=" * 60)
    print("  데이터셋 통계")
    print("=" * 60)
    print(f"{'클래스':<20} {'상태':<12} {'출처':<8} {'윈도우 수':>10}")
    print("-" * 60)

    rows = [
        ("static",    "라벨 0", "합성", n_synth.get("static", 0)),
        ("stand",     "라벨 0", "합성", n_synth.get("stand",  0)),
        ("lie_down",  "라벨 1", "실제", stats_real.get("lie_down", 0)),
        ("fall",      "라벨 1", "실제", stats_real.get("fall",     0)),
    ]
    for name, label, src, cnt in rows:
        print(f"  {name:<18} {label:<12} {src:<8} {cnt:>10,}")
    print("-" * 60)
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    total = len(y)
    print(f"  {'라벨 0 합계':<18} {'':12} {'':8} {n0:>10,}  ({n0/total*100:.1f}%)")
    print(f"  {'라벨 1 합계':<18} {'':12} {'':8} {n1:>10,}  ({n1/total*100:.1f}%)")
    print(f"  {'전체':<18} {'':12} {'':8} {total:>10,}")
    print("=" * 60)


def plot_class_distribution(y: np.ndarray, save_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = [int((y == 0).sum()), int((y == 1).sum())]
    bars = ax.bar(["라벨 0\n(static+stand)", "라벨 1\n(lie_down+fall)"],
                  counts, color=["steelblue", "tomato"], width=0.5)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{cnt:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("클래스 분포", fontsize=13, fontweight="bold")
    ax.set_ylabel("윈도우 수")
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  [저장] {save_path}")


def plot_feature_comparison(X_feat: np.ndarray, y: np.ndarray, save_path: Path):
    """라벨별 대표 특징 분포 비교 (박스플롯)."""
    # 상위 8개 특징만 표시 (mean_t 일부)
    n_show = 8
    feature_names = [f"mean_sub{i}" for i in range(n_show)]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.flatten()
    for i in range(n_show):
        ax = axes[i]
        data0 = X_feat[y == 0, i]
        data1 = X_feat[y == 1, i]
        ax.boxplot([data0, data1], labels=["라벨0", "라벨1"],
                   patch_artist=True,
                   boxprops=dict(facecolor="steelblue", alpha=0.6))
        parts = ax.boxplot([data0, data1], labels=["라벨0", "라벨1"],
                           patch_artist=True,
                           boxprops=dict(facecolor="steelblue", alpha=0.6))
        parts["boxes"][1].set_facecolor("tomato")
        ax.set_title(feature_names[i], fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("서브캐리어 평균 진폭 분포 (라벨별)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  [저장] {save_path}")


def plot_confusion_matrix_fig(cm: np.ndarray, title: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["정상(0)", "위험(1)"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  [저장] {save_path}")


def plot_roc_curves(results: list[dict], save_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in results:
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
        ax.plot(fpr, tpr, label=f"{r['name']} (AUC={r['auc']:.3f})", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR (False Positive Rate)")
    ax.set_ylabel("TPR (True Positive Rate)")
    ax.set_title("ROC 곡선 비교", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  [저장] {save_path}")


def plot_signal_examples(X_real: np.ndarray, y_real: np.ndarray,
                         X_synth: np.ndarray, y_synth: np.ndarray,
                         save_path: Path):
    """실제/합성 데이터 신호 예시 시각화."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    samples = {
        (0, 0): ("fall (실제)",     X_real[y_real == 1][0]),
        (0, 1): ("lie_down (실제)", X_real[y_real == 1][-1]),
        (1, 0): ("static (합성)",   X_synth[y_synth == 0][0]),
        (1, 1): ("stand (합성)",    X_synth[y_synth == 0][-1]),
    }

    for (r, c), (title, window) in samples.items():
        ax = axes[r][c]
        # 첫 10개 서브캐리어만 표시
        for ch in range(min(10, window.shape[1])):
            ax.plot(window[:, ch], alpha=0.5, lw=0.8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("프레임")
        ax.set_ylabel("CSI 진폭")
        ax.grid(alpha=0.3)

    plt.suptitle("CSI 신호 예시 (서브캐리어 0~9)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  [저장] {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. 모델 학습 및 평가
# ══════════════════════════════════════════════════════════════════════════════

def build_models(seed: int) -> dict:
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=seed)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=None,
                                            class_weight="balanced_subsample",
                                            random_state=seed, n_jobs=-1)),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                                learning_rate=0.05,
                                                random_state=seed)),
        ]),
    }


def evaluate_models(X_feat: np.ndarray, y: np.ndarray, seed: int) -> list[dict]:
    """홀드아웃 + 5-fold CV로 모든 모델을 평가한다."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_feat, y, test_size=0.2, stratify=y, random_state=seed
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    all_results = []
    print("\n" + "=" * 60)
    print("  모델 학습 및 평가")
    print("=" * 60)

    models = build_models(seed)
    for name, pipe in models.items():
        print(f"\n[{name}]")

        # 홀드아웃
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        y_prob = pipe.predict_proba(X_te)[:, 1]

        acc  = accuracy_score(y_te, y_pred)
        f1   = f1_score(y_te, y_pred, average="weighted")
        auc  = roc_auc_score(y_te, y_prob)
        cm   = confusion_matrix(y_te, y_pred)

        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1-score : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
        print("  Classification Report:")
        print(classification_report(y_te, y_pred,
                                    target_names=["정상(0)", "위험(1)"],
                                    digits=4))

        # 5-fold CV
        cv_accs, cv_f1s, cv_aucs = [], [], []
        for tr_idx, te_idx in skf.split(X_feat, y):
            p = build_models(seed)[name]
            p.fit(X_feat[tr_idx], y[tr_idx])
            yp = p.predict(X_feat[te_idx])
            yprob = p.predict_proba(X_feat[te_idx])[:, 1]
            cv_accs.append(accuracy_score(y[te_idx], yp))
            cv_f1s.append(f1_score(y[te_idx], yp, average="weighted"))
            cv_aucs.append(roc_auc_score(y[te_idx], yprob))

        print(f"  5-fold CV  Acc : {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}")
        print(f"  5-fold CV  F1  : {np.mean(cv_f1s):.4f} ± {np.std(cv_f1s):.4f}")
        print(f"  5-fold CV  AUC : {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")

        all_results.append({
            "name":   name,
            "acc":    acc,
            "f1":     f1,
            "auc":    auc,
            "cm":     cm,
            "y_true": y_te,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "cv_acc": (np.mean(cv_accs), np.std(cv_accs)),
            "cv_f1":  (np.mean(cv_f1s),  np.std(cv_f1s)),
            "cv_auc": (np.mean(cv_aucs), np.std(cv_aucs)),
        })

    return all_results


def plot_model_comparison(results: list[dict], save_path: Path):
    """모델별 성능 지표 막대그래프."""
    names = [r["name"] for r in results]
    accs  = [r["acc"]  for r in results]
    f1s   = [r["f1"]   for r in results]
    aucs  = [r["auc"]  for r in results]

    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w, accs, w, label="Accuracy",  color="steelblue")
    b2 = ax.bar(x,     f1s,  w, label="F1-score",  color="seagreen")
    b3 = ax.bar(x + w, aucs, w, label="ROC-AUC",   color="coral")

    for bars in (b1, b2, b3):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("모델별 성능 비교 (홀드아웃)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  [저장] {save_path}")


def print_summary_table(results: list[dict]):
    """최종 요약 표 출력."""
    print("\n" + "=" * 70)
    print("  최종 요약 (홀드아웃 20% 테스트)")
    print("=" * 70)
    fmt = "  {:<22} {:>8} {:>8} {:>8} {:>10} {:>10}"
    print(fmt.format("모델", "Acc", "F1", "AUC",
                      "CV Acc", "CV AUC"))
    print("-" * 70)
    for r in results:
        print(fmt.format(
            r["name"],
            f"{r['acc']:.4f}",
            f"{r['f1']:.4f}",
            f"{r['auc']:.4f}",
            f"{r['cv_acc'][0]:.4f}±{r['cv_acc'][1]:.4f}",
            f"{r['cv_auc'][0]:.4f}±{r['cv_auc'][1]:.4f}",
        ))
    print("=" * 70)

    best = max(results, key=lambda r: r["auc"])
    print(f"\n  최고 성능 모델: {best['name']}  (AUC={best['auc']:.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# 6. 메인
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  CSI 기반 낙상 감지 이진 분류 학습 시작")
    print("=" * 60)

    rng = np.random.default_rng(RANDOM_SEED)

    # ── 실제 데이터 로드 ─────────────────────────────────────────────────────
    print("\n[1단계] 실제 데이터 로드")
    X_real, y_real, stats_real = load_real_data(DATA_DIR)
    print(f"  실제 데이터 합계: {len(X_real):,}개 윈도우")

    # ── 합성 데이터 생성 (static + stand) ────────────────────────────────────
    print("\n[2단계] 합성 데이터 생성 (static, stand)")
    n_real_per_class = len(X_real) // 2   # fall/lie_down 각 절반 근사

    n_static = max(500, n_real_per_class // 2)
    n_stand  = max(500, n_real_per_class // 2)

    X_static = generate_synthetic(n_static, "static", rng)
    X_stand  = generate_synthetic(n_stand,  "stand",  rng)

    y_static = np.zeros(n_static, dtype=np.int64)
    y_stand  = np.zeros(n_stand,  dtype=np.int64)

    n_synth = {"static": n_static, "stand": n_stand}
    print(f"  static : {n_static:,}개")
    print(f"  stand  : {n_stand:,}개")

    X_synth = np.concatenate([X_static, X_stand], axis=0)
    y_synth = np.concatenate([y_static, y_stand], axis=0)

    # ── 전체 데이터 병합 ─────────────────────────────────────────────────────
    print("\n[3단계] 데이터 병합 및 라벨 확인")
    X_all = np.concatenate([X_real, X_synth], axis=0)
    y_all = np.concatenate([y_real, y_synth], axis=0)

    print_dataset_stats(y_all, stats_real, n_synth)

    # ── 특징 추출 ─────────────────────────────────────────────────────────────
    print("\n[4단계] 특징 추출")
    print("  (각 윈도우에서 평균/표준편차/최대값/차분/에너지 등 추출)")
    X_feat = extract_features(X_all)
    print(f"  특징 행렬: {X_feat.shape}  (샘플 × 특징)")

    # ── 시각화 ───────────────────────────────────────────────────────────────
    print("\n[5단계] 시각화 생성")
    plot_class_distribution(
        y_all, OUTPUT_DIR / "01_class_distribution.png"
    )
    plot_feature_comparison(
        X_feat, y_all, OUTPUT_DIR / "02_feature_comparison.png"
    )
    plot_signal_examples(
        X_real, y_real, X_synth, y_synth,
        OUTPUT_DIR / "03_signal_examples.png"
    )

    # ── 모델 학습 및 평가 ────────────────────────────────────────────────────
    print("\n[6단계] 모델 학습 및 평가")
    results = evaluate_models(X_feat, y_all, RANDOM_SEED)

    plot_model_comparison(results, OUTPUT_DIR / "04_model_comparison.png")
    plot_roc_curves(results,       OUTPUT_DIR / "05_roc_curves.png")

    for r in results:
        plot_confusion_matrix_fig(
            r["cm"],
            f"혼동 행렬 — {r['name']}",
            OUTPUT_DIR / f"06_confusion_{r['name'].replace(' ','_')}.png",
        )

    # ── 최종 요약 ─────────────────────────────────────────────────────────────
    print_summary_table(results)

    print(f"\n[완료] 모든 결과물이 {OUTPUT_DIR} 에 저장되었습니다.")
    print("=" * 60)


if __name__ == "__main__":
    main()
