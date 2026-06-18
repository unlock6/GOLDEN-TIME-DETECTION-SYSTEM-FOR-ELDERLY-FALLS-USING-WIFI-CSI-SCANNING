"""
CSI 전처리 파이프라인.
논문의 핵심 절차(슬라이딩 윈도우, Hampel, SG, 정규화, PCA)를 반영한다.

[추가] 시간차분(Temporal Differencing) 필터
  - 낙상 동작은 정적 자세 대비 급격한 진폭 변화를 동반한다.
  - 1차(또는 2차) 차분을 적용하면 DC 성분(정적 배경)이 제거되고,
    움직임 관련 동적 변화만 강조되어 CNN-LSTM 학습에 유리하다.
  - DIFF_STACK_MODE = "concat" 이면 원신호와 차분 신호를 채널 방향으로
    이어붙여 두 가지 관점(절대 진폭 + 변화량)을 동시에 활용한다.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from config import (
    DIFF_ORDER,
    DIFF_STACK_MODE,
    DOWNSAMPLE_FACTOR,
    HAMPEL_SIGMA,
    HAMPEL_WINDOW,
    MIN_LENGTH_FOR_DOWNSAMPLE,
    PCA_COMPONENTS,
    PREPROCESS_MODE,
    SG_POLY_ORDER,
    SG_WINDOW,
    WINDOW_SAMPLES,
    WINDOW_STRIDE,
)

try:
    from scipy.signal import savgol_filter
except Exception:
    savgol_filter = None


# ──────────────────────────────────────────────────────────────────────────────
# 슬라이딩 윈도우
# ──────────────────────────────────────────────────────────────────────────────

def sliding_window(sequence: np.ndarray, label: int) -> Tuple[np.ndarray, np.ndarray]:
    """(time, subcarriers) 샘플 하나를 겹치는 윈도우로 분할한다."""
    if sequence.ndim != 2:
        raise ValueError("sequence는 (time, subcarriers) 2차원이어야 합니다.")

    total_time = sequence.shape[0]
    if total_time < WINDOW_SAMPLES:
        pad_len = WINDOW_SAMPLES - total_time
        pad_block = np.repeat(sequence[-1:, :], pad_len, axis=0)
        sequence = np.concatenate([sequence, pad_block], axis=0)
        total_time = sequence.shape[0]

    windows = []
    for start in range(0, total_time - WINDOW_SAMPLES + 1, WINDOW_STRIDE):
        end = start + WINDOW_SAMPLES
        windows.append(sequence[start:end, :])

    window_array = np.array(windows, dtype=np.float32)
    label_array = np.full((len(window_array),), int(label), dtype=np.int64)
    return window_array, label_array


# ──────────────────────────────────────────────────────────────────────────────
# 기존 필터: Hampel / Savitzky-Golay
# ──────────────────────────────────────────────────────────────────────────────

def _hampel_filter_2d(signal_2d: np.ndarray, window_size: int, sigma: float) -> np.ndarray:
    """
    (time, subcarriers) 배열 전체에 Hampel 필터를 벡터화하여 적용한다.

    기존 1D 루프 대비 ~64배 빠름:
    - 기존: 샘플 N × 서브캐리어 64 × 시간 T 반복
    - 개선: 샘플 N × 시간 T 반복 (서브캐리어 축은 numpy 브로드캐스팅)
    """
    if signal_2d.size == 0:
        return signal_2d

    output = signal_2d.copy()
    T = signal_2d.shape[0]
    half = window_size // 2
    k = 1.4826  # Gaussian consistency factor

    for t in range(T):
        left  = max(0, t - half)
        right = min(T, t + half + 1)
        block  = signal_2d[left:right, :]               # (w, S)
        median = np.median(block, axis=0)               # (S,)
        mad    = np.median(np.abs(block - median), axis=0)  # (S,)
        threshold = sigma * k * mad                     # (S,)
        mask = (mad > 0) & (np.abs(signal_2d[t] - median) > threshold)
        output[t, mask] = median[mask]

    return output


def _hampel_filter_1d(signal_1d: np.ndarray, window_size: int, sigma: float) -> np.ndarray:
    """1D 호환용 래퍼 — 내부는 2D 벡터화 함수를 사용한다."""
    return _hampel_filter_2d(signal_1d[:, None], window_size, sigma)[:, 0]


def _smooth_signal(signal_1d: np.ndarray) -> np.ndarray:
    """SG 필터(가능 시) 또는 이동평균(대체)으로 신호를 평활화한다."""
    if savgol_filter is not None and len(signal_1d) >= SG_WINDOW:
        return savgol_filter(
            signal_1d, window_length=SG_WINDOW, polyorder=SG_POLY_ORDER, mode="interp"
        )
    kernel_size = 5
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    padded = np.pad(signal_1d, (kernel_size // 2, kernel_size // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# ──────────────────────────────────────────────────────────────────────────────
# [신규] 시간차분(Temporal Differencing) 필터
# ──────────────────────────────────────────────────────────────────────────────

def temporal_diff_filter(
    X: np.ndarray,
    order: int = DIFF_ORDER,
    stack_mode: str = DIFF_STACK_MODE,
) -> np.ndarray:
    """
    시간 축(axis=1) 방향으로 n차 차분을 적용한다.

    Parameters
    ----------
    X         : (N, time, subcarriers) — 전처리 전 또는 정규화 후 입력
    order     : 차분 차수. 1=속도(Δ), 2=가속도(ΔΔ)
    stack_mode: "replace" → 차분 결과만 반환 (shape 동일, 첫 order 행은 0 패딩)
                "concat"  → 원신호 + 차분 채널 방향 이어붙임
                            (N, time, subcarriers * (order+1))

    Returns
    -------
    X_out : 처리된 배열 (float32)

    동작 원리
    ---------
    CSI 진폭 신호에서 정적 배경(DC 성분)은 연속 프레임 사이에서 거의 변하지
    않는다. 1차 차분 x[t] - x[t-1] 을 계산하면 배경이 상쇄되고, 움직임이
    있는 구간(특히 낙상의 급격한 변화)만 두드러진다.
    "concat" 모드에서는 원신호(절대 진폭 정보)도 함께 유지하므로 모델이
    두 가지 관점을 모두 학습할 수 있다.
    """
    if X.ndim != 3:
        raise ValueError("temporal_diff_filter 입력은 (N, time, subcarriers) 3차원이어야 합니다.")

    diff_list = []
    current = X.astype(np.float32)

    for _ in range(order):
        # np.diff는 time 축 길이를 1 줄이므로, 앞에 0 패딩으로 길이를 맞춘다.
        delta = np.diff(current, n=1, axis=1)          # (N, time-1, subcarriers)
        pad = np.zeros((X.shape[0], 1, current.shape[2]), dtype=np.float32)
        delta = np.concatenate([pad, delta], axis=1)    # (N, time, subcarriers)
        diff_list.append(delta)
        current = delta  # 2차 차분이면 1차 결과를 다시 차분

    if stack_mode == "replace":
        return diff_list[-1]

    # "concat": 원신호와 모든 차수 차분 결과를 feature 축으로 이어붙임
    return np.concatenate([X.astype(np.float32)] + diff_list, axis=2)


# ──────────────────────────────────────────────────────────────────────────────
# 메인 전처리 파이프라인
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(
    X: np.ndarray,
    pca: PCA | None = None,
    scaler: MinMaxScaler | None = None,
    fit: bool = True,
    preprocess_mode: str | None = None,
):
    """
    입력 X  : (N, time, subcarriers)
    출력 X_proc: (N, time, components)

    전처리 순서 (hybrid 모드):
      0) 다운샘플링 (긴 신호 한정)
      1) Hampel 필터  → 이상치 제거
      2) SG 필터      → 고주파 노이즈 평활화
      3) [신규] 시간차분 필터 → 동적 변화 강조 + 원신호 유지(concat)
      4) Min-Max 정규화
      5) PCA           → feature 차원 축소
    """
    if X.ndim != 3:
        raise ValueError("X는 (N, time, subcarriers) 3차원이어야 합니다.")

    N, time_steps, num_subcarriers = X.shape

    # 0) 다운샘플링
    if time_steps >= MIN_LENGTH_FOR_DOWNSAMPLE and DOWNSAMPLE_FACTOR > 1:
        X = X[:, ::DOWNSAMPLE_FACTOR, :]
        N, time_steps, num_subcarriers = X.shape

    active_mode = preprocess_mode or PREPROCESS_MODE

    # ── lite 모드 (경량): 차분 필터 → MinMax만 적용 ──────────────────────────
    if active_mode == "lite":
        # [신규] 차분 필터 적용
        X_diff = temporal_diff_filter(X)
        N, time_steps, num_subcarriers = X_diff.shape

        flattened = X_diff.reshape(-1, num_subcarriers)
        if fit:
            scaler = MinMaxScaler()
            normalized = scaler.fit_transform(flattened)
        else:
            if scaler is None:
                raise ValueError("fit=False일 때는 학습된 scaler가 필요합니다.")
            normalized = scaler.transform(flattened)
        normalized = normalized.reshape(N, time_steps, num_subcarriers).astype(np.float32)
        return normalized, None, scaler

    # ── hybrid 모드 (정확도 우선) ─────────────────────────────────────────────

    # 1) Hampel + SG 필터 (샘플 단위 2D 벡터화)
    filtered = np.empty_like(X, dtype=np.float32)
    for sample_index in range(N):
        # Hampel: (time, subcarriers) 전체를 한 번에 처리
        cleaned_2d = _hampel_filter_2d(X[sample_index], HAMPEL_WINDOW, HAMPEL_SIGMA)

        # SG 필터: scipy가 있으면 axis=0 방향으로 전체 채널에 일괄 적용
        if savgol_filter is not None and cleaned_2d.shape[0] >= SG_WINDOW:
            smoothed_2d = savgol_filter(
                cleaned_2d, window_length=SG_WINDOW,
                polyorder=SG_POLY_ORDER, axis=0, mode="interp",
            ).astype(np.float32)
        else:
            # 폴백: 채널별 이동평균
            kernel = np.ones(5, dtype=np.float32) / 5
            smoothed_2d = np.stack(
                [np.convolve(
                    np.pad(cleaned_2d[:, c], (2, 2), mode="edge"), kernel, mode="valid"
                ) for c in range(cleaned_2d.shape[1])],
                axis=1,
            ).astype(np.float32)

        filtered[sample_index] = smoothed_2d

    # 2) [신규] 시간차분 필터 (Hampel+SG 완료 후 적용)
    #    정제된 신호에 차분을 적용해야 노이즈 증폭 없이 동적 성분만 추출된다.
    filtered = temporal_diff_filter(filtered)
    N, time_steps, num_subcarriers = filtered.shape

    # 3) Min-Max 정규화
    flattened = filtered.reshape(-1, num_subcarriers)
    if fit:
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(flattened)
    else:
        if scaler is None:
            raise ValueError("fit=False일 때는 학습된 scaler가 필요합니다.")
        normalized = scaler.transform(flattened)
    normalized = normalized.reshape(N, time_steps, num_subcarriers)

    # 4) PCA (feature 축 축소)
    feature_for_pca = normalized.reshape(N * time_steps, num_subcarriers)
    max_components = max(1, min(PCA_COMPONENTS, num_subcarriers))
    if fit:
        pca = PCA(n_components=max_components, random_state=42)
        reduced = pca.fit_transform(feature_for_pca)
    else:
        if pca is None:
            reduced = feature_for_pca
        else:
            reduced = pca.transform(feature_for_pca)

    X_processed = reduced.reshape(N, time_steps, -1).astype(np.float32)
    return X_processed, pca, scaler
