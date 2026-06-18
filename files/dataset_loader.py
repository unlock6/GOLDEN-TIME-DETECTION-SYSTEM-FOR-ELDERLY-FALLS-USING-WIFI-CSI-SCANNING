"""
데이터 로더.
1) RAW_DATA_DIR(새 폴더)의 실제 CSI CSV를 읽는다.
2) 없으면 파이프라인 점검용 합성 데이터를 생성한다.

CSV 형식: label | rx | raw_data
  raw_data 예: CSI_DATA,...,"[I0,Q0,I1,Q1,...]"
  → I/Q 쌍에서 진폭 = sqrt(I²+Q²) 계산
  → 연속 프레임을 슬라이딩 윈도우로 분할

라벨 매핑 (이진 분류):
  fall, lie_down → 1 (위험)
  static, stand  → 0 (정상)
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DANGER_CLASS_NAMES,
    MULTI_CLASS_NAMES,
    N_SUBCARRIERS,
    RANDOM_SEED,
    RAW_DATA_DIR,
    TEST_SIZE,
    USE_BINARY_CLASSIFICATION,
    WINDOW_SAMPLES,
    WINDOW_STRIDE,
)

_CSI_PATTERN = re.compile(r"\[([0-9,\s\-]+)\]")


# ──────────────────────────────────────────────────────────────────────────────
# CSI 진폭 추출
# ──────────────────────────────────────────────────────────────────────────────

def _extract_amplitude(raw_data: str) -> np.ndarray | None:
    """raw_data 문자열 → I/Q 쌍 파싱 → 진폭 배열 (길이=N_SUBCARRIERS)."""
    if not isinstance(raw_data, str):
        return None
    match = _CSI_PATTERN.search(raw_data)
    if not match:
        return None
    try:
        vals = ast.literal_eval("[" + match.group(1) + "]")
        arr = np.array(vals, dtype=np.float32)
        if len(arr) % 2 == 0:
            I = arr[0::2]
            Q = arr[1::2]
            amp = np.sqrt(I**2 + Q**2)
        else:
            amp = np.abs(arr)
        return amp[:N_SUBCARRIERS] if len(amp) >= N_SUBCARRIERS else None
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# CSV 파일 → 슬라이딩 윈도우
# ──────────────────────────────────────────────────────────────────────────────

def _csv_to_windows(file_path: Path) -> Tuple[np.ndarray, str | None]:
    """
    단일 CSV 파일을 읽어 슬라이딩 윈도우 배열과 클래스 이름을 반환한다.
    반환: (windows: (n, WINDOW_SAMPLES, N_SUBCARRIERS), class_name | None)
    """
    try:
        df = pd.read_csv(file_path, encoding="cp949")
    except Exception as e:
        print(f"[경고] 파일 읽기 실패: {file_path.name} ({e})")
        return np.empty((0, WINDOW_SAMPLES, N_SUBCARRIERS), dtype=np.float32), None

    if "raw_data" not in df.columns or "label" not in df.columns:
        return np.empty((0, WINDOW_SAMPLES, N_SUBCARRIERS), dtype=np.float32), None

    class_name = str(df["label"].iloc[0]).strip().lower()

    frames = []
    for raw in df["raw_data"]:
        amp = _extract_amplitude(str(raw))
        if amp is not None:
            frames.append(amp)

    if len(frames) < WINDOW_SAMPLES:
        return np.empty((0, WINDOW_SAMPLES, N_SUBCARRIERS), dtype=np.float32), class_name

    signal = np.array(frames, dtype=np.float32)  # (T, N_SUBCARRIERS)

    windows = []
    for start in range(0, len(signal) - WINDOW_SAMPLES + 1, WINDOW_STRIDE):
        windows.append(signal[start : start + WINDOW_SAMPLES])

    if not windows:
        return np.empty((0, WINDOW_SAMPLES, N_SUBCARRIERS), dtype=np.float32), class_name

    return np.array(windows, dtype=np.float32), class_name


# ──────────────────────────────────────────────────────────────────────────────
# 실제 데이터 로드 (RAW_DATA_DIR)
# ──────────────────────────────────────────────────────────────────────────────

def _load_real_dataset() -> Tuple[List[np.ndarray], np.ndarray]:
    """
    RAW_DATA_DIR 안의 CSV 파일을 읽어 샘플 리스트와 라벨을 반환한다.

    - RX1 파일만 사용 (RX2는 앙상블용 — 중복 방지)
    - 라벨은 CSV의 label 컬럼에서 직접 읽는다
    - 이진 분류: DANGER_CLASS_NAMES ∈ 1, 나머지 → 0
    """
    if not RAW_DATA_DIR.exists():
        print(f"[경고] RAW_DATA_DIR 없음: {RAW_DATA_DIR}")
        return [], np.array([], dtype=np.int64)

    # RX1 CSV 파일만 선택 (대소문자 모두 처리)
    all_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    rx1_files = [f for f in all_files if "rx1" in f.name.lower()]

    if not rx1_files:
        print(f"[경고] RX1 CSV 파일 없음: {RAW_DATA_DIR}")
        return [], np.array([], dtype=np.int64)

    sample_list: List[np.ndarray] = []
    label_list: List[int] = []
    class_counts: dict[str, int] = {}

    for file_path in rx1_files:
        windows, class_name = _csv_to_windows(file_path)
        if class_name is None or windows.shape[0] == 0:
            continue

        if USE_BINARY_CLASSIFICATION:
            binary_label = 1 if class_name in DANGER_CLASS_NAMES else 0
        else:
            if class_name not in MULTI_CLASS_NAMES:
                print(f"[경고] 알 수 없는 클래스: {class_name} ({file_path.name})")
                continue
            binary_label = MULTI_CLASS_NAMES.index(class_name)

        sample_list.extend([windows[i] for i in range(len(windows))])
        label_list.extend([binary_label] * len(windows))
        class_counts[class_name] = class_counts.get(class_name, 0) + len(windows)

    if not sample_list:
        return [], np.array([], dtype=np.int64)

    for cls, cnt in sorted(class_counts.items()):
        lbl = 1 if cls in DANGER_CLASS_NAMES else 0
        print(f"  [{cls}] {cnt:,}개 윈도우 → 라벨 {lbl}")

    return sample_list, np.array(label_list, dtype=np.int64)


# ──────────────────────────────────────────────────────────────────────────────
# 합성 데이터 (실제 데이터 없을 때 폴백)
# ──────────────────────────────────────────────────────────────────────────────

def _generate_synthetic_dataset(
    samples_per_class: int = 60,
    sequence_length: int = WINDOW_SAMPLES,
    subcarriers: int = N_SUBCARRIERS,
) -> Tuple[np.ndarray, np.ndarray]:
    """실제 데이터가 없을 때 기본 동작 확인용 합성 데이터를 생성한다."""
    rng = np.random.default_rng(RANDOM_SEED)
    all_samples, all_labels = [], []

    time_axis    = np.linspace(0, 1, sequence_length, dtype=np.float32)
    carrier_axis = np.linspace(0, 1, subcarriers, dtype=np.float32)

    n_classes = len(MULTI_CLASS_NAMES)
    for class_index in range(n_classes):
        base_freq = 1.5 + class_index * 0.6
        base_amp  = 0.5 + class_index * 0.2
        for _ in range(samples_per_class):
            temporal = np.sin(2 * np.pi * base_freq * time_axis)[:, None]
            spatial  = np.cos(np.pi * (class_index + 1) * carrier_axis)[None, :]
            noise    = rng.normal(0, 0.1, size=(sequence_length, subcarriers)).astype(np.float32)
            sample   = base_amp * temporal * spatial + noise
            all_samples.append(sample.astype(np.float32))

            if USE_BINARY_CLASSIFICATION:
                cls_name = MULTI_CLASS_NAMES[class_index]
                all_labels.append(1 if cls_name in DANGER_CLASS_NAMES else 0)
            else:
                all_labels.append(class_index)

    return np.array(all_samples, dtype=np.float32), np.array(all_labels, dtype=np.int64)


# ──────────────────────────────────────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset():
    """
    데이터셋을 train/test로 분할해 반환한다.
    반환 형태:
      X_train, X_test : (N, WINDOW_SAMPLES, N_SUBCARRIERS)
      y_train, y_test : (N,)
    """
    sample_list, labels = _load_real_dataset()

    if sample_list:
        features = np.array(sample_list, dtype=np.float32)
        print(f"[데이터] 실제 파일 데이터 사용: {features.shape[0]:,}개 윈도우")
    else:
        features, labels = _generate_synthetic_dataset()
        print("[데이터] 실제 데이터가 없어 합성 데이터로 실행합니다.")

    unique_labels = np.unique(labels)

    # 클래스가 하나뿐이면 합성 데이터로 나머지 클래스를 보충한다.
    if len(unique_labels) < 2:
        print(f"[경고] 라벨이 {unique_labels.tolist()} 하나뿐입니다. "
              "정상(라벨 0) 합성 데이터를 추가합니다.")
        synth_X, synth_y = _generate_synthetic_dataset(
            samples_per_class=max(60, len(features) // 4),
            sequence_length=features.shape[1],
            subcarriers=features.shape[2],
        )
        # 합성 데이터 중 빠진 라벨만 골라서 합친다.
        missing = [l for l in [0, 1] if l not in unique_labels]
        mask = np.isin(synth_y, missing)
        features = np.concatenate([features, synth_X[mask]])
        labels   = np.concatenate([labels,   synth_y[mask]])
        unique_labels = np.unique(labels)
        print(f"[데이터] 보충 후 라벨 분포: "
              f"{dict(zip(*np.unique(labels, return_counts=True)))}")

    stratify = labels if len(unique_labels) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )
    return X_train, X_test, y_train, y_test
