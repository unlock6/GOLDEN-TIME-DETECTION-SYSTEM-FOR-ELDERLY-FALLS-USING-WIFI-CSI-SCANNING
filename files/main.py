"""
메인 진입점.
실행 모드:
  - train: 모델 학습
  - eval:  저장된 모델 평가
  - realtime: 실시간(합성 스트림 기반) 추론
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np

from config import MODEL_DIR, SERIAL_PORT, WINDOW_SAMPLES
from cnn_lstm import (
    build_model,
    compile_and_train,
    evaluate,
    load_model,
    prepare_model_inputs,
    save_model,
)
from csi_pipeline import preprocess, sliding_window
from dataset_loader import load_dataset
from inference_engine import FallDetectionEngine
from visualizer import plot_realtime_dashboard, plot_training_history


def _window_if_needed(X: np.ndarray, y: np.ndarray):
    """입력 샘플 길이가 윈도우 길이와 다를 때 윈도우 분할을 적용한다."""
    if X.shape[1] == WINDOW_SAMPLES:
        return X, y

    windows, labels = [], []
    for sample, label in zip(X, y):
        sample_windows, sample_labels = sliding_window(sample, int(label))
        windows.append(sample_windows)
        labels.append(sample_labels)
    return np.concatenate(windows), np.concatenate(labels)


def _model_path(model_type: str) -> str:
    return os.path.join(MODEL_DIR, f"best_model_{model_type}.h5")


def _resolve_preprocess_mode(model_type: str) -> str:
    """
    모델별 권장 전처리 정책.
    - cnn_lstm: 정확도 중심 hybrid
    - efficientnet: 논문 스타일 경량 lite
    """
    return "lite" if model_type == "efficientnet" else "hybrid"


def train(model_type: str):
    print("\n" + "=" * 60)
    print("[모드] 학습 시작")
    print("=" * 60)

    # 1) 데이터 로드
    X_train, X_test, y_train, y_test = load_dataset()
    print(f"[데이터] Train={X_train.shape}, Test={X_test.shape}")

    # 2) 슬라이딩 윈도우 (필요 시)
    X_train, y_train = _window_if_needed(X_train, y_train)
    X_test, y_test = _window_if_needed(X_test, y_test)

    # 3) 전처리 (Hampel -> SG -> 정규화 -> PCA)
    preprocess_mode = _resolve_preprocess_mode(model_type)
    X_train_proc, pca, scaler = preprocess(X_train, fit=True, preprocess_mode=preprocess_mode)
    X_test_proc, _, _ = preprocess(
        X_test,
        pca=pca,
        scaler=scaler,
        fit=False,
        preprocess_mode=preprocess_mode,
    )
    print(f"[전처리] Train={X_train_proc.shape}, Test={X_test_proc.shape}")

    # 4) 전처리 객체 저장 (실시간/평가 재사용)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as file:
        pickle.dump(scaler, file)
    if pca is not None:
        with open(os.path.join(MODEL_DIR, "pca.pkl"), "wb") as file:
            pickle.dump(pca, file)

    # 5) 모델 타입별 입력 변환 + 학습
    X_train_input = prepare_model_inputs(X_train_proc, model_type=model_type)
    X_test_input = prepare_model_inputs(X_test_proc, model_type=model_type)
    model = build_model(input_shape=X_train_input.shape[1:], model_type=model_type)
    history = compile_and_train(model, X_train_input, y_train, X_test_input, y_test)

    # 6) 시각화 + 저장 + 평가
    plot_training_history(history)
    model_path = _model_path(model_type)
    save_model(model, model_path)
    evaluate(model, X_test_input, y_test)
    print(f"[완료] 모델 저장: {model_path}")


def eval_only(model_type: str):
    model_path = _model_path(model_type)
    pca_path = os.path.join(MODEL_DIR, "pca.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    if not (os.path.exists(model_path) or os.path.exists(model_path.replace(".h5", ".pkl"))):
        print("[오류] 학습 모델이 없습니다. 먼저 train 모드를 실행하세요.")
        return
    if not os.path.exists(scaler_path):
        print("[오류] 전처리 객체(scaler)가 없습니다. 먼저 train 모드를 실행하세요.")
        return

    model = load_model(model_path)
    pca = None
    if os.path.exists(pca_path):
        with open(pca_path, "rb") as file:
            pca = pickle.load(file)
    with open(scaler_path, "rb") as file:
        scaler = pickle.load(file)

    _, X_test, _, y_test = load_dataset()
    X_test, y_test = _window_if_needed(X_test, y_test)
    preprocess_mode = _resolve_preprocess_mode(model_type)
    X_test_proc, _, _ = preprocess(
        X_test,
        pca=pca,
        scaler=scaler,
        fit=False,
        preprocess_mode=preprocess_mode,
    )
    X_test_input = prepare_model_inputs(X_test_proc, model_type=model_type)
    evaluate(model, X_test_input, y_test)


def realtime(port: str, model_type: str):
    model_path = _model_path(model_type)
    pca_path = os.path.join(MODEL_DIR, "pca.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    if not (os.path.exists(model_path) or os.path.exists(model_path.replace(".h5", ".pkl"))):
        print("[오류] 학습 모델이 없습니다. 먼저 train 모드를 실행하세요.")
        return
    if not os.path.exists(scaler_path):
        print("[오류] 전처리 객체(scaler)가 없습니다. 먼저 train 모드를 실행하세요.")
        return

    pca = None
    if os.path.exists(pca_path):
        with open(pca_path, "rb") as file:
            pca = pickle.load(file)
    with open(scaler_path, "rb") as file:
        scaler = pickle.load(file)

    preprocess_mode = _resolve_preprocess_mode(model_type)
    engine = FallDetectionEngine(
        model_path=model_path,
        pca=pca,
        scaler=scaler,
        model_type=model_type,
        preprocess_mode=preprocess_mode,
    )
    engine.start(port=port)
    print(f"[실시간] 실행 중... 포트={port}, 종료하려면 Ctrl+C")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        engine.stop()
        print("[실시간] 종료")
        plot_realtime_dashboard(engine.history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WiFi CSI 기반 노인 낙상 감지 시스템")
    parser.add_argument("--mode", choices=["train", "eval", "realtime"], default="train")
    parser.add_argument("--model", choices=["cnn_lstm", "efficientnet"], default="cnn_lstm")
    parser.add_argument("--port", default=SERIAL_PORT)
    args = parser.parse_args()

    if args.mode == "train":
        train(model_type=args.model)
    elif args.mode == "eval":
        eval_only(model_type=args.model)
    else:
        realtime(args.port, model_type=args.model)
