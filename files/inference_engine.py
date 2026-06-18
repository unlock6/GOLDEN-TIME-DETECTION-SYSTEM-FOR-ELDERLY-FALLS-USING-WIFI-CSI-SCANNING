"""
실시간 추론 엔진.
현재는 입력 소스(시리얼/네트워크)가 없을 때도 동작 점검이 가능하도록
합성 스트림 모드를 함께 제공한다.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

import numpy as np

from csi_pipeline import preprocess
from cnn_lstm import load_model, prepare_model_inputs
from config import CLASS_NAMES, REALTIME_INTERVAL_SEC, WINDOW_SAMPLES


class FallDetectionEngine:
    """실시간 CSI 윈도우를 받아 활동 클래스를 예측하는 엔진."""

    def __init__(
        self,
        model_path: str,
        pca=None,
        scaler=None,
        model_type: str = "cnn_lstm",
        preprocess_mode: str = "hybrid",
        on_fall_detected=None,     # 낙상 감지 시 호출할 콜백 함수
        on_normal_detected=None,   # 정상 복귀 시 호출할 콜백 함수
        fall_confirm_count: int = 3,  # 연속 N회 fall 예측 시 확정
    ):
        self.model = load_model(model_path)
        self.pca = pca
        self.scaler = scaler
        self.model_type = model_type
        self.preprocess_mode = preprocess_mode
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.history: Dict[str, List] = {"predictions": [], "labels": []}

        # 콜백 및 확정 카운터
        self.on_fall_detected   = on_fall_detected
        self.on_normal_detected = on_normal_detected
        self.fall_confirm_count = fall_confirm_count
        self._consecutive_fall   = 0   # 연속 fall 카운터
        self._fall_active        = False  # 현재 낙상 경보 활성 여부

    def _read_one_window(self, _port: Optional[str]) -> np.ndarray:
        """
        실제 환경에서는 시리얼/소켓에서 CSI를 읽어야 한다.
        현재는 기본 동작 확인을 위해 합성 CSI를 생성한다.
        """
        subcarriers = 53
        time_axis = np.linspace(0, 1, WINDOW_SAMPLES, dtype=np.float32)
        carrier_axis = np.linspace(0, 1, subcarriers, dtype=np.float32)
        signal = np.sin(2 * np.pi * 2.0 * time_axis)[:, None] * np.cos(2 * np.pi * carrier_axis)[None, :]
        noise = np.random.normal(0, 0.07, size=signal.shape)
        return (signal + noise).astype(np.float32)

    def _predict(self, window: np.ndarray) -> int:
        input_batch = window[None, :, :]
        processed, _, _ = preprocess(
            input_batch,
            pca=self.pca,
            scaler=self.scaler,
            fit=False,
            preprocess_mode=self.preprocess_mode,
        )
        model_input = prepare_model_inputs(processed, model_type=self.model_type)

        output = self.model.predict(model_input)
        if isinstance(output, np.ndarray) and output.ndim > 1:
            predicted_index = int(np.argmax(output, axis=1)[0])
        elif isinstance(output, np.ndarray):
            predicted_index = int(output[0])
        else:
            predicted_index = int(output)
        return predicted_index

    def _loop(self, port: Optional[str]):
        while self.is_running:
            try:
                window = self._read_one_window(port)
                predicted_index = self._predict(window)
                predicted_label = CLASS_NAMES[predicted_index]
                self.history["predictions"].append(predicted_index)
                self.history["labels"].append(predicted_label)
                print(f"[실시간] 예측: {predicted_label}")

                # ── 낙상 확정 로직 ────────────────────────────────────────
                is_fall = (predicted_label == CLASS_NAMES[1])  # "fall"

                if is_fall:
                    self._consecutive_fall += 1
                    # 연속 N회 이상 fall → 낙상 확정 (첫 확정 시에만 콜백)
                    if (self._consecutive_fall >= self.fall_confirm_count
                            and not self._fall_active):
                        self._fall_active = True
                        print(f"[경보] 낙상 확정! (연속 {self._consecutive_fall}회)")
                        if self.on_fall_detected:
                            self.on_fall_detected()
                else:
                    self._consecutive_fall = 0
                    # 낙상 상태에서 정상으로 복귀
                    if self._fall_active:
                        self._fall_active = False
                        print("[경보] 정상 복귀 감지")
                        if self.on_normal_detected:
                            self.on_normal_detected()
                # ─────────────────────────────────────────────────────────

            except Exception as error:
                print(f"[실시간][경고] 추론 실패: {error}")
            time.sleep(REALTIME_INTERVAL_SEC)

    def start(self, port: Optional[str] = None):
        self.is_running = True
        self.thread = threading.Thread(target=self._loop, args=(port,), daemon=True)
        self.thread.start()
        return self.thread, None

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
