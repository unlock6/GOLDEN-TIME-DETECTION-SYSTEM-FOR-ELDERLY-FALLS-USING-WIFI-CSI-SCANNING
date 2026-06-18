"""
실시간 낙상 감지 시스템 (UDP 수신 버전)

하드웨어 연결 구조:
  ESP32-S3 ──(Serial 921600)──► Raspberry Pi
                                      │
                               rpi_sender_rx1.py
                                      │  UDP Port 5000
                                      ▼
                              이 PC (realtime_detector.py)
                                      │
                              FallAlertTimer GUI

실행 순서:
  1. 이 PC의 IP 확인:  ipconfig  →  PC_IP 수정
  2. rpi_sender_rx1.py 의 PC_IP 를 이 PC의 IP로 변경 후 라즈베리파이에서 실행
  3. 이 스크립트 실행:  python realtime_detector.py
"""

from __future__ import annotations

import ast
import re
import socket
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from fall_timer import FallAlertTimer

# ══════════════════════════════════════════════════════════════════════════════
# 설정  ←  여기만 수정하면 됩니다
# ══════════════════════════════════════════════════════════════════════════════

UDP_HOST        = "0.0.0.0"    # 모든 인터페이스에서 수신
UDP_PORT_RX1    = 5000         # RX1 수신 포트 (rpi_sender_rx1.py 와 일치)
UDP_PORT_RX2    = 5001         # RX2 수신 포트 (선택, 사용 안 할 시 None)

DATA_DIR        = Path(r"C:\Users\주현준\OneDrive\Desktop\새 폴더")

WINDOW_SAMPLES  = 100          # 윈도우 프레임 수
WINDOW_STRIDE   = 20           # 예측 간격 (프레임)
N_SUB           = 64           # 서브캐리어 수 (I/Q 128 → 진폭 64)
FALL_CONFIRM    = 3            # 연속 N회 fall 예측 시 경보 확정
NORMAL_CONFIRM  = 5            # 연속 N회 normal 예측 시 경보 해제

# ══════════════════════════════════════════════════════════════════════════════

LABEL_MAP = {"fall": 1, "lie_down": 1, "static": 0, "stand": 0}
_PAT      = re.compile(r"\[([0-9,\s\-]+)\]")


# ──────────────────────────────────────────────────────────────────────────────
# CSI 진폭 파싱
# ──────────────────────────────────────────────────────────────────────────────

def parse_csi_line(line: str) -> np.ndarray | None:
    """
    'RX1_동|CSI_DATA,...,[I0,Q0,...]' 또는
    'CSI_DATA,...,[I0,Q0,...]' 형식에서 진폭 벡터 반환
    """
    if "|" in line:
        _, line = line.split("|", 1)
    line = line.strip()
    if not line.startswith("CSI_DATA"):
        return None
    m = _PAT.search(line)
    if not m:
        return None
    try:
        v = np.array(ast.literal_eval("[" + m.group(1) + "]"), dtype=np.float32)
        if len(v) % 2 == 0 and len(v) // 2 >= N_SUB:
            return np.sqrt(v[0::2]**2 + v[1::2]**2)[:N_SUB]
        if len(v) >= N_SUB:
            return np.abs(v[:N_SUB])
        return None
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 특징 추출 (run_demo.py 와 동일)
# ──────────────────────────────────────────────────────────────────────────────

def extract_feat(window: np.ndarray) -> np.ndarray:
    d1 = np.diff(window, n=1, axis=0)
    d2 = np.diff(window, n=2, axis=0)
    mu = window.mean(0); sig = window.std(0) + 1e-8
    kurt = (((window - mu) / sig) ** 4).mean(0)
    e_t  = (window ** 2).sum(1)
    return np.concatenate([
        mu, sig, window.max(0),
        np.abs(d1).mean(0), np.abs(d1).max(0),
        np.abs(d2).mean(0),
        kurt, (window ** 2).sum(0),
        [e_t.std(), e_t.max(), e_t.mean()],
    ]).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 모델 학습 (수집 데이터 기반)
# ──────────────────────────────────────────────────────────────────────────────

def _load_file_windows(path: Path) -> tuple[list[np.ndarray], str]:
    df  = pd.read_csv(path, encoding="cp949")
    cls = str(df["label"].iloc[0]).strip().lower()
    frames = [a for raw in df["raw_data"]
              if (a := _parse_raw(str(raw))) is not None]
    stride = WINDOW_SAMPLES // 2
    wins   = [np.array(frames[s:s + WINDOW_SAMPLES], np.float32)
              for s in range(0, len(frames) - WINDOW_SAMPLES + 1, stride)]
    return wins, cls


def _parse_raw(raw: str) -> np.ndarray | None:
    m = _PAT.search(raw)
    if not m:
        return None
    try:
        v = np.array(ast.literal_eval("[" + m.group(1) + "]"), dtype=np.float32)
        amp = np.sqrt(v[0::2]**2 + v[1::2]**2) if len(v) % 2 == 0 else v
        return amp[:N_SUB] if len(amp) >= N_SUB else None
    except Exception:
        return None


def train_model(data_dir: Path) -> Pipeline:
    """수집 데이터로 RandomForest 학습 (수 초 소요)"""
    files = sorted(f for f in data_dir.glob("*.csv") if "rx1" in f.name.lower())
    X_list, y_list = [], []

    for f in files:
        wins, cls = _load_file_windows(f)
        lbl = LABEL_MAP.get(cls, -1)
        if lbl == -1 or not wins:
            continue
        feats = np.array([extract_feat(w) for w in wins], np.float32)
        X_list.append(feats)
        y_list.append(np.full(len(wins), lbl, np.int64))
        print(f"  [{cls:10s}]  {len(wins):>5}개  ({f.name[:40]})")

    if not X_list:
        raise RuntimeError(f"학습 데이터 없음: {data_dir}")

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)

    pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X, y)
    acc = (pipe.predict(X) == y).mean()
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    print(f"\n  학습 완료  정확도: {acc*100:.1f}%  "
          f"[정상:{n0}  위험:{n1}]")
    return pipe


# ──────────────────────────────────────────────────────────────────────────────
# UDP 수신 + 슬라이딩 윈도우 추론 엔진
# ──────────────────────────────────────────────────────────────────────────────

class RealtimeEngine:
    """
    UDP로 CSI 프레임을 받아 슬라이딩 윈도우로 낙상을 감지하고
    FallAlertTimer GUI를 업데이트한다.
    """

    def __init__(self, pipe: Pipeline, app: FallAlertTimer,
                 host: str = UDP_HOST, port: int = UDP_PORT_RX1):
        self.pipe     = pipe
        self.app      = app
        self.host     = host
        self.port     = port
        self.running  = False

        # 슬라이딩 윈도우 버퍼
        self._buf: deque[np.ndarray] = deque(maxlen=WINDOW_SAMPLES + WINDOW_STRIDE)
        self._since_last_pred = 0   # 마지막 예측 이후 수신 프레임 수

        # 연속 카운터
        self._consec_fall   = 0
        self._consec_normal = 0

        # 통계
        self.total_frames = 0
        self.total_preds  = 0
        self.fall_preds   = 0

    def start(self):
        self.running = True
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(1.0)
        threading.Thread(target=self._recv_loop, daemon=True).start()
        print(f"  UDP 수신 시작  {self.host}:{self.port}")

    def stop(self):
        self.running = False
        try:
            self._sock.close()
        except Exception:
            pass

    def _recv_loop(self):
        buf_text = ""
        print(f"\n  [대기 중]  라즈베리파이에서 데이터 전송을 기다리는 중...")

        while self.running:
            try:
                data, addr = self._sock.recvfrom(8192)
                text = data.decode("utf-8", errors="ignore")
            except socket.timeout:
                continue
            except Exception:
                break

            # 첫 프레임 수신 알림
            if self.total_frames == 0:
                print(f"\n  [수신 시작]  {addr[0]}:{addr[1]} 에서 데이터 수신 중")

            amp = parse_csi_line(text)
            if amp is None:
                continue

            self._buf.append(amp)
            self.total_frames += 1
            self._since_last_pred += 1

            # WINDOW_STRIDE 프레임마다 예측
            if (len(self._buf) >= WINDOW_SAMPLES and
                    self._since_last_pred >= WINDOW_STRIDE):
                self._since_last_pred = 0
                self._predict_current()

    def _predict_current(self):
        window = np.array(list(self._buf)[-WINDOW_SAMPLES:], np.float32)
        feat   = extract_feat(window).reshape(1, -1)

        label  = int(self.pipe.predict(feat)[0])
        conf   = float(self.pipe.predict_proba(feat)[0][label])
        self.total_preds += 1

        # 연속 카운터 갱신
        if label == 1:
            self._consec_fall   += 1
            self._consec_normal  = 0
            self.fall_preds     += 1
        else:
            self._consec_normal += 1
            self._consec_fall    = 0

        # GUI 신뢰도 업데이트
        self.app.confidence = conf if label == 1 else (1 - conf)

        # 상태 출력
        state = "낙상 감지!" if label == 1 else "정상      "
        bar   = f"F×{self._consec_fall}" if label == 1 else f"N×{self._consec_normal}"
        print(f"  프레임 {self.total_frames:>6} | {state} | "
              f"{conf*100:5.1f}% | {bar:>5} | "
              f"총예측:{self.total_preds}  낙상:{self.fall_preds}")

        # 낙상 경보 트리거
        if self._consec_fall >= FALL_CONFIRM and not self.app.fall_detected:
            self.app.root.after(0, lambda c=conf: _safe_trigger(self.app, c))
            print(f"  >>> 경보 활성화! (연속 {self._consec_fall}회 낙상 예측)")

        # 정상 복귀
        elif self._consec_normal >= NORMAL_CONFIRM and self.app.fall_detected:
            self.app.root.after(0, self.app.reset)
            print(f"  >>> 정상 복귀 (연속 {self._consec_normal}회 정상 예측)")


def _safe_trigger(app: FallAlertTimer, conf: float):
    if not app.fall_detected:
        app.trigger_fall(confidence=conf)


# ──────────────────────────────────────────────────────────────────────────────
# 연결 상태 확인
# ──────────────────────────────────────────────────────────────────────────────

def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "확인 불가"


def print_setup_guide(local_ip: str):
    sep = "=" * 58
    print(f"\n{sep}")
    print("  실시간 낙상 감지  —  연결 설정 가이드")
    print(sep)
    print(f"\n  이 PC의 IP 주소:  {local_ip}")
    print(f"  수신 포트   RX1:  {UDP_PORT_RX1}")
    print()
    print("  [라즈베리파이 RX1] rpi_sender_rx1.py 수정:")
    print(f"    PC_IP = '{local_ip}'   ← 이 값으로 변경")
    print( "    UDP_PORT = 5000")
    print()
    print("  [실행 순서]")
    print("    1. 라즈베리파이에서:")
    print("       python rpi_sender_rx1.py")
    print("    2. 이 PC에서 (이미 실행 중):")
    print("       python realtime_detector.py")
    print()
    print("  ※ 같은 Wi-Fi 네트워크에 연결되어 있어야 합니다.")
    print(sep)


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 58)
    print("  CSI 실시간 낙상 감지 시스템")
    print("=" * 58)

    local_ip = get_local_ip()
    print_setup_guide(local_ip)

    # 1) 모델 학습
    print("\n[1단계]  수집 데이터로 모델 학습 중...")
    pipe = train_model(DATA_DIR)

    # 2) GUI 초기화
    print("\n[2단계]  GUI 시작")
    app = FallAlertTimer()

    # 3) 추론 엔진 시작
    print("\n[3단계]  UDP 수신 대기")
    engine = RealtimeEngine(pipe, app,
                            host=UDP_HOST, port=UDP_PORT_RX1)
    engine.start()

    # 4) GUI 실행 (메인 스레드)
    try:
        app.run()
    finally:
        engine.stop()
        print("\n[종료]  수신 통계")
        print(f"  수신 프레임 : {engine.total_frames:,}")
        print(f"  총 예측 수  : {engine.total_preds:,}")
        print(f"  낙상 예측   : {engine.fall_preds:,}회")
        if engine.total_preds > 0:
            print(f"  낙상 비율   : {engine.fall_preds/engine.total_preds*100:.1f}%")


if __name__ == "__main__":
    main()
