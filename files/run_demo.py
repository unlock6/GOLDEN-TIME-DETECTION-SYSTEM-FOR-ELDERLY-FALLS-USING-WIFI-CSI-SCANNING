"""
수집된 실제 CSI 데이터 기반 낙상 감지 실시간 데모

흐름:
  1) 새 폴더의 실제 CSI 데이터 (4개 클래스) 로드
  2) RandomForest 모델 즉시 학습 (수 초 소요)
  3) 실제 윈도우를 순서대로 스트리밍하며 예측
  4) FallAlertTimer GUI에 결과 실시간 반영

  시연 순서: [정상] static → stand → [위험] fall → lie_down → 반복
"""

from __future__ import annotations

import ast
import pickle
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from fall_timer import FallAlertTimer

# ── 경로 ──────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(r"C:\Users\주현준\OneDrive\Desktop\새 폴더")
ARTIFACTS   = Path(__file__).parent / "artifacts"

# ── 파라미터 ──────────────────────────────────────────────────────────────────
WINDOW_SAMPLES   = 100
N_SUB            = 64
STREAM_INTERVAL  = 0.6   # 윈도우 간격 (초) — 빠르게 보려면 줄이기
FALL_CONFIRM     = 3     # 연속 N회 fall 예측 시 경보 확정
NORMAL_CONFIRM   = 4     # 연속 N회 normal 예측 시 경보 해제

LABEL_MAP = {"fall": 1, "lie_down": 1, "static": 0, "stand": 0}
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
        amp = np.sqrt(v[0::2]**2 + v[1::2]**2) if len(v) % 2 == 0 else v
        return amp[:N_SUB] if len(amp) >= N_SUB else None
    except Exception:
        return None


def load_windows(path: Path) -> tuple[list[np.ndarray], str, int]:
    """CSV → 윈도우 리스트 + 클래스명 + 라벨"""
    df  = pd.read_csv(path, encoding="cp949")
    cls = str(df["label"].iloc[0]).strip().lower()
    lbl = LABEL_MAP.get(cls, -1)
    if lbl == -1:
        return [], cls, lbl

    frames = [a for raw in df["raw_data"] if (a := _amp(str(raw))) is not None]
    stride = WINDOW_SAMPLES // 2
    wins = [np.array(frames[s:s + WINDOW_SAMPLES], np.float32)
            for s in range(0, len(frames) - WINDOW_SAMPLES + 1, stride)]
    return wins, cls, lbl


def load_all(data_dir: Path) -> dict[str, list[np.ndarray]]:
    """RX1 파일 기준으로 클래스별 윈도우 로드"""
    files = sorted(f for f in data_dir.glob("*.csv") if "rx1" in f.name.lower())
    pool: dict[str, list] = {c: [] for c in LABEL_MAP}

    for f in files:
        wins, cls, lbl = load_windows(f)
        if lbl == -1 or not wins:
            continue
        pool[cls].extend(wins)
        print(f"  [{cls:10s}] {f.name[:45]:<45}  {len(wins):>5}개 윈도우")

    return pool


# ══════════════════════════════════════════════════════════════════════════════
# 2. 특징 추출
# ══════════════════════════════════════════════════════════════════════════════

def extract_feat(window: np.ndarray) -> np.ndarray:
    """(T, S) → 1D 특징 벡터"""
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
    ])


def feat_batch(windows: list[np.ndarray]) -> np.ndarray:
    return np.array([extract_feat(w) for w in windows], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 모델 학습
# ══════════════════════════════════════════════════════════════════════════════

def train_model(pool: dict[str, list]) -> Pipeline:
    print("\n  모델 학습 중...")
    X_list, y_list = [], []
    for cls, wins in pool.items():
        if not wins:
            continue
        X_list.append(feat_batch(wins))
        y_list.append(np.full(len(wins), LABEL_MAP[cls], dtype=np.int64))
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)

    pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X, y)

    # 간단 검증
    y_pred = pipe.predict(X)
    acc = (y_pred == y).mean()
    print(f"  학습 완료  |  전체 정확도: {acc * 100:.1f}%  |  샘플: {len(y):,}개")
    return pipe


# ══════════════════════════════════════════════════════════════════════════════
# 4. 실시간 스트리밍
# ══════════════════════════════════════════════════════════════════════════════

SEP = "─" * 60

def stream(app: FallAlertTimer, pipe: Pipeline, pool: dict[str, list]):
    """백그라운드 스레드: 실제 윈도우를 순서대로 흘려 GUI 갱신"""

    # 시연 시퀀스 (클래스명, 최대 윈도우 수)
    sequence = [
        ("stand",    20),
        ("static",   15),
        ("fall",     25),
        ("lie_down", 20),
    ]

    state_kr = {0: "정상", 1: "낙상 감지!"}
    consec_fall   = 0
    consec_normal = 0
    round_n = 0

    while True:
        round_n += 1
        print(f"\n{SEP}")
        print(f"  [ 라운드 {round_n} ]  실제 데이터 스트리밍")
        print(SEP)

        for cls, max_n in sequence:
            wins = pool.get(cls, [])
            if not wins:
                print(f"  [{cls}] 데이터 없음 — 건너뜀")
                continue

            sample = wins[:max_n]
            true_lbl = LABEL_MAP[cls]
            lbl_kr   = "정상(0)" if true_lbl == 0 else "위험(1)"
            print(f"\n  ▸ {cls:10s}  (실제 라벨: {lbl_kr},  {len(sample)}개 재생)")
            print(f"  {'#':>4}  {'예측':^10}  {'확률':>7}  {'연속':>4}  {'판정'}")
            print(f"  {'─'*42}")

            for i, window in enumerate(sample):
                feat  = extract_feat(window).reshape(1, -1)
                pred  = int(pipe.predict(feat)[0])
                conf  = float(pipe.predict_proba(feat)[0][pred])

                # 연속 카운터
                if pred == 1:
                    consec_fall   += 1
                    consec_normal  = 0
                else:
                    consec_normal += 1
                    consec_fall    = 0

                # GUI 트리거
                if consec_fall >= FALL_CONFIRM and not app.fall_detected:
                    app.root.after(0, lambda c=conf: _trigger(app, c))

                elif consec_normal >= NORMAL_CONFIRM and app.fall_detected:
                    app.root.after(0, app.reset)

                # 신뢰도 실시간 업데이트
                app.confidence = conf if pred == 1 else (1 - conf)

                ok  = "O" if pred == true_lbl else "X"
                print(f"  {i+1:>4}  {state_kr[pred]:^10}  "
                      f"{conf*100:>6.1f}%  "
                      f"{'F' if pred==1 else 'N'}x{consec_fall if pred==1 else consec_normal:<2}"
                      f"  [{ok}]")

                time.sleep(STREAM_INTERVAL)

        print(f"\n  라운드 {round_n} 완료. 3초 후 재시작...")
        time.sleep(3)


def _trigger(app: FallAlertTimer, conf: float):
    """메인 스레드에서 안전하게 낙상 트리거"""
    if not app.fall_detected:
        app.trigger_fall(confidence=conf)


# ══════════════════════════════════════════════════════════════════════════════
# 5. 메인
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  수집 데이터 기반 낙상 감지 시스템  실시간 데모")
    print("=" * 60)

    # 1) 데이터 로드
    print(f"\n[1단계] 실제 CSI 데이터 로드  ({DATA_DIR})")
    pool = load_all(DATA_DIR)
    total = sum(len(v) for v in pool.values())
    if total == 0:
        print("  [오류] 데이터가 없습니다.")
        return
    print(f"\n  총 {total:,}개 윈도우 로드 완료")

    # 2) 모델 학습
    print("\n[2단계] 모델 학습")
    pipe = train_model(pool)

    # 3) GUI 초기화
    print("\n[3단계] GUI 시작")
    print("  (GUI 창이 열립니다. 콘솔에서도 예측 결과를 확인하세요.)")
    print(f"\n  스트리밍 간격: {STREAM_INTERVAL}초/윈도우")
    print(f"  낙상 확정 기준: 연속 {FALL_CONFIRM}회 예측")
    print(f"  경보 해제 기준: 연속 {NORMAL_CONFIRM}회 정상")
    print(SEP)

    app = FallAlertTimer()

    # 4) 스트리밍 스레드 (GUI 초기화 후 2초 뒤 시작)
    def start_stream():
        time.sleep(2.0)
        stream(app, pipe, pool)

    threading.Thread(target=start_stream, daemon=True).start()

    # 5) GUI 실행 (메인 스레드 블로킹)
    app.run()


if __name__ == "__main__":
    main()
