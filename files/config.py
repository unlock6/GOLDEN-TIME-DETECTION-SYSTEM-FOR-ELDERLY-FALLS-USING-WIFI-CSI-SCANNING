"""
전역 설정 파일.
논문에서 사용한 CSI 기반 낙상 감지 파이프라인 설정값을 모아둔다.
"""

from pathlib import Path

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).resolve().parent

# 데이터/모델 저장 경로
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 실제 수집 데이터 경로 (새 폴더)
RAW_DATA_DIR = Path(r"C:\Users\주현준\OneDrive\Desktop\새 폴더")

# 클래스 정의 — 실제 CSV label 컬럼 값과 일치해야 한다
# - 다중분류: 실제 수집된 4가지 상태
# - 이진분류: 라벨 0=정상(static+stand), 라벨 1=위험(fall+lie_down)
MULTI_CLASS_NAMES = ["fall", "lie_down", "static", "stand"]
BINARY_CLASS_NAMES = ["non_fall", "fall"]

# 이진 분류 시 라벨 1로 매핑할 클래스 (나머지는 자동으로 0)
DANGER_CLASS_NAMES = {"fall", "lie_down"}
USE_BINARY_CLASSIFICATION = True
CLASS_NAMES = BINARY_CLASS_NAMES if USE_BINARY_CLASSIFICATION else MULTI_CLASS_NAMES

# CSI 서브캐리어 수 (I/Q 128개 → 진폭 64개)
N_SUBCARRIERS = 64
NUM_CLASSES = len(CLASS_NAMES)

# 시간축 윈도우 설정 (논문: 약 100 프레임)
WINDOW_SAMPLES = 100
WINDOW_STRIDE = 20

# 다운샘플링 설정 (5초 대역 원신호에서 경량화할 때 사용)
DOWNSAMPLE_FACTOR = 8
MIN_LENGTH_FOR_DOWNSAMPLE = 500

# 전처리 파라미터 (논문 값 반영)
HAMPEL_WINDOW = 5
HAMPEL_SIGMA = 3.0
SG_WINDOW = 7
SG_POLY_ORDER = 3
PCA_COMPONENTS = 10

# ── 시간차분 필터 설정 ─────────────────────────────────────────────────────────
# DIFF_ORDER     : 1차 차분(속도) 또는 2차 차분(가속도)
# DIFF_STACK_MODE: "replace" → 차분 결과만 사용
#                  "concat"  → 원신호 + 차분 결과 채널 방향으로 이어붙임 (권장)
DIFF_ORDER = 1
DIFF_STACK_MODE = "concat"   # "replace" | "concat"

# 전처리 모드
# - "hybrid": Hampel + SG + MinMax + PCA (2025 논문 흐름 반영)
# - "lite":   Downsample + MinMax 중심 (2023 논문의 경량 전처리 반영)
PREPROCESS_MODE = "hybrid"

# 학습 파라미터 (논문 설정 근사)
TEST_SIZE = 0.2
RANDOM_SEED = 42
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 5e-4
EARLY_STOPPING_PATIENCE = 8

# ── Transformer 튜닝 헤드 설정 ────────────────────────────────────────────────
# CNN-LSTM 출력 위에 Transformer Encoder 블록을 쌓아 최종 분류를 수행한다.
# USE_TRANSFORMER_HEAD = True 로 설정하면 build_model()이 해당 아키텍처를 반환.
USE_TRANSFORMER_HEAD = True   # False → 기존 CNN-LSTM 유지
TRANSFORMER_D_MODEL = 128     # Self-Attention 내부 차원 (LSTM hidden 크기와 맞춤)
TRANSFORMER_NUM_HEADS = 4     # Multi-head attention 헤드 수
TRANSFORMER_FF_DIM = 256      # Feed-Forward 중간 차원
TRANSFORMER_NUM_LAYERS = 2    # Transformer Encoder 블록 반복 수
TRANSFORMER_DROPOUT = 0.1     # Transformer 내부 Dropout

# 실시간 추론 관련 기본값
SERIAL_PORT = "COM3"
REALTIME_INTERVAL_SEC = 0.5
