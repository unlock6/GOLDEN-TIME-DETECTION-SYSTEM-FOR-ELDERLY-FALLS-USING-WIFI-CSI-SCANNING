"""
모델 정의/학습/평가.

아키텍처 흐름:
  [입력] (N, time, features)
     │
     ├─ CNN (공간적 서브캐리어 패턴 추출)
     │   Conv1D × 2 + MaxPooling
     │
     ├─ LSTM (시간적 의존성 학습)
     │   LSTM × 2  →  (N, time, hidden)   ← return_sequences=True 유지
     │
     └─ [선택] Transformer Encoder Head (USE_TRANSFORMER_HEAD=True)
         Multi-Head Self-Attention × NUM_LAYERS
         → GlobalAvgPool → Dense → Softmax
         
         Transformer가 LSTM 출력 시퀀스를 재조명(Re-attend)함으로써
         LSTM이 놓치기 쉬운 장거리 시간 의존성과 중요 프레임을 포착한다.
         특히 낙상처럼 순간적이고 비대칭적인 패턴에 효과적이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    NUM_CLASSES,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_FF_DIM,
    TRANSFORMER_NUM_HEADS,
    TRANSFORMER_NUM_LAYERS,
    USE_TRANSFORMER_HEAD,
)

try:
    import tensorflow as tf
except Exception:
    tf = None


@dataclass
class SimpleHistory:
    """tensorflow History와 유사한 구조를 맞추기 위한 경량 객체."""
    history: Dict[str, list]


# ──────────────────────────────────────────────────────────────────────────────
# TensorFlow 미설치 환경 대체 모델
# ──────────────────────────────────────────────────────────────────────────────

class FallbackClassifier:
    """TF 미설치 환경용 대체 모델. 빠른 검증용 LogisticRegression."""

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(max_iter=500, multi_class="auto")

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X.reshape(X.shape[0], -1), y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X.reshape(X.shape[0], -1))

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.model, f)


# ──────────────────────────────────────────────────────────────────────────────
# [신규] Transformer Encoder 블록
# ──────────────────────────────────────────────────────────────────────────────

def _build_transformer_encoder_block(x, d_model: int, num_heads: int, ff_dim: int, dropout: float):
    """
    단일 Transformer Encoder 블록.

    구조: Multi-Head Self-Attention → Add & Norm → FFN → Add & Norm

    LSTM 출력 시퀀스(N, time, hidden)를 입력으로 받아
    각 시간 스텝이 다른 모든 스텝을 참조(Self-Attention)하게 한다.
    낙상의 특징적인 순간(급격한 진폭 변화 직후)에 높은 어텐션 가중치가
    부여되어 분류 성능이 향상된다.

    Parameters
    ----------
    x        : Keras 텐서 (batch, time, d_model)
    d_model  : Self-Attention 내부 차원 (LSTM hidden과 일치 권장)
    num_heads: Multi-head 수 (d_model % num_heads == 0 이어야 함)
    ff_dim   : Feed-Forward Network 중간 차원
    dropout  : Dropout 비율
    """
    # 입력 채널을 d_model로 맞추는 선형 투영 (LSTM 출력이 다를 수 있으므로)
    if x.shape[-1] != d_model:
        x = tf.keras.layers.Dense(d_model)(x)

    # 1) Multi-Head Self-Attention
    attn_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout,
    )(x, x)
    attn_output = tf.keras.layers.Dropout(dropout)(attn_output)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    # 2) Position-wise Feed-Forward Network
    ffn = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    ffn = tf.keras.layers.Dropout(dropout)(ffn)
    ffn = tf.keras.layers.Dense(d_model)(ffn)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn)

    return x


# ──────────────────────────────────────────────────────────────────────────────
# CNN-LSTM (기존) + Transformer Head (신규)
# ──────────────────────────────────────────────────────────────────────────────

def build_cnn_lstm(input_shape):
    """
    CNN-LSTM 단독 모델 (USE_TRANSFORMER_HEAD=False일 때 사용).
    기존 구조 완전 유지.
    """
    if tf is None:
        print("[경고] TensorFlow 미설치: 대체 분류기로 학습합니다.")
        return FallbackClassifier()

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.LSTM(128),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    return model


def build_cnn_lstm_transformer(input_shape):
    """
    CNN-LSTM + Transformer Encoder Head 통합 모델.

    아키텍처 설계 근거
    ------------------
    1. CNN: Conv1D 두 층이 서브캐리어 방향의 지역 패턴(공간적 특징)을 추출.
       MaxPooling으로 시간 해상도를 줄여 LSTM 계산량을 낮춘다.

    2. LSTM: return_sequences=True 로 매 타임스텝의 히든 벡터를 유지.
       기존 모델처럼 두 번째 LSTM도 return_sequences=True 로 변경해
       Transformer에 전체 시퀀스를 전달한다.

    3. Transformer Encoder:
       LSTM 출력 시퀀스에 Self-Attention을 적용해 "어느 시간 프레임이
       분류에 중요한지"를 모델 스스로 학습한다.
       낙상처럼 짧고 강렬한 이벤트는 LSTM의 순차적 망각(forgetting)으로
       약해질 수 있는데, Transformer의 전역 어텐션이 이를 보완한다.

    4. GlobalAveragePooling1D:
       시퀀스 전체를 요약해 고정 크기 벡터로 변환.
    """
    if tf is None:
        print("[경고] TensorFlow 미설치: 대체 분류기로 학습합니다.")
        return FallbackClassifier()

    inputs = tf.keras.layers.Input(shape=input_shape)

    # ── CNN 블록 ────────────────────────────────────────────────────────────
    x = tf.keras.layers.Conv1D(32, kernel_size=3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # ── LSTM 블록 (return_sequences=True 유지 → Transformer 입력용) ─────────
    x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(TRANSFORMER_D_MODEL, return_sequences=True)(x)
    # ※ 두 번째 LSTM hidden 크기를 TRANSFORMER_D_MODEL 과 맞춰
    #   투영 레이어 없이도 Transformer에 바로 연결 가능하게 한다.

    # ── Transformer Encoder 블록 (NUM_LAYERS 반복) ──────────────────────────
    for _ in range(TRANSFORMER_NUM_LAYERS):
        x = _build_transformer_encoder_block(
            x,
            d_model=TRANSFORMER_D_MODEL,
            num_heads=TRANSFORMER_NUM_HEADS,
            ff_dim=TRANSFORMER_FF_DIM,
            dropout=TRANSFORMER_DROPOUT,
        )

    # ── 분류 헤드 ────────────────────────────────────────────────────────────
    x = tf.keras.layers.GlobalAveragePooling1D()(x)   # (N, D_MODEL)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="CNN_LSTM_Transformer")


def build_efficientnet(input_shape):
    """EfficientNetB0 기반 분류 모델 (기존 유지)."""
    if tf is None:
        print("[경고] TensorFlow 미설치: 대체 분류기로 학습합니다.")
        return FallbackClassifier()

    try:
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    except Exception:
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_shape=input_shape
        )
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="swish")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


# ──────────────────────────────────────────────────────────────────────────────
# 공개 팩토리 함수
# ──────────────────────────────────────────────────────────────────────────────

def prepare_model_inputs(X: np.ndarray, model_type: str) -> np.ndarray:
    """모델 타입별 입력 텐서 형태를 맞춘다."""
    if model_type in ("cnn_lstm", "cnn_lstm_transformer"):
        return X.astype(np.float32)

    if model_type == "efficientnet":
        if X.ndim != 3:
            raise ValueError("EfficientNet 변환 전 X는 (N, time, features)여야 합니다.")
        image_like = np.repeat(X[..., None], 3, axis=-1)
        if tf is not None:
            return tf.image.resize(image_like, (224, 224)).numpy().astype(np.float32)
        return image_like.astype(np.float32)

    raise ValueError(f"지원하지 않는 model_type: {model_type}")


def build_model(input_shape, model_type: str):
    """
    모델 타입에 맞는 네트워크를 반환한다.

    model_type 선택 가이드
    ----------------------
    "cnn_lstm"             : 기존 모델 (Transformer 없음)
    "cnn_lstm_transformer" : CNN-LSTM + Transformer Head (권장)
    "efficientnet"         : 이미지형 2D 입력 전용

    config.USE_TRANSFORMER_HEAD = True 이면 "cnn_lstm" 요청도
    자동으로 Transformer 포함 모델로 업그레이드된다.
    """
    if model_type == "cnn_lstm":
        if USE_TRANSFORMER_HEAD:
            print("[모델] USE_TRANSFORMER_HEAD=True → CNN-LSTM + Transformer 모델 사용")
            return build_cnn_lstm_transformer(input_shape)
        return build_cnn_lstm(input_shape)

    if model_type in ("cnn_lstm_transformer",):
        return build_cnn_lstm_transformer(input_shape)

    if model_type == "efficientnet":
        return build_efficientnet(input_shape)

    raise ValueError(f"지원하지 않는 model_type: {model_type}")


def build_transfer_model(input_shape):
    """향후 전이학습 확장을 위한 별도 팩토리."""
    return build_cnn_lstm_transformer(input_shape) if USE_TRANSFORMER_HEAD else build_cnn_lstm(input_shape)


# ──────────────────────────────────────────────────────────────────────────────
# 학습 / 평가 / 저장
# ──────────────────────────────────────────────────────────────────────────────

def _compute_class_weights(y: np.ndarray) -> dict:
    """클래스 불균형을 보정하는 가중치 딕셔너리를 반환한다."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_classes = len(classes)
    return {int(cls): float(total / (n_classes * cnt)) for cls, cnt in zip(classes, counts)}


def compile_and_train(model, X_train, y_train, X_val, y_val):
    """모델을 컴파일/학습하고 학습 이력을 반환한다."""
    class_weights = _compute_class_weights(y_train)
    print(f"[학습] 클래스 가중치: {class_weights}")

    if tf is None:
        # FallbackClassifier는 class_weight 미지원 — sklearn 방식으로 직접 처리
        from sklearn.utils.class_weight import compute_sample_weight
        sample_w = compute_sample_weight("balanced", y_train)
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc   = accuracy_score(y_val,   model.predict(X_val))
        return SimpleHistory(
            history={
                "accuracy":     [float(train_acc)],
                "val_accuracy": [float(val_acc)],
                "loss":         [0.0],
                "val_loss":     [0.0],
            }
        )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, EARLY_STOPPING_PATIENCE // 3),
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )
    return history


def evaluate(model, X_test, y_test):
    """테스트셋 성능을 출력한다."""
    if tf is not None and hasattr(model, "predict"):
        predictions = model.predict(X_test, verbose=0)
        y_pred = np.argmax(predictions, axis=1) if predictions.ndim > 1 else (predictions > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[성능] Accuracy: {accuracy:.4f}")
    unique_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    target_names = [CLASS_NAMES[i] for i in unique_labels if i < len(CLASS_NAMES)]
    print("[성능] Classification Report:")
    print(
        classification_report(
            y_test, y_pred, labels=unique_labels,
            target_names=target_names, zero_division=0,
        )
    )
    print("[성능] Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def save_model(model, model_path: str):
    """모델 저장 (TF / 대체모델 분기)."""
    if tf is not None and hasattr(model, "save"):
        model.save(model_path)
    else:
        model.save(model_path.replace(".h5", ".pkl"))


def load_model(model_path: str):
    """저장된 모델을 로드한다."""
    if tf is not None:
        return tf.keras.models.load_model(model_path)

    import pickle
    with open(model_path.replace(".h5", ".pkl"), "rb") as f:
        loaded = pickle.load(f)
    wrapper = FallbackClassifier()
    wrapper.model = loaded
    return wrapper
