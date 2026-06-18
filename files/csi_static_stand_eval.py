# -*- coding: utf-8 -*-
"""
네 개의 CSI CSV(static / stand, RX1·RX2)를 읽어 이진 분류 정확도를 측정한다.
raw_data에서 대괄호 CSI 진폭 배열을 추출해 특징 벡터로 사용한다.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# CSI 배열: raw_data 문자열 끝부분의 [...] 블록을 찾는다
_CSI_ARRAY_PATTERN = re.compile(r"\[([0-9,\s\-]+)\]\s*(?:\"\"\"|\")?\s*$")


def extract_csi_amplitudes(raw_data: str) -> np.ndarray | None:
    """raw_data에서 마지막 대괄호 리스트(진폭)를 파싱한다. 실패 시 None."""
    if not isinstance(raw_data, str) or not raw_data.strip():
        return None
    match = _CSI_ARRAY_PATTERN.search(raw_data.strip())
    if not match:
        return None
    try:
        values = ast.literal_eval("[" + match.group(1) + "]")
        return np.asarray(values, dtype=np.float64)
    except (SyntaxError, ValueError, TypeError):
        return None


def load_labeled_frames(csv_paths: list[Path], encoding: str = "cp949") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """여러 CSV를 합쳐 (X, y, 출처_태그) 반환. y는 0=static, 1=stand."""
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    sources: list[str] = []

    for path in csv_paths:
        df = pd.read_csv(path, encoding=encoding)
        if "label" not in df.columns or "raw_data" not in df.columns:
            raise ValueError(f"필수 컬럼 없음: {path}")

        tag = f"{path.stem}"
        for _, row in df.iterrows():
            label_str = str(row["label"]).strip().lower()
            if label_str == "static":
                y_val = 0
            elif label_str == "stand":
                y_val = 1
            else:
                continue

            vec = extract_csi_amplitudes(str(row["raw_data"]))
            if vec is None or vec.size == 0:
                continue

            feature_rows.append(vec)
            labels.append(y_val)
            sources.append(tag)

    if not feature_rows:
        raise RuntimeError("유효한 샘플이 없습니다.")

    lengths = {r.size for r in feature_rows}
    if len(lengths) != 1:
        raise RuntimeError(f"CSI 벡터 길이가 일치하지 않음: {lengths}")

    X = np.vstack(feature_rows)
    y = np.asarray(labels, dtype=np.int64)
    return X, y, sources


def main() -> None:
    base = Path(r"c:\Users\주현준\OneDrive\Desktop\새 폴더")
    files = [
        base / "csi_rx1_static_20260513_230103.csv",
        base / "csi_rx1_stand_20260513_230539.csv",
        base / "csi_rx2_static_20260513_230103.csv",
        base / "csi_rx2_stand_20260513_230539.csv",
    ]
    for f in files:
        if not f.is_file():
            raise FileNotFoundError(f"파일 없음: {f}")

    X, y, sources = load_labeled_frames(files)
    print(f"총 샘플 수: {X.shape[0]}, 특징 차원: {X.shape[1]}")
    print(f"static(0): {(y == 0).sum()}, stand(1): {(y == 1).sum()}")

    # 1) 홀드아웃: 임의 분할(고정 시드) — 사용자가 말한 '임의의 데이터'에 대응
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # 로지스틱 + 스케일링 (고차원·상관 특징에도 비교적 안정)
    pipe_lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
            ),
        ]
    )
    pipe_lr.fit(X_train, y_train)
    y_pred_lr = pipe_lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    # 랜덤 포레스트 (비선형 경계)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    print("\n=== 홀드아웃 (train 80% / test 20%, stratified, seed=42) ===")
    print(f"LogisticRegression + StandardScaler 테스트 정확도: {acc_lr:.4f}")
    print(classification_report(y_test, y_pred_lr, target_names=["static", "stand"], digits=4))
    print(f"RandomForestClassifier 테스트 정확도: {acc_rf:.4f}")
    print(classification_report(y_test, y_pred_rf, target_names=["static", "stand"], digits=4))

    # 2) 5-겹 교차검증 — 분할에 덜 민감한 추정
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    fold_acc_lr: list[float] = []
    fold_acc_rf: list[float] = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        pl = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
                ),
            ]
        )
        pl.fit(X_tr, y_tr)
        fold_acc_lr.append(accuracy_score(y_te, pl.predict(X_te)))

        rfc = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
        rfc.fit(X_tr, y_tr)
        fold_acc_rf.append(accuracy_score(y_te, rfc.predict(X_te)))

    print("\n=== 5-fold 교차검증 (평균 ± 표준편차) ===")
    print(f"LogisticRegression: {np.mean(fold_acc_lr):.4f} ± {np.std(fold_acc_lr):.4f}")
    print(f"RandomForest:       {np.mean(fold_acc_rf):.4f} ± {np.std(fold_acc_rf):.4f}")
    print("각 fold LR:", [f"{a:.4f}" for a in fold_acc_lr])
    print("각 fold RF:", [f"{a:.4f}" for a in fold_acc_rf])


if __name__ == "__main__":
    main()
