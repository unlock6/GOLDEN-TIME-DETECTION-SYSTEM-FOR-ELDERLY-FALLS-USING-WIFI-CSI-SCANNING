"""
학습/실시간 결과 시각화 유틸리티.
"""

from __future__ import annotations

from typing import Dict, List


def plot_training_history(history):
    """학습 이력 그래프를 저장한다."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[경고] matplotlib 미설치: 학습 그래프 생성을 건너뜁니다.")
        return

    history_dict = history.history if hasattr(history, "history") else {}
    if not history_dict:
        print("[시각화] 표시할 학습 이력이 없습니다.")
        return

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history_dict.get("accuracy", []), label="train_acc")
    plt.plot(history_dict.get("val_accuracy", []), label="val_acc")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_dict.get("loss", []), label="train_loss")
    plt.plot(history_dict.get("val_loss", []), label="val_loss")
    plt.title("Loss")
    plt.legend()

    plt.tight_layout()
    output_path = "artifacts/training_history.png"
    plt.savefig(output_path)
    print(f"[시각화] 학습 그래프 저장: {output_path}")


def plot_realtime_dashboard(history: Dict[str, List]):
    """실시간 추론 결과를 간단 그래프로 저장한다."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[경고] matplotlib 미설치: 실시간 대시보드 생성을 건너뜁니다.")
        return

    if not history or not history.get("predictions"):
        print("[시각화] 실시간 결과가 없어 대시보드를 건너뜁니다.")
        return

    plt.figure(figsize=(9, 4))
    plt.plot(history["predictions"], marker="o")
    plt.title("Realtime Predicted Class Index")
    plt.xlabel("Step")
    plt.ylabel("Predicted Class")
    plt.grid(True, alpha=0.3)
    output_path = "artifacts/realtime_predictions.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[시각화] 실시간 대시보드 저장: {output_path}")
