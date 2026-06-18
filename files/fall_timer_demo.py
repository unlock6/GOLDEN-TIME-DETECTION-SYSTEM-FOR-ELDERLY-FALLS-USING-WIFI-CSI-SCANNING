"""
낙상 감지 타이머 데모 — 낙상 상태로 바로 시작
"""
import threading
import time
from fall_timer import FallAlertTimer

app = FallAlertTimer()

# 1.5초 뒤 자동으로 낙상 트리거 (정상 → 낙상 전환 애니메이션 확인용)
def _auto_fall():
    time.sleep(1.5)
    app.trigger_fall()

threading.Thread(target=_auto_fall, daemon=True).start()

app.run()
