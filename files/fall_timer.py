"""
낙상 감지 경보 시스템 v2.0 — 전체 기능

기능 목록:
  - 실시간 경과 시간 (HH:MM:SS)
  - 3단계 경보 에스컬레이션 (주의 → 위험 → 위급)
  - 응답 없음 반복 경보
  - 음성 TTS 안내
  - 낙상 이력 CSV 로그
  - 이메일 자동 발송
  - 카카오톡 알림
  - 오늘 통계 요약
  - 119 안내 팝업
  - 실시간 CSI 신호 그래프
  - 낙상 확률(신뢰도) 표시
"""

from __future__ import annotations

import csv
import json
import smtplib
import subprocess
import threading
import time
import winsound
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import random

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# ══════════════════════════════════════════════════════════════════════════════
# 상수 / 색상
# ══════════════════════════════════════════════════════════════════════════════

LOG_DIR     = Path(__file__).parent / "artifacts" / "fall_logs"
CONFIG_PATH = Path(__file__).parent / "artifacts" / "fall_timer_config.json"
LOG_DIR.mkdir(parents=True, exist_ok=True)

CAUTION_MIN     = 5    # 주의 단계 진입 (분)
DANGER_MIN      = 10   # 위험 단계 진입 (분)
NO_RESPONSE_MIN = 3    # 응답 없음 반복 경보 간격 (분)

PALETTE = {
    "normal":   {"bg": "#1a1a2e", "panel": "#16213e", "accent": "#2ecc71",
                 "fg": "#ecf0f1", "timer": "#2ecc71",  "btn": "#0f3460"},
    "caution":  {"bg": "#1c1700", "panel": "#2d2500", "accent": "#f1c40f",
                 "fg": "#fff3b0", "timer": "#f39c12",  "btn": "#5d4e00"},
    "danger":   {"bg": "#200a00", "panel": "#3d1500", "accent": "#e67e22",
                 "fg": "#ffd6a5", "timer": "#e67e22",  "btn": "#7d3000"},
    "critical": {"bg": "#200000", "panel": "#3d0000", "accent": "#e74c3c",
                 "fg": "#ffb3b3", "timer": "#ff2222",  "btn": "#7d0000"},
}

LEVEL_LABEL = {
    "normal":   "정상  -  이상 없음",
    "caution":  "낙상 감지!  [주의]",
    "danger":   "낙상 감지!  [위험]",
    "critical": "위급!  즉시 119 신고!",
}


# ══════════════════════════════════════════════════════════════════════════════
# FallLogger  —  CSV 이력 기록
# ══════════════════════════════════════════════════════════════════════════════

class FallLogger:
    def __init__(self):
        today = datetime.now().strftime("%Y%m%d")
        self.path = LOG_DIR / f"fall_log_{today}.csv"
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(
                    ["감지 시각", "안전 확인 시각", "경과(초)", "최고 경보 단계", "낙상 확률(%)"])

    def log_fall(self, fall_time: datetime, safe_time: datetime | None,
                 elapsed_sec: int, max_level: str, confidence: float):
        row = [
            fall_time.strftime("%Y-%m-%d %H:%M:%S"),
            safe_time.strftime("%H:%M:%S") if safe_time else "미확인",
            elapsed_sec,
            max_level,
            f"{confidence * 100:.1f}",
        ]
        with open(self.path, "a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(row)

    def get_all(self) -> list[list]:
        rows = []
        for p in sorted(LOG_DIR.glob("fall_log_*.csv")):
            with open(p, encoding="utf-8-sig") as f:
                rows.extend(list(csv.reader(f))[1:])
        return rows

    def get_today_stats(self) -> dict:
        today = datetime.now().strftime("%Y-%m-%d")
        rows  = [r for r in self.get_all() if r and r[0].startswith(today)]
        total     = len(rows)
        confirmed = sum(1 for r in rows if len(r) > 1 and r[1] != "미확인")
        elaps     = [int(r[2]) for r in rows if len(r) > 2 and r[2].isdigit()]
        max_e     = max(elaps) if elaps else 0
        avg_e     = int(sum(elaps) / len(elaps)) if elaps else 0
        return {"total": total, "confirmed": confirmed,
                "max_elapsed": max_e, "avg_elapsed": avg_e}


# ══════════════════════════════════════════════════════════════════════════════
# TTS 음성 안내
# ══════════════════════════════════════════════════════════════════════════════

class TTSEngine:
    def speak(self, text: str):
        threading.Thread(target=self._run, args=(text,), daemon=True).start()

    def _run(self, text: str):
        try:
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty("rate", 165)
            eng.say(text)
            eng.runAndWait()
        except Exception:
            try:
                safe = text.replace("'", "")
                cmd = (
                    f"Add-Type -AssemblyName System.Speech; "
                    f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f"$s.Speak('{safe}')"
                )
                subprocess.run(["powershell", "-Command", cmd],
                               capture_output=True, timeout=10)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# AlertNotifier  —  이메일 / 카카오톡
# ══════════════════════════════════════════════════════════════════════════════

class AlertNotifier:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def send_email(self, fall_time: datetime, elapsed_sec: int, confidence: float):
        c = self.cfg.get("email", {})
        if not (c.get("enabled") and c.get("sender") and
                c.get("password") and c.get("receiver")):
            return False
        try:
            msg = MIMEMultipart()
            msg["From"]    = c["sender"]
            msg["To"]      = c["receiver"]
            msg["Subject"] = f"[낙상 경보] {fall_time.strftime('%H:%M:%S')} 감지"
            body = (
                f"[낙상 감지 알림]\n\n"
                f"감지 시각 : {fall_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"경과 시간 : {elapsed_sec // 60}분 {elapsed_sec % 60}초\n"
                f"낙상 확률 : {confidence * 100:.1f}%\n\n"
                f"즉시 확인하시기 바랍니다."
            )
            msg.attach(MIMEText(body, "plain", "utf-8"))
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(c["sender"], c["password"])
                smtp.send_message(msg)
            return True
        except Exception as e:
            print(f"[이메일 오류] {e}")
            return False

    def send_kakao(self, fall_time: datetime, elapsed_sec: int, confidence: float):
        if not HAS_REQUESTS:
            return False
        c = self.cfg.get("kakao", {})
        if not (c.get("enabled") and c.get("access_token")):
            return False
        try:
            text = (
                f"[낙상 경보]\n"
                f"감지 시각: {fall_time.strftime('%H:%M:%S')}\n"
                f"경과: {elapsed_sec // 60}분 {elapsed_sec % 60}초\n"
                f"확률: {confidence * 100:.1f}%\n즉시 확인 바랍니다!"
            )
            url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
            headers = {"Authorization": f"Bearer {c['access_token']}"}
            data = {"template_object": json.dumps({
                "object_type": "text", "text": text,
                "link": {"web_url": "", "mobile_web_url": ""}
            })}
            r = requests.post(url, headers=headers, data=data, timeout=5)
            return r.status_code == 200
        except Exception as e:
            print(f"[카카오 오류] {e}")
            return False

    def notify_all(self, fall_time: datetime, elapsed_sec: int, confidence: float):
        threading.Thread(target=self.send_email,
                         args=(fall_time, elapsed_sec, confidence), daemon=True).start()
        threading.Thread(target=self.send_kakao,
                         args=(fall_time, elapsed_sec, confidence), daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# FallAlertTimer  —  메인 GUI
# ══════════════════════════════════════════════════════════════════════════════

class FallAlertTimer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("낙상 감지 경보 시스템 v2.0")
        self.root.geometry("820x680")
        self.root.resizable(True, True)

        # 상태
        self.fall_detected          = False
        self.fall_time: datetime | None = None
        self.max_level              = "normal"
        self.confidence             = 0.0
        self._blink_on              = True
        self._last_no_resp: datetime | None = None

        # 서브시스템
        self.logger   = FallLogger()
        self.tts      = TTSEngine()
        self.cfg      = self._load_config()
        self.notifier = AlertNotifier(self.cfg)

        # CSI 신호 버퍼
        self._sig_buf = [12.0] * 100

        self._build_ui()

    # ── 설정 파일 ────────────────────────────────────────────────────────────

    def _load_config(self) -> dict:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, encoding="utf-8") as f:
                return json.load(f)
        return {
            "email":  {"enabled": False, "sender": "", "password": "", "receiver": ""},
            "kakao":  {"enabled": False, "access_token": ""},
            "alerts": {"tts": True, "sound": True,
                       "no_response_min": NO_RESPONSE_MIN},
        }

    def _save_config(self):
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, ensure_ascii=False, indent=2)

    # ── UI 구성 ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        pal = PALETTE["normal"]
        self.root.configure(bg=pal["bg"])

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=6, pady=6)

        self._tab_alarm    = tk.Frame(self.nb, bg=pal["bg"])
        self._tab_history  = tk.Frame(self.nb, bg="#1a1a2e")
        self._tab_stats    = tk.Frame(self.nb, bg="#1a1a2e")
        self._tab_settings = tk.Frame(self.nb, bg="#1a1a2e")

        self.nb.add(self._tab_alarm,    text="  경보  ")
        self.nb.add(self._tab_history,  text="  이력  ")
        self.nb.add(self._tab_stats,    text="  통계  ")
        self.nb.add(self._tab_settings, text="  설정  ")

        self._build_alarm_tab()
        self._build_history_tab()
        self._build_stats_tab()
        self._build_settings_tab()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tick()

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 1 : 경보
    # ─────────────────────────────────────────────────────────────────────────

    def _build_alarm_tab(self):
        f   = self._tab_alarm
        pal = PALETTE["normal"]

        # 상단 바 (현재 시각 + 낙상 확률)
        top = tk.Frame(f, bg=pal["panel"])
        top.pack(fill="x", padx=8, pady=(8, 4))

        self.lbl_clock = tk.Label(
            top, text="00 : 00 : 00",
            font=("Consolas", 22, "bold"),
            bg=pal["panel"], fg=pal["fg"])
        self.lbl_clock.pack(side="left", padx=16, pady=6)

        rt = tk.Frame(top, bg=pal["panel"])
        rt.pack(side="right", padx=16)
        tk.Label(rt, text="낙상 확률", font=("Malgun Gothic", 9),
                 bg=pal["panel"], fg="#888").pack()
        self.lbl_conf = tk.Label(
            rt, text="-- %",
            font=("Consolas", 17, "bold"),
            bg=pal["panel"], fg=pal["accent"])
        self.lbl_conf.pack()

        # 상태 패널
        self.status_frame = tk.Frame(f, bg=pal["panel"], bd=2, relief="ridge")
        self.status_frame.pack(fill="x", padx=8, pady=4)
        self.lbl_status = tk.Label(
            self.status_frame, text=LEVEL_LABEL["normal"],
            font=("Malgun Gothic", 17, "bold"),
            bg=pal["panel"], fg=pal["accent"], pady=12)
        self.lbl_status.pack(fill="x")

        # 경과 시간
        t_outer = tk.Frame(f, bg=pal["bg"])
        t_outer.pack(fill="x", padx=8, pady=4)
        tk.Label(t_outer, text="낙상 감지 후 경과 시간",
                 font=("Malgun Gothic", 10), bg=pal["bg"], fg="#888").pack()

        t_box = tk.Frame(t_outer, bg=pal["panel"], bd=3, relief="groove")
        t_box.pack(pady=4)
        self.lbl_timer = tk.Label(
            t_box, text="00 : 00 : 00",
            font=("Consolas", 50, "bold"),
            bg=pal["panel"], fg=pal["timer"], padx=28, pady=8)
        self.lbl_timer.pack()

        # 감지 정보 행
        info = tk.Frame(f, bg=pal["bg"])
        info.pack(fill="x", padx=12, pady=2)
        self.lbl_fall_time = tk.Label(
            info, text="낙상 감지 시각: --:--:--",
            font=("Malgun Gothic", 10), bg=pal["bg"], fg="#ccc")
        self.lbl_fall_time.pack(side="left")
        self.lbl_level = tk.Label(
            info, text="경보 단계: 정상",
            font=("Malgun Gothic", 10, "bold"),
            bg=pal["bg"], fg=pal["accent"])
        self.lbl_level.pack(side="right")

        # CSI 신호 그래프
        g_frame = tk.Frame(f, bg=pal["bg"])
        g_frame.pack(fill="x", padx=8, pady=(2, 0))
        tk.Label(g_frame, text="실시간 CSI 신호",
                 font=("Malgun Gothic", 8), bg=pal["bg"], fg="#666").pack(anchor="w")
        self.sig_canvas = tk.Canvas(
            g_frame, height=65, bg="#0d0d1a",
            highlightthickness=1, highlightbackground="#2a2a3e")
        self.sig_canvas.pack(fill="x")

        # 오늘 카운트
        self.lbl_count = tk.Label(
            f, text="오늘 낙상 감지: 0 회",
            font=("Malgun Gothic", 9), bg=pal["bg"], fg="#888")
        self.lbl_count.pack(pady=2)

        # 버튼 영역
        btn_f = tk.Frame(f, bg=pal["bg"])
        btn_f.pack(fill="x", padx=8, pady=6)

        self.btn_safe = tk.Button(
            btn_f, text="✔  안전 확인  (타이머 초기화)",
            font=("Malgun Gothic", 12, "bold"),
            bg="#27ae60", fg="white", relief="flat",
            activebackground="#1e8449", cursor="hand2",
            command=self.reset, pady=9)
        self.btn_safe.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.btn_test = tk.Button(
            btn_f, text="테스트: 낙상",
            font=("Malgun Gothic", 10),
            bg="#e67e22", fg="white", relief="flat",
            activebackground="#ca6f1e", cursor="hand2",
            command=lambda: self.trigger_fall(confidence=0.97), pady=9)
        self.btn_test.pack(side="left", padx=(0, 4))

        self.btn_119 = tk.Button(
            btn_f, text="119 안내",
            font=("Malgun Gothic", 10, "bold"),
            bg="#c0392b", fg="white", relief="flat",
            activebackground="#a93226", cursor="hand2",
            command=self._show_119_popup, pady=9)
        self.btn_119.pack(side="left")

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 2 : 이력
    # ─────────────────────────────────────────────────────────────────────────

    def _build_history_tab(self):
        f = self._tab_history
        tk.Label(f, text="낙상 이력 (전체)",
                 font=("Malgun Gothic", 12, "bold"),
                 bg="#1a1a2e", fg="#ecf0f1").pack(pady=(10, 4))

        cols = ("감지 시각", "안전 확인", "경과(초)", "최고 단계", "확률(%)")
        self.tree = ttk.Treeview(f, columns=cols, show="headings", height=17)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=140, anchor="center")

        sb = ttk.Scrollbar(f, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True,
                       padx=(8, 0), pady=8)
        sb.pack(side="left", fill="y", pady=8)

        bp = tk.Frame(f, bg="#1a1a2e")
        bp.pack(side="right", padx=8, pady=8, anchor="n")
        tk.Button(bp, text="새로고침", bg="#2c3e50", fg="white",
                  command=self._refresh_history,
                  pady=6, width=10).pack(pady=4)
        tk.Button(bp, text="폴더 열기", bg="#2c3e50", fg="white",
                  command=lambda: __import__("os").startfile(str(LOG_DIR)),
                  pady=6, width=10).pack(pady=4)

        self._refresh_history()

    def _refresh_history(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for row in self.logger.get_all():
            self.tree.insert("", "end", values=row)

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 3 : 통계
    # ─────────────────────────────────────────────────────────────────────────

    def _build_stats_tab(self):
        f = self._tab_stats
        tk.Label(f, text="오늘 통계",
                 font=("Malgun Gothic", 12, "bold"),
                 bg="#1a1a2e", fg="#ecf0f1").pack(pady=(10, 4))

        grid = tk.Frame(f, bg="#1a1a2e")
        grid.pack(fill="x", padx=24)

        self._stat_labels: dict[str, tuple] = {}
        items = [
            ("총 낙상 감지",    "total",        "회"),
            ("안전 확인 완료",  "confirmed",     "회"),
            ("최장 경과 시간",  "max_elapsed",   "초"),
            ("평균 경과 시간",  "avg_elapsed",   "초"),
        ]
        for i, (name, key, unit) in enumerate(items):
            tk.Label(grid, text=name,
                     font=("Malgun Gothic", 11), bg="#1a1a2e", fg="#aaa",
                     width=16, anchor="w").grid(row=i, column=0, pady=8, sticky="w")
            lbl = tk.Label(grid, text=f"0 {unit}",
                           font=("Consolas", 15, "bold"),
                           bg="#1a1a2e", fg="#2ecc71", width=12, anchor="w")
            lbl.grid(row=i, column=1, pady=8, padx=20, sticky="w")
            self._stat_labels[key] = (lbl, unit)

        tk.Button(f, text="새로고침", bg="#2c3e50", fg="white",
                  font=("Malgun Gothic", 10),
                  command=self._refresh_stats,
                  pady=6).pack(pady=8)

        # 24시간 막대 그래프
        if HAS_MPL:
            tk.Label(f, text="24시간 낙상 감지 이력",
                     font=("Malgun Gothic", 9), bg="#1a1a2e", fg="#666").pack()
            fig = Figure(figsize=(6.5, 2.2), facecolor="#1a1a2e")
            ax  = fig.add_subplot(111, facecolor="#16213e")
            hours  = list(range(24))
            counts = [random.randint(0, 2) if random.random() < 0.15 else 0
                      for _ in hours]
            ax.bar(hours, counts, color="#e74c3c", width=0.7)
            ax.set_xlabel("시간(시)", color="#888", fontsize=7)
            ax.set_ylabel("감지 횟수", color="#888", fontsize=7)
            ax.tick_params(colors="#888", labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("#333")
            fig.tight_layout(pad=0.8)
            FigureCanvasTkAgg(fig, master=f).get_tk_widget().pack(
                fill="x", padx=12, pady=4)

        self._refresh_stats()

    def _refresh_stats(self):
        s = self.logger.get_today_stats()
        for key, (lbl, unit) in self._stat_labels.items():
            lbl.config(text=f"{s[key]} {unit}")

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 4 : 설정
    # ─────────────────────────────────────────────────────────────────────────

    def _build_settings_tab(self):
        f = self._tab_settings

        def section(title):
            tk.Label(f, text=title,
                     font=("Malgun Gothic", 11, "bold"),
                     bg="#1a1a2e", fg="#3498db").pack(anchor="w", padx=12, pady=(12, 2))
            tk.Frame(f, bg="#3498db", height=1).pack(fill="x", padx=12)

        def entry_row(parent, label, var, show=""):
            r = tk.Frame(parent, bg="#16213e")
            r.pack(fill="x", padx=8, pady=2)
            tk.Label(r, text=label, width=18, anchor="w",
                     bg="#16213e", fg="#ccc",
                     font=("Malgun Gothic", 9)).pack(side="left", padx=8)
            tk.Entry(r, textvariable=var, show=show, width=34,
                     bg="#0d0d1a", fg="white",
                     insertbackground="white",
                     font=("Consolas", 9)).pack(side="left", pady=3)

        # 이메일
        section("이메일 알림 (Gmail)")
        self._v_email_on = tk.BooleanVar(value=self.cfg["email"]["enabled"])
        self._v_sender   = tk.StringVar(value=self.cfg["email"]["sender"])
        self._v_password = tk.StringVar(value=self.cfg["email"]["password"])
        self._v_receiver = tk.StringVar(value=self.cfg["email"]["receiver"])
        ep = tk.Frame(f, bg="#16213e")
        ep.pack(fill="x", padx=8, pady=4)
        tk.Checkbutton(ep, text="이메일 알림 활성화",
                       variable=self._v_email_on,
                       bg="#16213e", fg="#ecf0f1",
                       selectcolor="#0d0d1a",
                       activebackground="#16213e",
                       font=("Malgun Gothic", 9)).pack(anchor="w", padx=8, pady=3)
        entry_row(ep, "Gmail 발신 주소", self._v_sender)
        entry_row(ep, "앱 비밀번호",    self._v_password, show="*")
        entry_row(ep, "수신자 이메일",  self._v_receiver)
        tk.Label(ep, text="※ Gmail 보안 > 2단계 인증 후 앱 비밀번호 발급 필요",
                 font=("Malgun Gothic", 7), bg="#16213e", fg="#666").pack(anchor="w", padx=8)

        # 카카오톡
        section("카카오톡 알림")
        self._v_kakao_on  = tk.BooleanVar(value=self.cfg["kakao"]["enabled"])
        self._v_kakao_tok = tk.StringVar(value=self.cfg["kakao"]["access_token"])
        kp = tk.Frame(f, bg="#16213e")
        kp.pack(fill="x", padx=8, pady=4)
        tk.Checkbutton(kp, text="카카오톡 알림 활성화",
                       variable=self._v_kakao_on,
                       bg="#16213e", fg="#ecf0f1",
                       selectcolor="#0d0d1a",
                       activebackground="#16213e",
                       font=("Malgun Gothic", 9)).pack(anchor="w", padx=8, pady=3)
        entry_row(kp, "Access Token", self._v_kakao_tok, show="*")
        tk.Label(kp, text="※ developers.kakao.com > 내 애플리케이션 > REST API 키",
                 font=("Malgun Gothic", 7), bg="#16213e", fg="#666").pack(anchor="w", padx=8)

        # 알림 옵션
        section("알림 옵션")
        self._v_tts   = tk.BooleanVar(value=self.cfg["alerts"]["tts"])
        self._v_sound = tk.BooleanVar(value=self.cfg["alerts"]["sound"])
        op = tk.Frame(f, bg="#16213e")
        op.pack(fill="x", padx=8, pady=4)
        for text, var in [("음성 TTS 안내", self._v_tts),
                          ("경보음(비프)", self._v_sound)]:
            tk.Checkbutton(op, text=text, variable=var,
                           bg="#16213e", fg="#ecf0f1",
                           selectcolor="#0d0d1a",
                           activebackground="#16213e",
                           font=("Malgun Gothic", 9)).pack(anchor="w", padx=8, pady=2)

        tk.Button(f, text="   설정 저장   ",
                  font=("Malgun Gothic", 11, "bold"),
                  bg="#2980b9", fg="white", relief="flat",
                  command=self._save_settings, pady=7).pack(pady=12)

    def _save_settings(self):
        self.cfg["email"]["enabled"]      = self._v_email_on.get()
        self.cfg["email"]["sender"]       = self._v_sender.get()
        self.cfg["email"]["password"]     = self._v_password.get()
        self.cfg["email"]["receiver"]     = self._v_receiver.get()
        self.cfg["kakao"]["enabled"]      = self._v_kakao_on.get()
        self.cfg["kakao"]["access_token"] = self._v_kakao_tok.get()
        self.cfg["alerts"]["tts"]         = self._v_tts.get()
        self.cfg["alerts"]["sound"]       = self._v_sound.get()
        self.notifier = AlertNotifier(self.cfg)
        self._save_config()
        messagebox.showinfo("저장 완료", "설정이 저장되었습니다.")

    # ─────────────────────────────────────────────────────────────────────────
    # 119 팝업
    # ─────────────────────────────────────────────────────────────────────────

    def _show_119_popup(self):
        win = tk.Toplevel(self.root)
        win.title("긴급 연락처")
        win.geometry("360x300")
        win.configure(bg="#200000")
        win.grab_set()

        tk.Label(win, text="긴급 신고 안내",
                 font=("Malgun Gothic", 15, "bold"),
                 bg="#200000", fg="#ff4444").pack(pady=(18, 6))

        contacts = [
            ("119", "소방서 / 응급 구조"),
            ("112", "경찰서"),
            ("1577-1389", "노인 돌봄 긴급전화"),
            ("1588-3060", "치매 상담 콜센터"),
        ]
        for num, desc in contacts:
            row = tk.Frame(win, bg="#3d0000")
            row.pack(fill="x", padx=20, pady=3)
            tk.Label(row, text=num,
                     font=("Consolas", 15, "bold"),
                     bg="#3d0000", fg="#ff6666", width=12).pack(side="left", padx=8)
            tk.Label(row, text=desc,
                     font=("Malgun Gothic", 9),
                     bg="#3d0000", fg="#ffaaaa").pack(side="left")

        if self.fall_time:
            elapsed = int((datetime.now() - self.fall_time).total_seconds())
            m, s = elapsed // 60, elapsed % 60
            tk.Label(win,
                     text=f"경과: {m}분 {s}초  |  감지: {self.fall_time.strftime('%H:%M:%S')}",
                     font=("Malgun Gothic", 8),
                     bg="#200000", fg="#ff9999").pack(pady=6)

        tk.Button(win, text="닫기", bg="#5d0000", fg="white",
                  font=("Malgun Gothic", 10),
                  command=win.destroy, pady=6, width=12).pack(pady=8)

    # ─────────────────────────────────────────────────────────────────────────
    # 낙상 트리거 / 초기화
    # ─────────────────────────────────────────────────────────────────────────

    def trigger_fall(self, confidence: float = 0.95):
        """낙상 감지 시 호출. 추론 엔진의 콜백으로도 연결 가능."""
        if not self.fall_detected:
            self.fall_detected = True
            self.fall_time     = datetime.now()
            self.confidence    = confidence
            self.max_level     = "caution"
            self._last_no_resp = self.fall_time
            self.root.after(0, self._apply_level, "caution")
            if self.cfg["alerts"]["sound"]:
                threading.Thread(target=self._beep,
                                 args=(3, 880, 350), daemon=True).start()
            if self.cfg["alerts"]["tts"]:
                self.tts.speak("낙상이 감지되었습니다. 즉시 확인하세요.")

    def reset(self):
        """안전 확인 — 이력 저장 후 초기화."""
        if self.fall_detected and self.fall_time:
            elapsed = int((datetime.now() - self.fall_time).total_seconds())
            self.logger.log_fall(
                self.fall_time, datetime.now(),
                elapsed, self.max_level, self.confidence)
            self._refresh_history()
            self._refresh_stats()
        self.fall_detected = False
        self.fall_time     = None
        self.confidence    = 0.0
        self.max_level     = "normal"
        self._last_no_resp = None
        self.root.after(0, self._apply_level, "normal")
        if self.cfg["alerts"]["tts"]:
            self.tts.speak("안전이 확인되었습니다.")

    # ─────────────────────────────────────────────────────────────────────────
    # 경보 단계 적용
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_level(self, level: str):
        pal = PALETTE[level]
        self._tab_alarm.configure(bg=pal["bg"])
        self.status_frame.configure(bg=pal["panel"])
        self.lbl_status.configure(text=LEVEL_LABEL[level],
                                   bg=pal["panel"], fg=pal["accent"])
        self.lbl_timer.configure(bg=pal["panel"], fg=pal["timer"])
        self.lbl_clock.configure(bg=pal["panel"])
        self.lbl_conf.configure(bg=pal["panel"], fg=pal["accent"])
        kr = {"normal": "정상", "caution": "주의",
              "danger": "위험", "critical": "위급"}
        self.lbl_level.configure(text=f"경보 단계: {kr[level]}",
                                  fg=pal["accent"], bg=pal["bg"])
        for w in (self.lbl_fall_time, self.lbl_count, self.btn_safe,
                  self.btn_test, self.btn_119):
            try:
                w.configure(bg=pal["bg"])
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # 메인 틱 (100 ms)
    # ─────────────────────────────────────────────────────────────────────────

    def _tick(self):
        now = datetime.now()
        self.lbl_clock.config(text=now.strftime("%H : %M : %S"))

        if self.fall_detected and self.fall_time:
            elapsed = (now - self.fall_time).total_seconds()
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            self.lbl_timer.config(text=f"{h:02d} : {m:02d} : {s:02d}")
            self.lbl_fall_time.config(
                text=f"낙상 감지 시각: {self.fall_time.strftime('%H:%M:%S')}")
            self.lbl_conf.config(text=f"{self.confidence * 100:.1f} %")

            # 에스컬레이션
            mins = elapsed / 60
            new_level = ("critical" if mins >= DANGER_MIN else
                         "danger"   if mins >= CAUTION_MIN else "caution")
            if new_level != self.max_level:
                self.max_level = new_level
                self._apply_level(new_level)
                self._escalation_alarm(new_level)

            # 깜빡임
            self._blink_on = not self._blink_on
            pal = PALETTE[self.max_level]
            self.lbl_status.config(
                fg=pal["accent"] if self._blink_on else pal["panel"])

            # 응답 없음 반복 경보
            no_resp_sec = self.cfg["alerts"].get(
                "no_response_min", NO_RESPONSE_MIN) * 60
            if (self._last_no_resp and
                    (now - self._last_no_resp).total_seconds() >= no_resp_sec):
                self._last_no_resp = now
                if self.cfg["alerts"]["sound"]:
                    threading.Thread(target=self._beep,
                                     args=(5, 1200, 200), daemon=True).start()
                if self.cfg["alerts"]["tts"]:
                    self.tts.speak("아직 안전 확인이 되지 않았습니다. 즉시 확인하세요.")
        else:
            self.lbl_timer.config(text="00 : 00 : 00")
            self.lbl_fall_time.config(text="낙상 감지 시각: --:--:--")
            self.lbl_conf.config(text="-- %")

        # 오늘 통계
        st = self.logger.get_today_stats()
        self.lbl_count.config(text=f"오늘 낙상 감지: {st['total']} 회")

        # CSI 신호 업데이트
        self._draw_signal()

        self.root.after(100, self._tick)

    def _escalation_alarm(self, level: str):
        info = {
            "danger":   ("위험 단계입니다. 5분 경과. 즉시 확인하세요.", 5, 1000, 300),
            "critical": ("위급 상황. 10분 경과. 즉시 119 신고하세요!", 8, 1400, 200),
        }
        if level not in info:
            return
        msg, n, freq, dur = info[level]
        if self.cfg["alerts"]["sound"]:
            threading.Thread(target=self._beep,
                             args=(n, freq, dur), daemon=True).start()
        if self.cfg["alerts"]["tts"]:
            self.tts.speak(msg)
        if level == "critical" and self.fall_time:
            elapsed = int((datetime.now() - self.fall_time).total_seconds())
            self.notifier.notify_all(self.fall_time, elapsed, self.confidence)
            self.root.after(600, self._show_119_popup)

    # ─────────────────────────────────────────────────────────────────────────
    # CSI 신호 그래프
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_signal(self):
        c = self.sig_canvas
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10:
            return

        new_val = (random.uniform(28, 55) + random.gauss(0, 7)
                   if self.fall_detected
                   else random.uniform(7, 17) + random.gauss(0, 1.5))
        self._sig_buf.append(new_val)
        self._sig_buf = self._sig_buf[-100:]

        c.delete("sig")
        n = len(self._sig_buf)
        if n < 2:
            return
        mn, mx = 0, 65
        pts = []
        for i, v in enumerate(self._sig_buf):
            px = int(i / (n - 1) * w)
            py = int(h - (max(mn, min(mx, v)) - mn) / (mx - mn) * (h - 6) - 3)
            pts.extend([px, py])
        color = "#e74c3c" if self.fall_detected else "#2ecc71"
        if len(pts) >= 4:
            c.create_line(*pts, fill=color, width=1.5,
                          tags="sig", smooth=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 비프음
    # ─────────────────────────────────────────────────────────────────────────

    def _beep(self, n: int, freq: int, dur: int):
        for _ in range(n):
            try:
                winsound.Beep(freq, dur)
                time.sleep(0.06)
            except Exception:
                break

    # ─────────────────────────────────────────────────────────────────────────
    # 종료
    # ─────────────────────────────────────────────────────────────────────────

    def _on_close(self):
        if self.fall_detected:
            if not messagebox.askyesno("종료 확인",
                    "현재 낙상 경보가 활성 상태입니다.\n정말 종료하시겠습니까?"):
                return
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    FallAlertTimer().run()
