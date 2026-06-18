import socket
import numpy as np
import ast
import threading
import time
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ── 설정 ──────────────────────────────
UDP_PORT_RX1 = 5000     # RX1 (동쪽) 포트
UDP_PORT_RX2 = 5001     # RX2 (북쪽) 포트
WINDOW = 100            # 화면에 보여줄 데이터 수
THRESHOLD = 15.0        # 낙상 감지 임계값
# ──────────────────────────────────────

rx1_data = deque([0] * WINDOW, maxlen=WINDOW)
rx2_data = deque([0] * WINDOW, maxlen=WINDOW)
stop_flag = threading.Event()

def receive_udp(port, data_queue, rx_name):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', port))
    sock.settimeout(1.0)
    prev = None
    print(f'[{rx_name}] UDP 수신 대기 중 (포트 {port})')
    while not stop_flag.is_set():
        try:
            data, _ = sock.recvfrom(4096)
            msg = data.decode('utf-8', errors='ignore')
            if '|' in msg:
                _, line = msg.split('|', 1)
                line = line.strip()
                if line.startswith('CSI_DATA'):
                    try:
                        arr_start = line.index('[')
                        arr_end = line.index(']') + 1
                        arr = ast.literal_eval(line[arr_start:arr_end])
                        if prev is not None:
                            diff = float(np.abs(np.array(arr) - np.array(prev)).mean())
                            data_queue.append(diff)
                        prev = arr
                    except:
                        pass
        except socket.timeout:
            pass
        except Exception as e:
            pass
    sock.close()

t1 = threading.Thread(target=receive_udp, args=(UDP_PORT_RX1, rx1_data, 'RX1_동'))
t2 = threading.Thread(target=receive_udp, args=(UDP_PORT_RX2, rx2_data, 'RX2_북'))
t1.daemon = True
t2.daemon = True
t1.start()
t2.start()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('CSI 실시간 모니터링 - 낙상 감지 시스템', fontsize=14, fontweight='bold')

line1, = ax1.plot([], [], 'b-', linewidth=1.5, label='RX1 (동)')
ax1.axhline(y=THRESHOLD, color='r', linestyle='--', label=f'낙상 임계값 ({THRESHOLD})')
ax1.set_xlim(0, WINDOW)
ax1.set_ylim(0, 30)
ax1.set_title('RX1 (동쪽)')
ax1.set_ylabel('CSI 변화량')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
status1 = ax1.text(0.02, 0.88, '정상', transform=ax1.transAxes,
                   fontsize=12, color='green', fontweight='bold')
count1 = ax1.text(0.85, 0.88, '수신 대기 중', transform=ax1.transAxes,
                  fontsize=9, color='gray')

line2, = ax2.plot([], [], 'g-', linewidth=1.5, label='RX2 (북)')
ax2.axhline(y=THRESHOLD, color='r', linestyle='--', label=f'낙상 임계값 ({THRESHOLD})')
ax2.set_xlim(0, WINDOW)
ax2.set_ylim(0, 30)
ax2.set_title('RX2 (북쪽)')
ax2.set_ylabel('CSI 변화량')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
status2 = ax2.text(0.02, 0.88, '정상', transform=ax2.transAxes,
                   fontsize=12, color='green', fontweight='bold')
count2 = ax2.text(0.85, 0.88, '수신 대기 중', transform=ax2.transAxes,
                  fontsize=9, color='gray')

plt.tight_layout()

def update(frame):
    data1 = list(rx1_data)
    data2 = list(rx2_data)
    line1.set_data(range(len(data1)), data1)
    line2.set_data(range(len(data2)), data2)
    if len(data1) >= 10:
        recent1 = np.mean(data1[-10:])
        count1.set_text(f'평균: {recent1:.2f}')
        if recent1 > THRESHOLD:
            ax1.set_facecolor('#ffe0e0')
            status1.set_text('⚠️ 낙상 의심!')
            status1.set_color('red')
        else:
            ax1.set_facecolor('white')
            status1.set_text('정상')
            status1.set_color('green')
    if len(data2) >= 10:
        recent2 = np.mean(data2[-10:])
        count2.set_text(f'평균: {recent2:.2f}')
        if recent2 > THRESHOLD:
            ax2.set_facecolor('#ffe0e0')
            status2.set_text('⚠️ 낙상 의심!')
            status2.set_color('red')
        else:
            ax2.set_facecolor('white')
            status2.set_text('정상')
            status2.set_color('green')
    return line1, line2, status1, status2, count1, count2

ani = animation.FuncAnimation(fig, update, interval=100, blit=False)

try:
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    stop_flag.set()
    print('종료')