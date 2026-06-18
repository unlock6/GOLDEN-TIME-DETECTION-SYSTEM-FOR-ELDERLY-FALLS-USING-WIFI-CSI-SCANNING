import socket
import csv
import threading
import time
from datetime import datetime

# ── 설정 ──────────────────────────────
UDP_PORT_RX1 = 5000     # RX1 (동쪽) 포트
UDP_PORT_RX2 = 5001     # RX2 (북쪽) 포트
# ──────────────────────────────────────

label = input('라벨 입력 (static / stand / fall / lie_down): ')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

file1 = f'csi_rx1_{label}_{timestamp}.csv'
file2 = f'csi_rx2_{label}_{timestamp}.csv'

print(f'\n저장 파일')
print(f'  RX1 (동): {file1}')
print(f'  RX2 (북): {file2}')
print(f'\n3초 후 수집 시작...')

for i in range(3, 0, -1):
    print(f'{i}...')
    time.sleep(1)
print('시작! Ctrl+C로 종료\n')

stop_flag = threading.Event()

def collect_udp(port, filename, rx_name):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', port))
    sock.settimeout(1.0)
    count = 0
    print(f'[{rx_name}] UDP 수신 대기 중 (포트 {port})')
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'rx', 'raw_data'])
        while not stop_flag.is_set():
            try:
                data, _ = sock.recvfrom(4096)
                msg = data.decode('utf-8', errors='ignore')
                if '|' in msg:
                    _, line = msg.split('|', 1)
                    line = line.strip()
                    if line.startswith('CSI_DATA'):
                        writer.writerow([label, rx_name, line])
                        f.flush()
                        count += 1
                        if count % 100 == 0:
                            print(f'[{rx_name}] {count}개 수집')
            except socket.timeout:
                pass
            except Exception as e:
                pass
    sock.close()
    print(f'[{rx_name}] 저장 완료: {filename} ({count}개)')

# 두 스레드 동시 실행
t1 = threading.Thread(target=collect_udp, args=(UDP_PORT_RX1, file1, 'RX1_동'))
t2 = threading.Thread(target=collect_udp, args=(UDP_PORT_RX2, file2, 'RX2_북'))

t1.start()
t2.start()

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print('\n종료 중...')
    stop_flag.set()
    t1.join()
    t2.join()
    print('완료!')