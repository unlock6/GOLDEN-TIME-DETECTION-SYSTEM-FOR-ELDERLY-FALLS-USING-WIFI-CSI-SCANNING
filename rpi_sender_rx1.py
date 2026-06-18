import serial
import socket
import time

# ── 설정 ──────────────────────────────
ESP32_PORT = '/dev/ttyACMO'   # ESP32-S3 연결 포트
BAUD = 921600

PC_IP = '192.168.123.6'       # PC IP 주소 (수요일에 확인 후 변경)
UDP_PORT = 5000               # RX1 전용 포트
RX_NAME = 'RX1_동'
# ──────────────────────────────────────

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(f'[{RX_NAME}] 시작 → {PC_IP}:{UDP_PORT}')

try:
    ser = serial.Serial(ESP32_PORT, BAUD, timeout=1)
    buffer = ''
    count = 0
    while True:
        try:
            chunk = ser.read(ser.in_waiting or 1).decode('utf-8', errors='ignore')
            buffer += chunk
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()
                if line.startswith('CSI_DATA'):
                    msg = f'{RX_NAME}|{line}'
                    sock.sendto(msg.encode(), (PC_IP, UDP_PORT))
                    count += 1
                    if count % 100 == 0:
                        print(f'[{RX_NAME}] {count}개 전송')
        except Exception as e:
            pass
except KeyboardInterrupt:
    print(f'[{RX_NAME}] 종료')
finally:
    sock.close()