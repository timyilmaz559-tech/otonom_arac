import cv2
import numpy as np
import time
import serial
import struct
from ultralytics import YOLO
from picamera2 import Picamera2

# Tek model - hem tespit hem sınıflandırma
detector = YOLO('last.pt')
siniflar = {0: 'park', 1: 'railroad', 2: 'walkside', 3: 'objects'}

MIN_ALAN = 200
MAX_ASPECT = 1.8
MIN_ASPECT = 0.6
MIN_GUVEN = 0.5

# Durum değişkenleri
stop_red_triangle = False
red_triangle_count = 0
ignore_red_triangle = False
detected_green_traffic_light = False

# Seri port
uart_serial = None
for port in ["/dev/ttyUSB0", "/dev/ttyAMA0", "/dev/ttyACM0"]:
    try:
        uart_serial = serial.Serial(port, baudrate=9600, timeout=2)
        print(f"Seri port açıldı: {port}")
        break
    except serial.SerialException:
        continue

START_BYTES = bytes([170, 85])
END_BYTE = bytes([13])


def motor_control(speed=0, steering=0):
    if uart_serial is None:
        print(f"[Motor] speed={speed}, steering={steering}")
        return
    speed = max(0, min(255, int(speed)))
    steering = max(-100, min(100, int(steering)))
    payload = struct.pack("Bb", speed, steering)
    uart_serial.write(START_BYTES + payload + END_BYTE)
    uart_serial.flush()
    print("Sent:", speed, steering)


# ------------------ Kamera Başlatma ------------------
def init_cameras():
    # Çizgi kamerası (index 0)
    cam_lane = Picamera2(0)
    cam_lane.configure(cam_lane.create_preview_configuration(
        main={"format": "BGR888", "size": (640, 480)}
    ))
    cam_lane.start()

    # Tespit kamerası (index 1)
    cam_detect = Picamera2(1)
    cam_detect.configure(cam_detect.create_preview_configuration(
        main={"format": "BGR888", "size": (640, 480)}
    ))
    cam_detect.start()

    time.sleep(0.5)  # Kameraların stabil olması için bekle
    print("Kameralar başlatıldı.")
    return cam_lane, cam_detect


# ------------------ Şerit Takibi ------------------
def detect_lane(frame, lane_type='both'):
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    mask = np.zeros_like(edges)
    roi_vertices = np.array([[
        (0, height), (width, height),
        (width, int(height * 0.6)), (0, int(height * 0.6))
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    roi = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
    left_lines_x, right_lines_x = [], []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            if dx == 0:
                continue
            slope = (y2 - y1) / dx
            if slope < -0.3:
                left_lines_x.append((x1 + x2) / 2)
            elif slope > 0.3:
                right_lines_x.append((x1 + x2) / 2)

    if lane_type == 'both' and left_lines_x and right_lines_x:
        lane_center = (np.mean(left_lines_x) + np.mean(right_lines_x)) / 2
    elif lane_type == 'left' and left_lines_x:
        lane_center = np.mean(left_lines_x) + 50
    elif lane_type == 'right' and right_lines_x:
        lane_center = np.mean(right_lines_x) - 50
    elif lane_type == 'park':
        lane_center = np.mean(right_lines_x) - 70 if right_lines_x else width * 0.7
    else:
        lane_center = width / 2

    return (width / 2) - lane_center


# ------------------ Tabela Tespiti ------------------
def detect_sign(frame):
    results = detector(frame)[0]

    for box in results.boxes:
        sinif_id = int(box.cls[0])
        guven = float(box.conf[0])

        if sinif_id not in siniflar or guven < MIN_GUVEN:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        alan = w * h
        aspect = w / h if h > 0 else 0

        if alan < MIN_ALAN or aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue

        label = siniflar[sinif_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: %{guven * 100:.1f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return label

    return None


# ------------------ Araç Model Tespiti ------------------
def detect_obj(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
    mask_orange = cv2.inRange(hsv, np.array([5, 100, 100]), np.array([15, 255, 255]))

    def detect_shape(contour):
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        return len(approx)

    for cnt in cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(cnt) > 400 and detect_shape(cnt) == 3:
            x, y, w, h = cv2.boundingRect(cnt)
            if 0.5 < w / h < 1.5:
                return "SOLLAMA SERBEST", h

    for cnt in cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(cnt) > 400 and detect_shape(cnt) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            return "SOLLAMA YASAK", h

    return None, None


# ------------------ Trafik Işığı ------------------
def detect_traffic_light(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([80, 255, 255]))

    for cnt in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(cnt)
        if 20 < area < 500:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0 and (4 * np.pi * area / (perimeter ** 2)) > 0.7:
                return True
    return False


# ------------------ Ana Döngü ------------------
def main():
    global ignore_red_triangle, stop_red_triangle
    global red_triangle_count, detected_green_traffic_light

    last_detection_time = 0
    cooldown = 10

    cam_lane, cam_detect = init_cameras()

    KP = 0.4
    speed = 100

    try:
        while True:
            # Picamera2'den frame al (numpy array, BGR)
            frame_lane = cam_lane.capture_array()
            frame_detect = cam_detect.capture_array()

            green_light = detect_traffic_light(frame_detect)
            if green_light and not detected_green_traffic_light:
                print("Yeşil ışık yandı")
                detected_green_traffic_light = True

            sign = detect_sign(frame_detect)
            obj, obj_height = detect_obj(frame_detect)

            offset = detect_lane(frame_lane, lane_type='both')
            steering = max(-100, min(100, offset * KP))

            if sign == "park":
                print("PARK ALANI - Sağa yanaş ve dur")
                offset = detect_lane(frame_lane, lane_type='park')
                steering = max(-100, min(100, offset * KP))
                motor_control(0, steering)
                time.sleep(2)
                motor_control(0, 0)
                break

            if (sign == "walkside" or sign == "railroad") and not stop_red_triangle:
                red_triangle_count += 1
                if red_triangle_count == 1:
                    motor_control(0, 0)
                    time.sleep(5)
                    ignore_red_triangle = True
                    print("First stop at red triangle")
                elif red_triangle_count == 2 and not ignore_red_triangle:
                    motor_control(0, 0)
                    time.sleep(5)
                    print("Second and last time at stop red triangle")
                    stop_red_triangle = True

            else:
                if detected_green_traffic_light:
                    motor_control(speed, steering)

            if obj == "SOLLAMA YASAK":
                motor_control(0, 45)
                break
            elif obj == "SOLLAMA SERBEST":
                motor_control(50, 45)
                continue

            current_time = time.time()
            if ignore_red_triangle and (current_time - last_detection_time > cooldown):
                last_detection_time = current_time
                ignore_red_triangle = False

            cv2.imshow("Lane Camera", frame_lane)
            cv2.imshow("Detection Camera", frame_detect)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cam_lane.stop()
        cam_detect.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
