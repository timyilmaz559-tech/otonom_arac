import cv2
import numpy as np
import time
import serial
import struct
from ultralytics import YOLO
import tensorflow as tf

detector = YOLO('yolo26n.pt')

classifier = tf.keras.models.load_model('tabela_cnn.h5')
siniflar = ['crosswalk', 'railroad', 'park', 'other']

MIN_ALAN = 200
MAX_ASPECT = 1.8
MIN_ASPECT = 0.6
TABELA_SINIFLARI = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]

# Durum değişkenleri
stop_crosswalk = False
stop_red_triangle = False
red_triangle_count = 0
ignore_red_triangle = False
detected_green_traffic_light = False

uart_serial = serial.Serial("COM10", baudrate=9600, timeout=2)

START_BYTES = bytes([170, 85])
END_BYTE = bytes([13])


def motor_control(speed=0, steering=0):
    speed = max(0, min(255, int(speed)))
    steering = max(-100, min(100, int(steering)))

    payload = struct.pack("Bb", speed, steering)
    packet = START_BYTES + payload + END_BYTE

    uart_serial.write(packet)
    uart_serial.flush()

    print("Sent:", speed, steering)


# ------------------ Şerit Takibi ------------------
def detect_lane(frame, lane_type='both'):
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    mask = np.zeros_like(edges)
    roi_vertices = np.array([[
        (0, height),
        (width, height),
        (width, int(height * 0.6)),
        (0, int(height * 0.6))
    ]], dtype=np.int32)

    cv2.fillPoly(mask, roi_vertices, 255)
    roi = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(roi, 1, np.pi/180, 50, minLineLength=40, maxLineGap=100)

    left_lines_x = []
    right_lines_x = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue

            slope = dy / dx

            if slope < -0.3:
                left_lines_x.append((x1 + x2) / 2)

            elif slope > 0.3:
                right_lines_x.append((x1 + x2) / 2)

    if lane_type == 'both' and left_lines_x and right_lines_x:
        left_mean = np.mean(left_lines_x)
        right_mean = np.mean(right_lines_x)
        lane_center = (left_mean + right_mean) / 2

    elif lane_type == 'left' and left_lines_x:
        lane_center = np.mean(left_lines_x) + 50

    elif lane_type == 'right' and right_lines_x:
        lane_center = np.mean(right_lines_x) - 50

    elif lane_type == 'park':
        if right_lines_x:
            lane_center = np.mean(right_lines_x) - 70
        else:
            lane_center = width * 0.7

    else:
        lane_center = width / 2

    car_center = width / 2
    offset = car_center - lane_center

    return offset


# ------------------ Tabela Tespiti ------------------
def detect_sign(frame):

    results = detector(frame)[0]

    for box in results.boxes:

        sinif_id = int(box.cls[0])

        if sinif_id not in TABELA_SINIFLARI:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        w, h = x2-x1, y2-y1
        alan = w * h
        aspect = w / h if h > 0 else 0

        if alan < MIN_ALAN:
            continue

        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue

        tabela = frame[y1:y2, x1:x2]

        if tabela.size == 0:
            continue

        img = cv2.resize(tabela, (64, 64))
        img = np.expand_dims(img, 0) / 255.0

        tahmin = classifier.predict(img, verbose=0)[0]

        sinif = np.argmax(tahmin)
        guven = tahmin[sinif]

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        cv2.putText(frame, f"{siniflar[sinif]}: %{guven*100:.1f}",
                   (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5,
                   (0,255,0),
                   2)

        return siniflar[sinif]


# ------------------ Araç Model Tespiti ------------------
def detect_obj(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([15, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    def detect_shape(contour):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        return len(approx)

    contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_orange:

        area = cv2.contourArea(cnt)

        if area > 400:

            shape = detect_shape(cnt)

            if shape == 3:

                x, y, w, h = cv2.boundingRect(cnt)

                aspect_ratio = w / h

                if 0.5 < aspect_ratio < 1.5:
                    return "SOLLAMA SERBEST", h


    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_yellow:

        area = cv2.contourArea(cnt)

        if area > 400:

            shape = detect_shape(cnt)

            if shape == 4:

                x, y, w, h = cv2.boundingRect(cnt)

                return "SOLLAMA YASAK", h

    return None, None


# ------------------ Trafik Işığı ------------------
def detect_traffic_light(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    conturs_green, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in conturs_green:

        area = cv2.contourArea(cnt)

        if area > 20 and area < 500:

            perimeter = cv2.arcLength(cnt, True)

            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if circularity > 0.7:
                return True

    return False


# ------------------ Ana Döngü ------------------
def main():

    global ignore_red_triangle
    global stop_crosswalk
    global stop_red_triangle
    global red_triangle_count
    global detected_green_traffic_light

    last_detection_time = 0
    cooldown = 10

    # ==========================================================
    # Camera 0 -> Çizgi Takip
    # Camera 1 -> Diğer
    # ==========================================================

    cap_lane = cv2.VideoCapture(0)
    cap_detect = cv2.VideoCapture(1)

    if not cap_lane.isOpened():
        print("Çizgi camera açılamadı!")
        return

    if not cap_detect.isOpened():
        print("Misc. camera açılamadı!")
        return

    # ==========================================================

    KP = 0.4
    speed = 100

    while True:

        ret_lane, frame_lane = cap_lane.read()
        ret_detect, frame_detect = cap_detect.read()

        if not ret_lane or not ret_detect:
            break

        green_light = detect_traffic_light(frame_detect)

        if green_light == True and detected_green_traffic_light == False:
            print("Yeşil ışık yandı")
            detected_green_traffic_light = True



        sign = detect_sign(frame_detect)
        obj, obj_height = detect_obj(frame_detect)

        offset = detect_lane(frame_lane, lane_type='both')
        steering = max(-100, min(100, offset * KP))

        # ======================================================


        if sign == "park":

            print("PARK ALANI - Sağa yanaş ve dur")

            offset = detect_lane(frame_lane, lane_type='park')

            steering = max(-100, min(100, offset * KP))

            motor_control(0, steering)

            time.sleep(2)

            motor_control(0, 0)

        elif (sign == "crosswalk" or sign == "railroad") and (not stop_red_triangle == True):

            red_triangle_count += 1

            if red_triangle_count == 1:

                motor_control(0, 0)

                time.sleep(5)

                ignore_red_triangle = True

                print("First stop at red triangle")

            elif red_triangle_count == 2 and not ignore_red_triangle == True:

                motor_control(0, 0)

                time.sleep(5)

                print("Second and last time at stop red triangle")

                stop_red_triangle = True

        else:

            if detected_green_traffic_light == True:

                motor_control(speed, steering)


        if obj == "SOLLAMA YASAK":

            motor_control(0, 45)
            return

        elif obj == "SOLLAMA SERBEST":

            motor_control(50, 45)
            continue


        current_time = time.time()

        if ignore_red_triangle == True:

            if current_time - last_detection_time > cooldown:

                last_detection_time = current_time

                ignore_red_triangle = False


        # ======================================================
        # DEBUG 
        # ======================================================

        cv2.imshow("Lane Camera", frame_lane)
        cv2.imshow("Detection Camera", frame_detect)

        # ======================================================


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap_lane.release()
    cap_detect.release()

    # ==========================================================

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

