import cv2
import numpy as np
import time
import serial
import struct

# Kamerayı aç
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

# Seri haberleşme
uart_serial = serial.Serial("/dev/ttyUSB0", baudrate=9600, timeout=2)
START_BYTES = bytes([170, 85])
END_BYTE = bytes([13])

# Renk aralıkları
# Kırmızı (dur, yaya geçidi, demiryolu)
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Mavi (park)
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([130, 255, 255])

# Parametreler
MIN_AREA = 500  # Minimum tabela alanı (uzaktakileri ele)
MAX_AREA = 20000  # Maksimum tabela alanı
DETECTION_DISTANCE = 5000  # Algılama mesafesi (alan olarak) - 5000 pikselden büyükse yakın
COOLDOWN_TIME = 10  # Kırmızı tabela cooldown süresi (saniye)
PARK_DURATION = 3  # Parkta bekleme süresi (saniye)

# Durum değişkenleri
last_red_time = 0
red_cooldown_active = False
is_parked = False
park_start_time = 0

def motor_control(speed=0, steering=0):
    speed = max(0, min(255, int(speed)))
    steering = max(-100, min(100, int(steering)))
    
    payload = struct.pack("Bb", speed, steering)
    packet = bytes([170, 85]) + payload + bytes([13])
    
    uart_serial.write(packet)
    uart_serial.flush()
    print(f"🚗 Hız: {speed}, Direksiyon: {steering}")

def get_distance_from_area(area):
    """Alan büyüklüğünden yaklaşık mesafe tahmini"""
    if area > 10000:
        return "ÇOK YAKIN"
    elif area > DETECTION_DISTANCE:
        return "YAKIN"
    elif area > 2000:
        return "ORTA"
    else:
        return "UZAK"

def detect_red_triangle(mask):
    """Kırmızı üçgen tespiti (yaya geçidi, demiryolu)"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA:
            # Şekil analizi
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            
            # 3 köşe varsa üçgen
            if len(approx) == 3:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / h if h > 0 else 0
                
                # Üçgen şeklinde ve düzgün oranlarda mı?
                if 0.7 < aspect_ratio < 1.3:
                    results.append((area, x, y, w, h))
    
    # En büyük alana göre sırala (en yakın tabela)
    if results:
        results.sort(reverse=True)  # En büyük alan en yakın
        area, x, y, w, h = results[0]
        return True, (x, y, w, h, area), "TRAFFIC_SIGN"
    
    return False, None, None

def detect_blue_square(mask):
    """Mavi kare tespiti (park)"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA:
            # Şekil analizi
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            
            # 4 köşe varsa kare/dikdörtgen
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / h if h > 0 else 0
                
                # Kareye yakın mı? (0.8-1.2 arası)
                if 0.8 < aspect_ratio < 1.2:
                    results.append((area, x, y, w, h))
    
    # En büyük alana göre sırala (en yakın tabela)
    if results:
        results.sort(reverse=True)  # En büyük alan en yakın
        area, x, y, w, h = results[0]
        return True, (x, y, w, h, area), "PARK"
    
    return False, None, None

print("🔵 Sistem başlatıldı...")
print("SADECE YAKIN MESAFEDEKİ TABELALAR ALGILANACAK")
print("Kırmızı üçgen: Yaya geçidi/Demiryolu (10sn cooldown)")
print("Mavi kare: Park alanı (3sn bekle)")
print(f"Algılama eşiği: {DETECTION_DISTANCE} piksel alan")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamıyor!")
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    current_time = time.time()
    height, width = frame.shape[:2]
    
    # Cooldown kontrolü
    if red_cooldown_active and (current_time - last_red_time) > COOLDOWN_TIME:
        red_cooldown_active = False
        print("⏱️ Kırmızı tabela cooldown bitti, tekrar algılanabilir")
    
    # Park kontrolü
    if is_parked and (current_time - park_start_time) > PARK_DURATION:
        is_parked = False
        print("✅ Park süresi doldu, harekete geçiliyor")
        motor_control(60, 0)
    
    # Eğer park halindeyse tespit yapma
    if is_parked:
        cv2.putText(frame, "PARK HALINDE", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Tabela Tespit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    # Kırmızı tespiti
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # Mavi tespiti
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Morfolojik işlemler
    kernel = np.ones((5,5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
    
    # Mavi kontrol (park)
    blue_detected, blue_bbox, blue_type = detect_blue_square(mask_blue)
    
    if blue_detected:
        x, y, w, h, area = blue_bbox
        mesafe = get_distance_from_area(area)
        
        # Görselleştirme
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        
        # Renk koduna göre etiket rengi
        if area > DETECTION_DISTANCE:
            etiket_rengi = (0, 255, 0)  # Yeşil - yakın
            cv2.putText(frame, "PARK ALANI - YAKIN", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, etiket_rengi, 2)
        else:
            etiket_rengi = (0, 255, 255)  # Sarı - uzak
            cv2.putText(frame, f"PARK ALANI - {mesafe}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, etiket_rengi, 2)
        
        # Sadece yakınsa park et
        if area > DETECTION_DISTANCE:
            print(f"🅿️ MAVİ KARE TESPİT EDİLDİ - YAKIN MESAFE ({int(area)} px)")
            motor_control(0, 0)
            is_parked = True
            park_start_time = current_time
            continue
    
    # Kırmızı kontrol (cooldown yoksa ve mavi yoksa)
    if not red_cooldown_active and not blue_detected:
        red_detected, red_bbox, red_type = detect_red_triangle(mask_red)
        
        if red_detected:
            x, y, w, h, area = red_bbox
            mesafe = get_distance_from_area(area)
            
            # Görselleştirme
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
            
            # Sadece yakınsa dur
            if area > DETECTION_DISTANCE:
                cv2.putText(frame, "TRAFIK TABELASI - DUR!", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(f"🔴 KIRMIZI ÜÇGEN TESPİT EDİLDİ - YAKIN MESAFE ({int(area)} px)")
                motor_control(0, 0)
                last_red_time = current_time
                red_cooldown_active = True
                continue
            else:
                cv2.putText(frame, f"TRAFIK TABELASI - {mesafe}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                print(f"🔴 KIRMIZI ÜÇGEN GÖRÜLDÜ AMA UZAK - {mesafe} ({int(area)} px)")
    
    # Normal sürüş
    if not is_parked:
        motor_control(60, 0)
        
        # Bilgilendirme yazıları
        if red_cooldown_active:
            kalan = int(COOLDOWN_TIME - (current_time - last_red_time))
            cv2.putText(frame, f"COOLDOWN: {kalan}sn", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Mesafe bilgisi
        cv2.putText(frame, f"Algilama Esigi: {DETECTION_DISTANCE} px", (width-300, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Görüntüyü göster
    cv2.imshow("Tabela Tespit - Mesafe Bazli", frame)
    
    # Çıkış
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🔴 Sistem durduruldu")
