import cv2
import speech_recognition as sr
import threading
import time
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from gpiozero import PWMOutputDevice
import board
import neopixel

# --- Hardware Initialization ---
pixels = neopixel.NeoPixel(board.MOSI, 8, brightness=0.3, auto_write=False, pixel_order=neopixel.GRB)
buzzer = PWMOutputDevice(12)

camera_active = False 
current_light_state = 0 # 0:Normal, 2:Fatigue, 3:Danger
buzzer_state = 0
user_color = (0, 50, 0) # Default: Green
brightness_level = 0.3  # LED brightness level

# --- MediaPipe Initialization ---
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# --- Math Helpers ---
def dist(p1, p2): return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
def get_ear(l): return (dist(l[159], l[145]) + dist(l[158], l[153])) / (2.0 * dist(l[133], l[33]))
def get_mar(l): return dist(l[13], l[14]) / dist(l[78], l[308])

# --- Buzzer Thread ---
def buzzer_worker():
    global buzzer_state
    while True:
        if buzzer_state == 2:
            buzzer.frequency = 800
            buzzer.value = 0.3 if int(time.time() * 2) % 2 == 0 else 0
        elif buzzer_state == 3:
            buzzer.frequency = 1300
            buzzer.value = 0.5 if int(time.time() * 10) % 2 == 0 else 0
        else:
            buzzer.value = 0
        time.sleep(0.05)

def update_hardware():
    global current_light_state, buzzer_state, camera_active, user_color, brightness_level
    
    pixels.brightness = brightness_level
    
    if not camera_active:
        pixels.fill((0, 0, 0)); pixels.show(); buzzer_state = 0
        return
    
    if current_light_state == 3: # Danger
        pixels.fill((255, 0, 0) if int(time.time() * 10) % 2 == 0 else (50, 0, 0)); buzzer_state = 3
    elif current_light_state == 2: # Fatigue
        pixels.fill((255, 0, 0)); buzzer_state = 2
    else: # Normal
        pixels.fill(user_color); buzzer_state = 0
    pixels.show()

# --- Camera Processing Thread ---
def camera_thread():
    global camera_active, current_light_state
    cap = cv2.VideoCapture(0)
    eye_counter = 0; mouth_counter = 0
    
    while True:
        ret, frame = cap.read()
        if ret and camera_active:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = detector.detect(mp_image)
            
            if res.face_landmarks:
                l = res.face_landmarks[0]
                ear, mar = get_ear(l), get_mar(l)
                
                # --- Visualization: Draw all 478 landmarks ---
                h, w, _ = frame.shape
                for lm in l:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 255), -1)
                
                eye_counter = eye_counter + 1 if ear < 0.18 else 0
                mouth_counter = mouth_counter + 1 if mar > 0.6 else 0
                
                if mouth_counter >= 30: current_light_state = 3
                elif eye_counter >= 20: current_light_state = 2
                elif eye_counter < 5 and mouth_counter < 5: current_light_state = 0
                
                update_hardware()
            cv2.imshow('Fatigue Detection', frame)
        else:
            cv2.destroyAllWindows()
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()

# --- Voice Control Thread ---
def listen_voice():
    global camera_active, user_color, brightness_level
    r = sr.Recognizer()
    mic = sr.Microphone()
    while True:
        try:
            with mic as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.listen(source, timeout=5, phrase_time_limit=3)
                text = r.recognize_google(audio, language="zh-TW")
                
                if "開啟" in text: camera_active = True
                elif "關閉" in text: camera_active = False
                elif "紅色" in text: user_color = (255, 0, 0)
                elif "綠色" in text: user_color = (0, 255, 0)
                elif "藍色" in text: user_color = (0, 0, 255)
                elif "白色" in text: user_color = (255, 255, 255)
                elif any(word in text for word in ["調亮", "亮一點"]): 
                    brightness_level = min(1.0, brightness_level + 0.1)
                elif any(word in text for word in ["調暗", "暗一點"]): 
                    brightness_level = max(0.0, brightness_level - 0.1)
                elif "全亮" in text: brightness_level = 1.0
                elif any(word in text for word in ["全暗", "關燈"]): brightness_level = 0.0
        except: pass

threading.Thread(target=buzzer_worker, daemon=True).start()
threading.Thread(target=listen_voice, daemon=True).start()
camera_thread()
