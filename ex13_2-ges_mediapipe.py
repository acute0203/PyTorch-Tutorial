import cv2
import mediapipe as mp
import math

# 初始化 mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 計算兩點距離
def distance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

# 判斷哪些手指是伸直的（依據 landmark y 值）
def get_finger_status(hand_landmarks):
    finger_status = []

    # 手掌方向（判斷是左手還是右手）
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    if wrist.x < index_mcp.x:
        is_right_hand = True
    else:
        is_right_hand = False

    # 大拇指 (左右方向)
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    if is_right_hand:
        finger_status.append(thumb_tip.x < thumb_ip.x)  # 右手大拇指向左張開
    else:
        finger_status.append(thumb_tip.x > thumb_ip.x)  # 左手大拇指向右張開

    # 其他手指 (由下往上張開)
    tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP]
    pips = [mp_hands.HandLandmark.INDEX_FINGER_PIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            mp_hands.HandLandmark.RING_FINGER_PIP,
            mp_hands.HandLandmark.PINKY_PIP]

    for tip, pip in zip(tips, pips):
        finger_status.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)

    return finger_status  # [thumb, index, middle, ring, pinky]

# 判斷手勢
def detect_gesture(finger_status, landmarks):
    # 👍 比讚：只有大拇指伸直
    if finger_status == [True, False, False, False, False]:
        return "👍 Thumbs Up"

    # 💗 mini 愛心：大拇指與食指指尖靠近，其餘彎曲
    d = distance(landmarks[mp_hands.HandLandmark.THUMB_TIP],
                 landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP])
    if d < 0.03 and finger_status[2:] == [False, False, False]:
        return "💗 Mini Heart"

    # 數字 1~5：依據張開手指的數量
    count = sum(finger_status)
    if 1 <= count <= 5:
        return f"🖐️ Number {count}"

    return None

# 啟動攝影機
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV 影像前處理
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        # OpenCV 繪圖後處理
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 繪製關鍵點
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 手勢判斷
                finger_status = get_finger_status(hand_landmarks)
                gesture = detect_gesture(finger_status, hand_landmarks.landmark)

                if gesture:
                    print(f"偵測到手勢：{gesture}")
                    # 可視化標示
                    cv2.putText(image, gesture, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2.imshow("Hand Gesture Recognition", image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
