import cv2  # 用來存取攝影機與顯示畫面
import mediapipe as mp  # 導入 MediaPipe 函式庫

# 初始化 mediapipe 的人臉偵測模組
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 開啟攝影機
cap = cv2.VideoCapture(0)  # 0 表示預設攝影機

# 使用 mediapipe 的 FaceDetection 模型
with mp_face_detection.FaceDetection(
        model_selection=0,  # 0: 適合近距離人臉（約 2 公尺以內）
        min_detection_confidence=0.5  # 最小偵測信心門檻
    ) as face_detection:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("無法讀取畫面")
            break

        # 轉換 BGR 影像為 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 改為不可寫模式以加速處理
        image.flags.writeable = False

        # 偵測人臉
        results = face_detection.process(image)

        # 還原為可寫入模式
        image.flags.writeable = True

        # 再轉回 BGR 以便 OpenCV 顯示
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 如果有偵測到人臉
        if results.detections:
            for detection in results.detections:
                # 畫出人臉偵測框與關鍵點
                mp_drawing.draw_detection(image, detection)

        # 顯示影像
        cv2.imshow('Face Detection', image)

        # 按下 q 鍵即可結束
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# 釋放攝影機與關閉視窗
cap.release()
cv2.destroyAllWindows()

'''
練習建議：
1. 嘗試調整 `min_detection_confidence` 的值，觀察效果
2. 嘗試更換 `model_selection` 為 1，適合遠距影像
'''