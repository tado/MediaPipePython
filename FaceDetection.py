#OpenCVとMediaPipeをインポート
import cv2
import mediapipe as mp
#MediaPipeと顔検出と描画のユーティリティを宣言
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

#webカメラから入力
cap = cv2.VideoCapture(0)
#顔検出開始
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)
    #カメラ映像の上に検出された顔のアノテーションを描画
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        mp_drawing.draw_detection(image, detection)
    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    #escキーで終了
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()