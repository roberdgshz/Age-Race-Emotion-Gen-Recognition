from deepface import DeepFace
import cv2
import mediapipe as mp

detros = mp.solutions.face_detection
rostros = detros.FaceDetection(min_detection_confidence=0.0, model_selection=0)

dibujoRostro = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    img = cv2.imread("img.png")
    img = cv2.resize(img, (0,0), None, 0.18, 0.18)
    ani, ali, c = img.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    respuesta_rostros = rostros.process(rgb)

    if respuesta_rostros.detections is not None:
        for rostro in respuesta_rostros.detections:
            al, an, c = frame.shape
            box = rostro.location_data.relative_bounding_box
            xi, yi, w, h = int(box.xmin * an), int(box.ymin * al), int(box.width * an), int(box.height * al)
            xf, yf = xi + w, yi + h

            cv2.rectangle(frame, (xi, yi), (xf, yi), (255, 255, 0), 1)
            frame[10:ani + 10, 10:ali+10] = img

            info = DeepFace.analyze(rgb, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
            edad = info['age']
            emociones = info['dominant_emotion']
            race = info['dominant_race']
            gen = info['gender']

            cv2.putText(frame, str(gen), (65, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, str(edad), (75, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, str(emociones), (75, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, str(race), (75, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Edad: ", frame)

    t = cv2.waitKey(5)
    if t == 27:
        break

cv2.destroyAllWindows()
cap.release()