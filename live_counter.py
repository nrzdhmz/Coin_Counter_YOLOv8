import cv2
import math
from detect.utils import load_model, count_coins, calculate_total

model = load_model()
class_names = model.names

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam Error occured, check if you are using it for any other app")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5, stream=True)  
    coin_counts = {class_names[i]: 0 for i in class_names}

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = class_names[class_id]
            coin_counts[label] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{label}", (max(0,x1),max(0,y1+20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    total = calculate_total(coin_counts)

    cv2.putText(frame, f"Total: {total:.2f} AZN", (10,40),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
    cv2.imshow("Coin Counter", frame)
    if cv2.waitKey(1)==ord('q'):
        break

    

cap.release()
cv2.destroyAllWindows()
