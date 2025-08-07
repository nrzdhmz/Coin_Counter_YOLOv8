import cv2
from detect.utils import load_model, count_coins, calculate_total

model = load_model()
class_names = model.names

image = cv2.imread('my_image.png')
results = model(image)
result = results[0]

coin_counts = count_coins(result, class_names)
total = calculate_total(coin_counts)

annotated_frame = result.plot()
cv2.putText(annotated_frame, f"Total: {total:.2f} AZN", (10,40),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

cv2.imshow("Detected Coins", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Coin counts:", coin_counts)
print(f"Total amount: {total} AZN")
