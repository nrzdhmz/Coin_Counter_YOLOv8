from ultralytics import YOLO

coin_values = {
    '5q': 0.05,
    '10q': 0.10,
    '20q': 0.20,
    '50q': 0.50
}

def load_model(model_path='runs/detect/train/weights/best.pt'):
    return YOLO(model_path)

def count_coins(result, class_names):
    coin_counts = {class_names[k]: 0 for k in class_names}
    for box in result.boxes:
        class_id = int(box.cls[0])
        label = class_names[class_id]
        print(f"Detected class_id={class_id}, label={label}")  
        coin_counts[label] += 1
    print(f"Coin counts: {coin_counts}")  
    return coin_counts


def calculate_total(coin_counts):
    return sum(coin_counts[k] * coin_values[k] for k in coin_counts)
