from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os

# Paths
data_yaml_path = r"C:\Users\SAINATH NIKAM\Desktop\CV&DL Lab\CV CODES\Final Practical Practice\Persian_Car_Plates_YOLOV8\data.yaml"
raw_test_folder = r"C:\Users\SAINATH NIKAM\Desktop\CV&DL Lab\CV CODES\Final Practical Practice\Persian_Car_Plates_YOLOV8\extra"

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Ensure yolov8n.pt is available locally

# Train YOLOv8 model
model.train(
    data=data_yaml_path,
    epochs=5,             
    imgsz=640,
    batch=8,        
    name='offline_plate_model_quick'
)

# test image
test_image_path = os.path.join(raw_test_folder, "test3.jpg") 
# Load the image
img = cv2.imread(test_image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

# Run prediction
results = model.predict(source=img, conf=0.5, save=False)
result = results[0]

# Draw bounding boxes on the image
for box in result.boxes.xyxy:
    x1, y1, x2, y2 = map(int, box[:4])
    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

# Plot the image with bounding boxes
plt.figure(figsize=(8, 6))
plt.imshow(img_rgb)
plt.axis('off')
plt.title("Predictions for Test Image")
plt.show()
