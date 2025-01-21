import cv2
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection

model_name = "facebook/detr-resnet-50"
processor = DetrImageProcessor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)

cap = cv2.VideoCapture('/Users/georgijgavrilov/programm_engineering_4/7440_Surfer_Beach_1920x1080.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    inputs = processor(images=frame, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([frame.shape[0:2]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 3)
        cv2.putText(frame, f"Label: {label} | Score: {round(score.item(), 3)}", (int(box[0]), int(box[1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
