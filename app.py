import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# startup par ek baar ONNX model load karo (RAM & speed safe)
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
CLASSES = ["hole", "oil", "slub", "snag"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    conf_threshold = float(request.form.get("conf", 0.4))
    iou_threshold = 0.45

    file_bytes = np.frombuffer(request.files["file"].read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    orig_h, orig_w = img.shape[:2]

    # Preprocess (640x640 RGB normalized)
    blob = cv2.resize(img, (640, 640))
    blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None, ...]

    # omnx Inference -> output shape (1, 8, 8400) -> transpose to (8400, 8)
    preds = session.run(None, {input_name: blob})[0][0].T

    boxes, scores, class_ids = [], [], []
    for row in preds:
        cls_scores = row[4:]
        cls_id = int(np.argmax(cls_scores))
        score = float(cls_scores[cls_id])
        if score >= conf_threshold:
            cx, cy, w, h = row[:4]
            x1 = int((cx - w / 2) * orig_w / 640)
            y1 = int((cy - h / 2) * orig_h / 640)
            boxes.append([x1, y1, int(w * orig_w / 640), int(h * orig_h / 640)])
            scores.append(score)
            class_ids.append(cls_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
    detections = []
    for i in indices:
        idx = int(i)
        x, y, w, h = boxes[idx]
        detections.append({
            "defect": CLASSES[class_ids[idx]],
            "confidence": round(scores[idx], 4),
            "bbox": [x, y, x + w, y + h],
        })

    return jsonify({"count": len(detections), "detections": detections})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
