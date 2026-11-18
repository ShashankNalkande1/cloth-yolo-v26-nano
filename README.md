
# Fabric Defect Detection System

**Real-time textile defect detection powered by YOLO26n + ONNX Runtime + Flask — engineered for ultra-low-resource CPU environments and edge deployment.**

Detects four textile defects: `hole`, `oil`, `slub`, and `snag`.

---

## 1. Why This Model Exists

### The Problem

Traditional textile quality inspection depends heavily on:
- Expensive GPU-powered computer vision infrastructure.
- Manual visual inspection, which is time-consuming and inconsistent.
- Resource-intensive deep learning models that are difficult to deploy on low-end hardware.

### The Engineering Goal: Extreme Portability

This project focuses on deploying an efficient 4-class YOLO26n object detection model without requiring heavy PyTorch dependencies in production.

| Metric | Value |
|---|---|
| Original PyTorch model | 5.5 MB (`.pt`) |
| Exported ONNX model | 9.4 MB (`.onnx`) |
| Detection classes | 4 |
| Production inference | ONNX Runtime |
| Execution provider | CPUExecutionProvider |
| Target RAM | Under 150 MB |
| Deployment target | Render 512 MB RAM Free Tier |

The objective is simple: make computer vision practical on low-end devices, edge CPUs, and constrained cloud infrastructure.

---

## 2. Tech Stack

### Model & Export
![YOLO](https://img.shields.io/badge/Ultralytics-YOLO26n-111111?logo=ultralytics&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Model_Export-005CED?logo=onnx&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-CPU_Inference-0078D4)
![OpenCV](https://img.shields.io/badge/OpenCV-Headless-5C3EE8?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Array_Processing-013243?logo=numpy&logoColor=white)

- Ultralytics YOLO26n
- ONNX model export
- ONNX Runtime (`CPUExecutionProvider`)
- OpenCV Headless
- NumPy

### Backend & Server
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Backend-000000?logo=flask&logoColor=white)
![Gunicorn](https://img.shields.io/badge/Gunicorn-WSGI_Server-499848)

- Python
- Flask
- Gunicorn

### Frontend UI
![HTML5](https://img.shields.io/badge/HTML5-Canvas-E34F26?logo=html5&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-Vanilla-F7DF1E?logo=javascript&logoColor=black)

- HTML5 Canvas
- Vanilla JavaScript
- Live webcam detection
- Image upload and inference
- Client-side bounding box rendering

### Hosting
![Render](https://img.shields.io/badge/Hosted_on-Render-46E3B7?logo=render&logoColor=black)
![CPU](https://img.shields.io/badge/Inference-CPU_Only-success)
![Memory](https://img.shields.io/badge/Target_RAM-512_MB-blue)

- Render Cloud
- 512 MB RAM Free Tier target

---

## 3. Architectural Decisions

### Why ONNX Instead of PyTorch?

Deploying the original `.pt` model requires PyTorch and its associated runtime dependencies, increasing deployment size and memory requirements.

ONNX provides a standalone computational graph that can be executed through ONNX Runtime.

| Factor | PyTorch (.pt) | ONNX (.onnx) |
|---|---|---|
| Runtime | PyTorch | ONNX Runtime |
| Model size | 5.5 MB | 9.4 MB |
| CPU execution | Supported | Supported |
| Heavy PyTorch dependency | Required | Not required |
| Deployment footprint | Higher | Lower |
| Edge portability | More complex | Simplified |

**Engineering advantages:**
- Eliminates the approximately 2 GB PyTorch dependency footprint.
- Targets inference memory usage below 150 MB.
- Provides optimized CPU execution through ONNX Runtime.
- Supports deployment without a dedicated GPU.
- Enables a portable model-serving architecture.

The 2x–3x CPU inference improvement is a target performance advantage, not a guaranteed benchmark across hardware. Actual speed depends on the CPU, input resolution, preprocessing, and runtime configuration.

### Why Flask?

Flask provides a lightweight HTTP application layer for model inference and frontend delivery.

**Design decisions:**
- Minimal backend framework overhead.
- Load the ONNX model once during application startup.
- Use a single Gunicorn worker (`--workers 1`) to avoid duplicating model memory.
- Expose a JSON-based `/predict` inference endpoint.
- Serve the live HTML5 Canvas webcam interface from the same application.
- Return bounding box coordinates and class predictions instead of transmitting processed video streams.

### Why Render?

Render provides a constrained cloud deployment environment for testing CPU-based computer vision applications.

**Deployment objectives:**
- Target the 512 MB RAM Free Tier.
- Avoid GPU infrastructure.
- Keep model loading and application startup lightweight.
- Render the detection interface in the browser.
- Transfer lightweight JSON detection results instead of heavy video streams.

The architecture demonstrates how computer vision inference can be separated from video rendering, reducing server-side processing and bandwidth requirements.

Note: Free-tier availability, sleeping behavior, cold starts, and resource limits are subject to Render's current plan policies. Continuous real-time availability is not guaranteed on a sleeping free instance.

---

## 4. Model Performance & Metrics

### Validation Results — 25 Epochs

| Metric | Result |
|---|---:|
| Overall mAP50 | **83.4%** |
| Hole mAP50 | 92.3% |
| Oil mAP50 | 91.7% |
| Snag mAP50 | 85.7% |
| Slub mAP50 | 64.1% |
| Inference Speed | **7.6 ms** |

### Supported Defect Classes

| Class | Description |
|---|---|
| `hole` | Holes or missing fabric material |
| `oil` | Oil stains or contamination |
| `slub` | Irregular thick yarn or uneven fabric texture |
| `snag` | Pulled or hooked yarn defects |

**Performance observations:**
- Hole and oil demonstrate the strongest validation performance.
- Snag detection achieves 85.7% mAP50.
- Slub detection is the primary improvement target at 64.1%.
- Overall validation mAP50 reaches 83.4%.

Inference speed is hardware- and configuration-dependent. The reported 7.6 ms should be interpreted in the context of the original benchmark environment, not as a guaranteed Render latency.

---

## 5. Quickstart & Deployment

### Prerequisites

- Python 3.10+
- Git
- Webcam (for live detection)
- ONNX model file

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fabric-defect-detection.git

cd fabric-defect-detection
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the ONNX Model

Place the exported model in the configured model directory.

Example:

```text
models/
└── best.onnx
```

Ensure the model path in the Flask application matches the actual file location.

### 5. Start the Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

### 6. Deploy on Render

1. Push the project to GitHub.
2. Create a new Web Service on Render.
3. Connect the GitHub repository.
4. Configure the Python build environment.
5. Add the required environment variables, if applicable.
6. Configure the following start command:

**Gunicorn Start Command**

```bash
gunicorn app:app --workers 1 --threads 2 --bind 0.0.0.0:$PORT
```

The single-worker configuration helps avoid loading multiple copies of the ONNX model into memory.

---

## 6. Future Roadmap

### Edge & Mobile Rollout

Extend inference beyond cloud-hosted Flask deployment.

Planned improvements:

- Quantized ONNX models for lower memory consumption.
- TFLite deployment for compatible edge devices.
- ONNX Runtime Web for browser-based inference.
- Factory-floor deployment on low-end CPU hardware.
- Mobile browser support.
- Reduced inference latency through optimized preprocessing and model execution.

### Automated Event Pipeline — Jev Integration

Integrate the detection engine with **Jev** to convert model predictions into automated inspection events.

**Planned workflow:**

```text
Fabric Image / Live Webcam
          |
          v
    YOLO26n ONNX
          |
          v
    Defect Detection
          |
          v
 Confidence Threshold
          |
          v
     Jev Integration
          |
          v
   Severity Scoring
          |
          v
    Event Logging
          |
          v
 Real-Time Notifications
```

**Planned capabilities:**

- Stream high-confidence defect triggers directly into Jev.
- Automatically score defect severity.
- Log inspection events with timestamps and defect classifications.
- Trigger instant notifications when damage crosses configured threshold limits.
- Maintain inspection history for quality-control analysis.
- Support automated textile quality monitoring workflows.

---

**Built for efficient computer vision. Designed for constrained hardware. Engineered for edge deployment.**
