# 🦯 VisionVoice AI – Real-Time Blind Assistance & Live Navigation

VisionVoice AI is a real-time AI-powered vision and voice assistance system designed to help visually impaired people identify obstacles and receive simple navigation instructions.

The system uses a webcam to continuously capture the surroundings, **YOLOv8n** to detect objects, spatial analysis to determine whether an object is on the left, center, or right, and a rule-based navigation engine to provide guidance such as **Move Forward, Move Left, Move Right, Slow Down, or Stop**.

---

## 🎯 Problem Statement

Visually impaired people face difficulties identifying obstacles and understanding their surroundings while walking. Traditional walking aids can detect physical obstacles but cannot provide detailed information about the type, position, or relative distance of objects.

VisionVoice AI aims to provide an intelligent assistance system that detects surrounding objects and provides real-time voice-based navigation guidance.

---

## 💡 Proposed Solution

The system follows this pipeline:

```text
Live Camera
     ↓
OpenCV Frame Capture
     ↓
YOLOv8n Object Detection
     ↓
Bounding Box & Confidence
     ↓
Left / Center / Right Analysis
     ↓
Near / Medium / Far Estimation
     ↓
Rule-Based Risk Analysis
     ↓
Navigation Decision
     ↓
Browser Voice Guidance
```

---

## 🚀 Features

* 📷 Real-time webcam monitoring
* 🤖 YOLOv8n object detection
* 🎯 Bounding-box visualization
* 📍 Left / Center / Right spatial analysis
* 📏 Relative Near / Medium / Far estimation
* ⚠️ Rule-based risk assessment
* 🛑 Emergency STOP detection
* ⬅️ Move Left guidance
* ➡️ Move Right guidance
* ⬆️ Move Forward guidance
* 🔊 Browser-based voice output
* 📊 Live risk and path-status display
* ♿ Accessibility-focused high-contrast interface
* 🖥️ Streamlit-based web interface

---

## 🧠 Technologies Used

| Technology                     | Purpose                      |
| ------------------------------ | ---------------------------- |
| Python                         | Main programming language    |
| YOLOv8n                        | Object detection             |
| OpenCV                         | Webcam and image processing  |
| NumPy                          | Image/frame array processing |
| Streamlit                      | Web application interface    |
| JavaScript SpeechSynthesis API | Voice output                 |
| HTML/CSS                       | Accessibility and UI styling |

---

## 🔍 Object Detection

VisionVoice AI uses the lightweight **YOLOv8n pretrained model**.

YOLO stands for:

> **You Only Look Once**

The model detects objects and provides:

* Object class
* Bounding box
* Confidence score

Example:

```text
Person — 0.91
Chair — 0.84
Bottle — 0.76
```

The Nano (`n`) version is selected because the application requires fast inference for a real-time webcam demonstration.

---

## 📍 Spatial Analysis

The camera frame is divided into three zones:

```text
┌────────────┬────────────┬────────────┐
│            │            │            │
│    LEFT    │   CENTER   │   RIGHT    │
│            │            │            │
└────────────┴────────────┴────────────┘
```

The center point of each detected object's bounding box is used to determine its zone.

```python
cx = (x1 + x2) / 2
```

The system then compares `cx` with the left and right boundaries.

---

## 📏 Relative Distance Estimation

The prototype estimates relative distance using the size and position of the bounding box.

```text
Large bounding box   → NEAR
Medium bounding box  → MEDIUM
Small bounding box   → FAR
```

The system does **not** calculate exact physical distance in meters.

This is a heuristic estimation intended for the prototype.

---

## 🧭 Navigation Algorithm

The navigation engine uses predefined rules.

### Example:

If a near obstacle is detected in the center:

```text
Center blocked?
       ↓
 Is Left clear?
   ↓         ↓
 YES         NO
 ↓           ↓
LEFT     Is Right clear?
             ↓
          YES / NO
           ↓    ↓
         RIGHT STOP
```

### Navigation outputs

| Situation                     | Action                       |
| ----------------------------- | ---------------------------- |
| Clear path                    | MOVE FORWARD                 |
| Center obstacle + left clear  | MOVE LEFT                    |
| Center obstacle + right clear | MOVE RIGHT                   |
| Medium center obstacle        | SLOW DOWN / change direction |
| Center + both sides blocked   | STOP! DANGER                 |

---

## ⚠️ Risk Assessment

The current prototype uses a rule-based risk score.

| Situation                    | Risk Score |
| ---------------------------- | ---------: |
| Near obstacle in center      |        90% |
| Medium obstacle in center    |        55% |
| Near/medium obstacle on side |        25% |
| Clear path                   |        10% |

These values are predefined rules and are **not a machine-learning prediction of probability**.

---

## 🔊 Voice Guidance

The generated navigation message is converted into speech using the browser's native **SpeechSynthesis API**.

Example:

```text
YOLO detects:
Person → CENTER → NEAR

        ↓

Navigation Engine

        ↓

"Obstacle in front: person. Move LEFT!"

        ↓

Browser SpeechSynthesis

        ↓

🔊 User hears the instruction
```

Voice announcements are controlled with a short time interval to avoid continuously repeating the same instruction.

---

## 🖥️ User Interface

The Streamlit interface provides:

* Live camera feed
* Object detection boxes
* Navigation direction
* Path status
* Risk level
* Detected objects
* Voice testing buttons
* Emergency voice test

The interface uses large, high-contrast components for accessibility and demonstration purposes.

---

## 📊 Evaluation Metrics

Since the current implementation uses a pretrained YOLO model, a new project-specific accuracy percentage has not been calculated.

Standard object-detection evaluation metrics include:

* Precision
* Recall
* mAP@50
* mAP@50–95
* Inference time
* FPS

For the navigation component, useful evaluation metrics would include:

* Navigation success rate
* Missed obstacle rate
* False warning rate
* Response time
* FPS

A proper project-specific evaluation would require a labeled test dataset.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd VisionVoice-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install streamlit opencv-python numpy ultralytics
```

---

## ▶️ Run the Application

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

Allow webcam access and click:

```text
▶️ START LIVE ASSISTANCE
```

---

## 📁 Project Structure

```text
VisionVoice-AI/
│
├── app.py
├── yolov8n.pt
├── requirements.txt
├── README.md
└── assets/
    └── screenshots/
```

---

## 🔬 Algorithms Used

### 1. YOLO Object Detection

Used for detecting objects in real-time.

### 2. Spatial Zone Algorithm

Divides the camera view into:

```text
LEFT | CENTER | RIGHT
```

### 3. Relative Distance Heuristic

Classifies detected objects as:

```text
NEAR | MEDIUM | FAR
```

### 4. Rule-Based Navigation Algorithm

Uses object position and relative distance to determine:

```text
FORWARD
LEFT
RIGHT
SLOW DOWN
STOP
```

---

## ⚠️ Limitations

The current prototype has some limitations:

1. Distance estimation is relative rather than exact.
2. Detection performance depends on lighting and camera quality.
3. The pretrained model may not recognize every mobility-specific hazard.
4. The current navigation system uses predefined rules.
5. Streamlit is suitable for the prototype/demo but is not an ideal production-grade real-time video architecture.
6. The system should not be considered a replacement for established mobility aids or human assistance.

---

## 🔮 Future Scope

Future improvements can include:

* Custom YOLO training for mobility-specific hazards
* Pothole detection
* Stair detection
* Door detection
* Depth estimation
* Object tracking
* Collision prediction
* GPS-based outdoor navigation
* Mobile application
* Edge-device deployment
* Multilingual voice guidance
* Personalized navigation
* More advanced risk prediction

---

## 🎓 Project Objective

The main objective of VisionVoice AI is to demonstrate how **computer vision, artificial intelligence, spatial reasoning, and speech synthesis** can be combined to create an assistive technology prototype for visually impaired users.

---

## 👩‍💻 Project Type

**Artificial Intelligence / Deep Learning / Computer Vision / Assistive Technology**

---

## 📌 Disclaimer

VisionVoice AI is an academic/prototype project intended for demonstration and research purposes. It should not be relied upon as the sole safety system for real-world mobility.

---

## ⭐ Conclusion

VisionVoice AI combines:

```text
Computer Vision
       +
YOLO Object Detection
       +
Spatial Analysis
       +
Rule-Based Navigation
       +
Voice Assistance
```

to create a prototype real-time assistant that can detect surrounding objects and communicate navigation guidance to the user.
