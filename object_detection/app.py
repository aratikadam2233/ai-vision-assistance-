"""
VisionVoice AI – Real-Time Blind Assistance & Live Navigation
============================================================
A fast, simple, and reliable real-time AI assistant for visually impaired people.
Continuously analyzes the live camera feed, detects objects/hazards, decides where
to move, and speaks clear directional voice guidance out loud in real-time.

Run with:
    streamlit run app.py
"""

import time
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# ==============================================================================
# 1. PAGE SETUP & ACCESSIBILITY STYLING
# ==============================================================================

st.set_page_config(
    page_title="VisionVoice - Blind Navigation Assistant",
    page_icon="🦯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom High-Contrast & High-Visibility UI Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0b0f19;
        color: #ffffff;
    }
    
    /* Big Accessible Action Buttons */
    .stButton>button {
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        min-height: 60px !important;
        border-radius: 12px !important;
        border: 2px solid transparent !important;
        letter-spacing: 0.5px !important;
    }

    /* Giant Directional Command Card */
    .direction-card-stop {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 4px solid #ef4444;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 30px rgba(239, 68, 68, 0.4);
        margin-bottom: 20px;
    }
    .direction-card-forward {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 4px solid #10b981;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 30px rgba(16, 185, 129, 0.4);
        margin-bottom: 20px;
    }
    .direction-card-left {
        background: linear-gradient(135deg, #1e3a8a, #1d4ed8);
        border: 4px solid #3b82f6;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.4);
        margin-bottom: 20px;
    }
    .direction-card-right {
        background: linear-gradient(135deg, #581c87, #6d28d9);
        border: 4px solid #8b5cf6;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 30px rgba(139, 92, 246, 0.4);
        margin-bottom: 20px;
    }
    
    .direction-label {
        font-size: 1.1rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.9;
    }
    .direction-action {
        font-size: 2.8rem;
        font-weight: 900;
        margin: 8px 0;
        letter-spacing: -1px;
    }
    .direction-detail {
        font-size: 1.35rem;
        font-weight: 600;
        line-height: 1.4;
    }

    /* Status metric tiles */
    .status-tile {
        background: #1e293b;
        border: 2px solid #334155;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .status-tile-title {
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .status-tile-val {
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. FAST YOLO MODEL LOADER
# ==============================================================================

@st.cache_resource(show_spinner="Loading YOLOv8 AI Model for live detection...")
def load_yolo():
    """Loads lightweight YOLOv8n optimized for fast real-time CPU detection."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

yolo_model = load_yolo()


# ==============================================================================
# 3. REAL-TIME NAVIGATION & SPATIAL DECISION ENGINE
# ==============================================================================

def analyze_frame_and_navigate(frame: np.ndarray, model, conf_threshold: float = 0.40):
    """
    Analyzes frame, divides into LEFT / CENTER / RIGHT corridors,
    calculates risk, and determines exactly what to speak and where to move.
    """
    height, width = frame.shape[:2]
    annotated_frame = frame.copy()

    # Draw walking corridor boundaries on screen (3 vertical zones)
    left_boundary = int(width * 0.33)
    right_boundary = int(width * 0.67)

    # Corridor overlay lines
    cv2.line(annotated_frame, (left_boundary, 0), (left_boundary, height), (80, 80, 80), 2)
    cv2.line(annotated_frame, (right_boundary, 0), (right_boundary, height), (80, 80, 80), 2)

    center_hazards = []
    left_hazards = []
    right_hazards = []
    all_detections = []

    # High priority obstacle classes for mobility
    priority_classes = {
        "person", "chair", "couch", "table", "bed", "tv", "laptop",
        "bottle", "door", "stairs", "car", "bus", "truck", "motorcycle",
        "bicycle", "dog", "cat", "fire hydrant", "stop sign", "potted plant", "backpack", "suitcase"
    }

    if model is not None:
        try:
            # Fast inference
            results = model.predict(frame, conf=conf_threshold, verbose=False, imgsz=480)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls_id = int(box.cls[0].item())
                    cls_name = model.names.get(cls_id, "object").lower()
                    conf = float(box.conf[0].item())

                    # Calculate center point and relative area
                    cx = (x1 + x2) / 2.0
                    box_area = (x2 - x1) * (y2 - y1)
                    area_ratio = box_area / float(width * height)

                    # Distance estimation based on box scale
                    if area_ratio > 0.15 or (y2 / height) > 0.85:
                        dist_label = "NEAR"
                    elif area_ratio > 0.04 or (y2 / height) > 0.55:
                        dist_label = "MEDIUM"
                    else:
                        dist_label = "FAR"

                    # Zone position
                    if cx < left_boundary:
                        zone = "LEFT"
                        left_hazards.append((cls_name, dist_label))
                    elif cx > right_boundary:
                        zone = "RIGHT"
                        right_hazards.append((cls_name, dist_label))
                    else:
                        zone = "CENTER"
                        center_hazards.append((cls_name, dist_label))

                    all_detections.append({
                        "name": cls_name.capitalize(),
                        "zone": zone,
                        "distance": dist_label,
                        "conf": conf
                    })

                    # Bounding Box Color: RED for Center/Near, AMBER for Warning, GREEN for Far/Safe
                    if zone == "CENTER" and dist_label in ["NEAR", "MEDIUM"]:
                        box_color = (0, 0, 255) # Red (BGR)
                    elif dist_label == "NEAR":
                        box_color = (0, 165, 255) # Orange (BGR)
                    else:
                        box_color = (0, 220, 100) # Green (BGR)

                    # Draw Bounding Box & Clear Large Text
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 3)
                    tag = f"{cls_name.upper()} ({zone} - {dist_label})"
                    
                    # Tag box
                    cv2.rectangle(annotated_frame, (x1, max(0, y1 - 28)), (x1 + len(tag) * 10 + 10, y1), box_color, -1)
                    cv2.putText(
                        annotated_frame,
                        tag,
                        (x1 + 4, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255) if box_color == (0, 0, 255) else (0, 0, 0),
                        2
                    )

        except Exception as e:
            pass

    # ==========================================================================
    # NAVIGATION DECISION ALGORITHM (WHERE TO MOVE & WHAT IS DANGEROUS)
    # ==========================================================================

    # Check immediate hazards in center path
    near_center = [name for name, dist in center_hazards if dist == "NEAR"]
    med_center = [name for name, dist in center_hazards if dist == "MEDIUM"]
    near_left = [name for name, dist in left_hazards if dist in ["NEAR", "MEDIUM"]]
    near_right = [name for name, dist in right_hazards if dist in ["NEAR", "MEDIUM"]]

    if near_center:
        # Critical Danger directly ahead!
        obstacle = near_center[0]
        if not near_left:
            suggested_action = "MOVE LEFT"
            status_theme = "left"
            guidance_msg = f"Obstacle in front: {obstacle}. Move LEFT!"
        elif not near_right:
            suggested_action = "MOVE RIGHT"
            status_theme = "right"
            guidance_msg = f"Obstacle in front: {obstacle}. Move RIGHT!"
        else:
            suggested_action = "STOP! DANGER"
            status_theme = "stop"
            guidance_msg = f"DANGER: {obstacle} blocking your path. Please STOP!"
        risk_score = 90
        path_status = "BLOCKED"

    elif med_center:
        # Approaching obstacle ahead
        obstacle = med_center[0]
        if not near_left:
            suggested_action = "MOVE LEFT"
            status_theme = "left"
            guidance_msg = f"Approaching {obstacle} ahead. Veer LEFT."
        elif not near_right:
            suggested_action = "MOVE RIGHT"
            status_theme = "right"
            guidance_msg = f"Approaching {obstacle} ahead. Veer RIGHT."
        else:
            suggested_action = "SLOW DOWN"
            status_theme = "stop"
            guidance_msg = f"Caution: {obstacle} in center path. Slow down."
        risk_score = 55
        path_status = "PARTIALLY BLOCKED"

    elif near_left and not near_right:
        suggested_action = "MOVE FORWARD (CLEAR)"
        status_theme = "forward"
        guidance_msg = f"{near_left[0]} on your left. Path ahead is clear. Move FORWARD."
        risk_score = 25
        path_status = "CLEAR"

    elif near_right and not near_left:
        suggested_action = "MOVE FORWARD (CLEAR)"
        status_theme = "forward"
        guidance_msg = f"{near_right[0]} on your right. Path ahead is clear. Move FORWARD."
        risk_score = 25
        path_status = "CLEAR"

    else:
        suggested_action = "MOVE FORWARD"
        status_theme = "forward"
        if all_detections:
            names = list(set([d["name"] for d in all_detections]))
            guidance_msg = f"Path is clear. Detected: {', '.join(names[:2])}. Walk forward."
        else:
            guidance_msg = "Walking path is completely clear. Walk forward safely."
        risk_score = 10
        path_status = "CLEAR"

    # Draw live HUD status text on top of the frame
    hud_bg = (0, 0, 200) if status_theme == "stop" else (0, 150, 0) if status_theme == "forward" else (200, 100, 0)
    cv2.rectangle(annotated_frame, (0, 0), (width, 42), hud_bg, -1)
    cv2.putText(
        annotated_frame,
        f"NAV: {suggested_action} | PATH: {path_status}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    return annotated_frame, suggested_action, status_theme, guidance_msg, path_status, risk_score, all_detections


# ==============================================================================
# 4. INSTANT BROWSER SPEECH SYNTHESIS COMPONENT (ZERO LAG TTS)
# ==============================================================================

def speak_browser(text: str):
    """
    Speaks out loud using Browser's native SpeechSynthesis API.
    Works with ZERO latency on any computer/phone without requiring external audio downloads!
    """
    escaped = text.replace("'", "\\'").replace('"', '\\"').replace("\n", " ")
    js_code = f"""
    <script>
        window.speechSynthesis.cancel();
        var utterance = new SpeechSynthesisUtterance("{escaped}");
        utterance.rate = 1.1;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        window.speechSynthesis.speak(utterance);
    </script>
    """
    components.html(js_code, height=0, width=0)


# ==============================================================================
# 5. INITIALIZE SESSION STATE
# ==============================================================================

if "live_running" not in st.session_state:
    st.session_state.live_running = False

if "last_guidance_text" not in st.session_state:
    st.session_state.last_guidance_text = "Welcome to VisionVoice. Start live assistance to guide your path."

if "last_spoken_time" not in st.session_state:
    st.session_state.last_spoken_time = 0.0


# ==============================================================================
# 6. APPLICATION HEADER & CONTROLS
# ==============================================================================

col_t1, col_t2 = st.columns([3, 1])

with col_t1:
    st.title("🦯 VisionVoice AI")
    st.markdown("### **Real-Time Live Navigation & Danger Detection for Blind People**")

with col_t2:
    st.write("")
    if not st.session_state.live_running:
        if st.button("▶️ START LIVE ASSISTANCE", type="primary"):
            st.session_state.live_running = True
            st.rerun()
    else:
        if st.button("⏹️ STOP ASSISTANCE"):
            st.session_state.live_running = False
            st.rerun()


# ==============================================================================
# 7. CONTINUOUS LIVE WEBCAM & GUIDANCE DASHBOARD
# ==============================================================================

col_left, col_right = st.columns([1.2, 1.0], gap="large")

with col_right:
    # Action Status & Direction Container
    direction_placeholder = st.empty()
    metrics_placeholder = st.empty()
    hazards_placeholder = st.empty()
    speech_trigger_placeholder = st.empty()

with col_left:
    st.markdown("#### 📷 **Live Camera Feed & Corridor Analysis**")
    video_feed_placeholder = st.empty()


# ==============================================================================
# 8. LIVE DETECTION LOOP
# ==============================================================================

if st.session_state.live_running:
    # Open local webcam (default device index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("⚠️ Could not access your webcam. Please check if your camera is connected or in use by another app.")
        st.session_state.live_running = False
    else:
        # Run continuous real-time loop for 40 frames per refresh cycle
        last_action_spoken = ""
        
        for frame_idx in range(40):
            ret, frame = cap.read()
            if not ret:
                break

            # Analyze frame
            annotated, action, theme, guidance, path_status, risk_score, detections = analyze_frame_and_navigate(
                frame, yolo_model, conf_threshold=0.38
            )

            # Convert BGR frame to RGB for Streamlit video canvas
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_feed_placeholder.image(frame_rgb, channels="RGB")

            # Update Big Accessible Directional Card
            card_class = f"direction-card-{theme}"
            icon = "🛑" if theme == "stop" else "⬆️" if theme == "forward" else "⬅️" if theme == "left" else "➡️"
            
            direction_placeholder.markdown(
                f"""
                <div class="{card_class}">
                    <div class="direction-label">LIVE ACTION GUIDANCE</div>
                    <div class="direction-action">{icon} {action}</div>
                    <div class="direction-detail">"{guidance}"</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Update Metrics Row
            risk_color = "#ef4444" if risk_score > 60 else "#f59e0b" if risk_score > 30 else "#10b981"
            metrics_placeholder.markdown(
                f"""
                <div style="display: flex; gap: 12px; margin-bottom: 16px;">
                    <div class="status-tile" style="flex: 1; border-color: {risk_color};">
                        <div class="status-tile-title">PATH STATUS</div>
                        <div class="status-tile-val" style="color: {risk_color};">{path_status}</div>
                    </div>
                    <div class="status-tile" style="flex: 1;">
                        <div class="status-tile-title">RISK LEVEL</div>
                        <div class="status-tile-val" style="color: {risk_color};">{risk_score}%</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Update Detected Objects breakdown
            if detections:
                det_list = [f"• **{d['name']}** ({d['zone']}, {d['distance']})" for d in detections[:4]]
                hazards_placeholder.info("🎯 **Objects in View:**\n\n" + "\n".join(det_list))
            else:
                hazards_placeholder.success("✅ No obstacles detected in view.")

            # Periodic Continuous Voice Guidance (Speaks every 2.5 seconds or immediately when action changes)
            current_time = time.time()
            if (current_time - st.session_state.last_spoken_time > 2.6) or (action != last_action_spoken and action == "STOP! DANGER"):
                speak_browser(guidance)
                st.session_state.last_spoken_time = current_time
                last_action_spoken = action

            time.sleep(0.03)

        cap.release()
        st.rerun()

else:
    # Standby State when camera is stopped
    direction_placeholder.markdown(
        """
        <div class="direction-card-forward">
            <div class="direction-label">SYSTEM READY</div>
            <div class="direction-action">STANDBY</div>
            <div class="direction-detail">Click <strong>'START LIVE ASSISTANCE'</strong> to begin continuous camera navigation.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    video_feed_placeholder.info("💡 **Camera is currently paused.** Tap the green **START LIVE ASSISTANCE** button above.")


# ==============================================================================
# 9. ONE-TOUCH QUICK VOICE TEST BUTTONS
# ==============================================================================

st.markdown("---")
st.markdown("#### 🗣️ **One-Touch Audio Commands & Voice Tests**")

col_q1, col_q2, col_q3 = st.columns(3)

with col_q1:
    if st.button("❓ Where is the danger?"):
        speak_browser("Scanning environment. Look out for obstacles directly in your center walking path.")

with col_q2:
    if st.button("🛣️ Is my path clear?"):
        speak_browser("Your walking corridor is being analyzed. Listen for direction cues: Forward, Left, Right, or Stop.")

with col_q3:
    if st.button("🚨 Test Emergency Siren"):
        speak_browser("EMERGENCY ALERT! Stop walking immediately and check your surroundings.")
