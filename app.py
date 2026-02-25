"""
AI Surveillance System - Standalone Application
FastAPI backend + GradCAM++ + LIME + Local RAG (FAISS)
Auto-opens browser on startup
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import base64
from pathlib import Path
import time
from datetime import datetime
import json
import os
import threading
import webbrowser

from storage_manager import StorageManager
from sentence_transformers import SentenceTransformer
import faiss
from lime import lime_image
from skimage.segmentation import mark_boundaries

# ============================================================================
# CONFIGURATION
# ============================================================================

YOLO_MODEL_PATH   = "03_models/yolo_medium_improved/weights/best.pt"
RESNET_MODEL_PATH = "03_models/resnet_verifier/resnet_verifier_best.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Device]  Using device: {device}")

storage = StorageManager()

# Global state
current_session_id     = None
current_chat_history   = []
current_detections     = None
current_original_image = None
current_detected_image = None
current_stats          = None
current_xai_data       = []

# ============================================================================
# RAG / FAISS SETUP
# ============================================================================

print("[Loading] Loading embedding model...")
embed_model    = SentenceTransformer('all-MiniLM-L6-v2')
dimension      = 384
index          = faiss.IndexFlatL2(dimension)
metadata_store = []

DOMAIN_KNOWLEDGE = [
    "Very High Confidence 90 percent or above means the system is highly certain this is a person. Both YOLO and ResNet agree strongly. GradCAM++ shows clear activation on human features like head shoulders and body outline. Trust this detection.",
    "Medium Confidence between 70 and 89 percent means the system detects a person but with some uncertainty. This can occur due to partial occlusion camouflage effectiveness or background interference. Manual review suggested.",
    "Low Confidence below 70 percent means weak detection signals. Significant background noise or heavy camouflage. High chance of false positive. Verification required.",
    "The detection pipeline works in two stages. Stage 1 is YOLO Detection which is fast object detection to identify potential persons trained on camouflaged person dataset. Stage 2 is ResNet Verification which is deep learning verification using ResNet50 to confirm each YOLO detection and filter false positives.",
    "GradCAM++ is an improved version of Grad-CAM that provides better localization of important regions. It uses second-order gradients to compute more accurate class activation maps. Red and yellow areas indicate high importance regions that the model focused on.",
    "LIME stands for Local Interpretable Model-agnostic Explanations. It explains predictions by perturbing the input image and observing how the model output changes. Green highlighted regions in LIME indicate features that support the detection. Red regions indicate features against the detection.",
    "High confidence detections occur when there is a clear human silhouette distinct edge patterns no occlusion good lighting texture patterns matching human clothing and strong GradCAM++ activation on key features.",
    "Low confidence detections occur when there is partial occlusion by trees or bushes effective camouflage blending with background poor lighting small or distant target weak activation or conflicting signals between YOLO and ResNet.",
    "YOLO alone has 15 to 20 percent false positive rate. ResNet verification reduces this to about 5 percent. Common false positives include tree stumps rock formations shadows dense vegetation and mannequins.",
    "Threat levels are Critical for confidence above 90 percent High for 80 to 90 percent Medium for 70 to 80 percent and Low for 50 to 70 percent.",
    "GradCAM++ red and yellow regions mean high importance for the detection decision. Green means moderate importance. Blue and purple mean low importance or background. If GradCAM++ highlights head shoulders and torso it means strong human features were detected.",
    "The combined score is the average of YOLO score and ResNet score. It represents the overall detection confidence. A combined score above 85 percent is considered reliable.",
    "ResNet50 is used as the verification model. It takes each cropped detection from YOLO and classifies it as a person or not a person. This two stage approach significantly reduces false positives compared to using YOLO alone.",
    "LIME explanation shows which image segments or superpixels were most important for the model decision. Positive regions shown in green support the person classification. Negative regions shown in red contradict the classification.",
    "When GradCAM++ and LIME agree on the same regions it means the model is focusing on genuine human features. When they disagree it may indicate the model is using texture or background cues rather than actual human body parts.",
]

print("[Docs] Loading domain knowledge into FAISS index...")
for text in DOMAIN_KNOWLEDGE:
    emb = embed_model.encode([text])
    index.add(np.array(emb).astype('float32'))
    metadata_store.append(text)
print(f"[OK] Loaded {len(DOMAIN_KNOWLEDGE)} knowledge chunks into FAISS")

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class ResNetVerifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetVerifier, self).__init__()
        self.resnet = models.resnet50(weights=None)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


# ============================================================================
# GRAD-CAM++ IMPLEMENTATION
# ============================================================================

class GradCAMPlusPlus:
    """
    Grad-CAM++: Improved Visual Explanations for Deep CNNs
    Uses second-order gradients for better localization
    """
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        gradients   = self.gradients[0]
        activations = self.activations[0]

        # GradCAM++ alpha computation (second-order gradients)
        grad_sq   = gradients ** 2
        grad_cube = gradients ** 3
        sum_act   = activations.sum(dim=(1, 2), keepdim=True)
        denom     = 2 * grad_sq + grad_cube * sum_act
        denom     = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha     = grad_sq / denom

        relu_grad = F.relu(score.exp() * gradients)
        weights   = (alpha * relu_grad).sum(dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], device=input_tensor.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam.cpu().detach().numpy()


# ============================================================================
# LIME EXPLAINER
# ============================================================================

class LIMEExplainer:
    """LIME: Local Interpretable Model-agnostic Explanations"""

    def __init__(self, model, transform, device):
        self.model     = model
        self.transform = transform
        self.device    = device
        self.explainer = lime_image.LimeImageExplainer()

    def predict_fn(self, images):
        batch = torch.cat([
            self.transform(Image.fromarray(img.astype(np.uint8))).unsqueeze(0)
            for img in images
        ], dim=0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(batch), dim=1)
        return probs.cpu().numpy()

    def explain(self, crop_rgb, num_samples=80, num_features=5):
        try:
            explanation = self.explainer.explain_instance(
                crop_rgb.astype(np.uint8),
                self.predict_fn,
                top_labels=2,
                hide_color=0,
                num_samples=num_samples,
            )
            temp, mask = explanation.get_image_and_mask(
                label=1,
                positive_only=False,
                num_features=num_features,
                hide_rest=False
            )
            overlay = mark_boundaries(temp / 255.0, mask)
            return (overlay * 255).astype(np.uint8), mask
        except Exception as e:
            print(f"[WARNING] LIME error: {e}")
            return crop_rgb, None


# ============================================================================
# INTEGRATED DETECTOR
# ============================================================================

class IntegratedDetector:
    def __init__(self, yolo_path, resnet_path):
        print(" Loading models...")
        self.yolo_model = YOLO(yolo_path)
        print("[OK] YOLO loaded")

        self.resnet_model = ResNetVerifier(num_classes=2).to(device)
        self.resnet_model.load_state_dict(torch.load(resnet_path, map_location=device))
        self.resnet_model.eval()
        print("[OK] ResNet loaded")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        target_layer   = self.resnet_model.resnet.layer4[-1]
        self.gradcampp = GradCAMPlusPlus(self.resnet_model, target_layer)
        self.lime      = LIMEExplainer(self.resnet_model, self.transform, device)
        print("[OK] GradCAM++ + LIME ready!")

    def detect(self, image, yolo_conf=0.4, resnet_conf=0.5):
        start_time   = time.time()
        img          = image.copy() if not isinstance(image, str) else cv2.imread(image)
        yolo_results = self.yolo_model(img, conf=yolo_conf, verbose=False)
        yolo_boxes   = yolo_results[0].boxes
        yolo_count   = len(yolo_boxes)

        verified_detections = []
        all_detections      = []  # ALL YOLO detections (for drawing)
        filtered_count      = 0
        xai_data            = []

        for box in yolo_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            yolo_score      = float(box.conf[0])
            crop_bgr        = img[y1:y2, x1:x2]

            if crop_bgr.size == 0:
                filtered_count += 1
                continue

            crop_rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_pil    = Image.fromarray(crop_rgb)
            crop_tensor = self.transform(crop_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output       = self.resnet_model(crop_tensor)
                probs        = torch.softmax(output, dim=1)
                resnet_score = float(probs[0, 1])

            # Add to ALL detections list (for visualization)
            all_detections.append({
                'bbox':           [x1, y1, x2, y2],
                'yolo_score':     yolo_score,
                'resnet_score':   resnet_score,
                'combined_score': (yolo_score + resnet_score) / 2,
                'verified':       resnet_score >= resnet_conf
            })

            if resnet_score >= resnet_conf:
                # GradCAM++
                cam             = self.gradcampp.generate_cam(crop_tensor, class_idx=1)
                cam_resized     = cv2.resize(cam, (crop_bgr.shape[1], crop_bgr.shape[0]))
                heatmap_col     = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                gradcampp_overlay = cv2.addWeighted(crop_bgr, 0.5, heatmap_col, 0.5, 0)

                # LIME
                lime_overlay, _ = self.lime.explain(crop_rgb, num_samples=80, num_features=5)
                lime_bgr        = cv2.cvtColor(lime_overlay, cv2.COLOR_RGB2BGR)

                verified_detections.append({
                    'bbox':           [x1, y1, x2, y2],
                    'yolo_score':     yolo_score,
                    'resnet_score':   resnet_score,
                    'combined_score': (yolo_score + resnet_score) / 2
                })
                xai_data.append({
                    'person_id':      len(verified_detections),
                    'gradcampp':      gradcampp_overlay,
                    'lime':           lime_bgr,
                    'crop':           crop_bgr,
                    'resnet_score':   resnet_score,
                    'yolo_score':     yolo_score,
                    'combined_score': (yolo_score + resnet_score) / 2,
                    'bbox':           [x1, y1, x2, y2],
                    'activation_peak': float(cam.max()),
                })
            else:
                filtered_count += 1

        statistics = {
            'yolo_detections':     yolo_count,
            'verified_detections': len(verified_detections),
            'filtered_out':        filtered_count,
            'process_time':        time.time() - start_time
        }
        return verified_detections, statistics, img, xai_data, all_detections

    def visualize_detections(self, image, all_detections):
        """Draw ALL detections - verified get solid boxes, unverified get dashed red"""
        result = image.copy()
        verified_count = 0
        
        for det in all_detections:
            x1, y1, x2, y2 = det['bbox']
            
            if det['verified']:
                # Verified detection - solid box
                verified_count += 1
                score  = det['combined_score']
                color  = (0, 255, 0)   if det['resnet_score'] >= 0.9 else \
                         (0, 165, 255) if det['resnet_score'] >= 0.7 else \
                         (0, 255, 255)  # Yellow for 0.5-0.7
                cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
                label  = f"Person {verified_count}: {score*100:.1f}%"
                lsz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(result, (x1, y1-lsz[1]-10), (x1+lsz[0], y1), color, -1)
                cv2.putText(result, label, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                # Unverified - dashed red box
                color = (0, 0, 255)  # Red
                # Draw dashed rectangle
                dash_length = 10
                for i in range(x1, x2, dash_length * 2):
                    cv2.line(result, (i, y1), (min(i+dash_length, x2), y1), color, 2)
                    cv2.line(result, (i, y2), (min(i+dash_length, x2), y2), color, 2)
                for i in range(y1, y2, dash_length * 2):
                    cv2.line(result, (x1, i), (x1, min(i+dash_length, y2)), color, 2)
                    cv2.line(result, (x2, i), (x2, min(i+dash_length, y2)), color, 2)
                
                label  = f"Unverified: {det['resnet_score']*100:.0f}%"
                lsz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(result, (x1, y1-lsz[1]-8), (x1+lsz[0], y1), color, -1)
                cv2.putText(result, label, (x1, y1-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result


# ============================================================================
# INITIALIZE DETECTOR
# ============================================================================

print("=" * 70)
print("[Initializing] INITIALIZING AI SURVEILLANCE SYSTEM")
print("=" * 70)
detector = IntegratedDetector(YOLO_MODEL_PATH, RESNET_MODEL_PATH)
print("=" * 70)
print("[OK] SYSTEM READY!")
print("=" * 70)

# ============================================================================
# XAI EXPLANATION GENERATOR
# ============================================================================

def generate_xai_explanation(xai_entry: dict, person_idx: int) -> dict:
    score    = xai_entry['resnet_score']
    yolo_s   = xai_entry['yolo_score']
    combined = xai_entry['combined_score']
    peak     = xai_entry.get('activation_peak', 0)
    x1, y1, x2, y2 = xai_entry['bbox']
    w = x2 - x1
    h = y2 - y1

    threat_level = "HIGH"   if score >= 0.9 else \
                   "MEDIUM" if score >= 0.7 else "LOW"
    confidence_label = "Very High" if score >= 0.9 else \
                       "Medium"    if score >= 0.7 else "Low"
    trust_note = (
        "This detection is highly reliable and should be actioned immediately." if score >= 0.9 else
        "This detection is moderately reliable. Manual verification is recommended." if score >= 0.7 else
        "This detection has low confidence. Further review is required."
    )
    gradcam_desc = (
        "very strong and focused activation indicating clear human body features"  if peak >= 0.8 else
        "moderate activation across the region suggesting partial human features"   if peak >= 0.5 else
        "weak or scattered activation indicating difficulty locating key features"
    )

    # RAG retrieval
    q_emb   = embed_model.encode([f"confidence {confidence_label} GradCAM LIME explanation"])
    D, I    = index.search(np.array(q_emb).astype('float32'), 2)
    rag_ctx = " ".join([metadata_store[i] for i in I[0] if i < len(metadata_store)])

    return {
        "person_id":         person_idx,
        "threat_level":      threat_level,
        "confidence_label":  confidence_label,
        "scores": {
            "yolo":     round(yolo_s   * 100, 1),
            "resnet":   round(score    * 100, 1),
            "combined": round(combined * 100, 1),
        },
        "location": {
            "bbox":   [x1, y1, x2, y2],
            "width":  w,
            "height": h,
        },
        "gradcampp_explanation": (
            f"GradCAM++ shows {gradcam_desc}. "
            f"The heatmap highlights the most discriminative regions the ResNet model "
            f"used to classify this crop as a camouflaged person. "
            f"Peak activation value: {peak:.2f}. "
            f"Red and yellow areas indicate where the model focused most strongly."
        ),
        "lime_explanation": (
            f"LIME analysis segmented the cropped region into superpixels and tested "
            f"which ones influenced the model's decision. "
            f"Green regions positively contributed to the person classification. "
            f"Red regions worked against it. "
            f"This verifies whether the model focused on actual human body parts "
            f"rather than background textures."
        ),
        "combined_explanation": (
            f"Person {person_idx} detected at ({x1},{y1})({x2},{y2}) "
            f"[{w}x{h}px]. "
            f"Combined confidence: {combined*100:.1f}%. "
            f"YOLO flagged with {yolo_s*100:.1f}%, "
            f"ResNet confirmed with {score*100:.1f}%. "
            f"Threat level: {threat_level}. "
            f"{trust_note}"
        ),
        "rag_context":    rag_ctx[:400],
        "recommendation": trust_note,
        "timestamp":      datetime.now().isoformat(),
    }


# ============================================================================
# RULE-BASED + RAG CHATBOT
# ============================================================================

class RuleBasedResponder:
    def check(self, message: str, session_data: dict):
        msg        = message.lower()
        detections = session_data.get("detections") or []
        stats      = session_data.get("statistics") or {}

        # Greetings and casual conversation
        if any(w in msg for w in ["hello", "hi", "hey", "greetings"]):
            return "Hello! I'm your AI surveillance assistant. Upload an image and run a scan, then I can help you understand the detections."
        
        if any(w in msg for w in ["thank", "thanks", "thx"]):
            return "You're welcome! Let me know if you have any other questions about the detections."
        
        if msg.strip() in ["ok", "okay", "k", "alright", "cool"]:
            return "Great! Feel free to ask me anything about the scan results."
        
        if any(w in msg for w in ["who are you", "what are you", "introduce"]):
            return "I'm an AI assistant specialized in camouflaged person detection. I can explain detection results, confidence scores, threat levels, and XAI visualizations."
        
        if any(w in msg for w in ["good", "nice", "great", "awesome", "excellent"]) and len(msg.split()) < 4:
            return "Glad to help! Ask me anything about the detections."

        if any(w in msg for w in ["how many", "count", "total person"]):
            if stats:
                return (f"{stats.get('verified_detections', 0)} person(s) verified "
                        f"out of {stats.get('yolo_detections', 0)} YOLO detections.")
            return "No scan has been run yet."

        if "threat" in msg:
            if detections:
                return "\n".join([
                    f"Person {i+1}: {'HIGH' if d['resnet_score']>=0.9 else 'MEDIUM' if d['resnet_score']>=0.7 else 'LOW'} "
                    f"threat ({d['resnet_score']*100:.1f}%)"
                    for i, d in enumerate(detections)
                ])
            return "No detections available yet."

        if any(w in msg for w in ["statistics", "stats", "summary"]):
            if stats:
                return (f"YOLO Detections:     {stats.get('yolo_detections',0)}\n"
                        f"Verified Detections: {stats.get('verified_detections',0)}\n"
                        f"Filtered Out:        {stats.get('filtered_out',0)}\n"
                        f"Processing Time:     {stats.get('process_time',0):.2f}s")
            return "No statistics available yet."

        if "confidence" in msg and any(w in msg for w in ["what", "show", "tell"]):
            if detections:
                return "\n".join([f"Person {i+1}: {d['resnet_score']*100:.1f}%"
                                  for i, d in enumerate(detections)])
            return "No detections available yet."

        if any(w in msg for w in ["how long", "time", "speed"]):
            if stats:
                return f"Processing completed in {stats.get('process_time',0):.2f} seconds."
            return "No scan run yet."

        if any(w in msg for w in ["help", "what can", "commands"]):
            return ("I can answer:\n"
                    " How many persons detected?\n"
                    " What is the threat level?\n"
                    " Show statistics\n"
                    " What is the confidence?\n"
                    " Why was this detected?\n"
                    " What does GradCAM++ show?\n"
                    " What does LIME show?\n"
                    " Should I trust this detection?")
        return None


class LocalRAGResponder:
    def retrieve(self, query: str, top_k=3):
        q_emb = embed_model.encode([query])
        D, I  = index.search(np.array(q_emb).astype('float32'), top_k)
        return [metadata_store[i] for i in I[0] if i < len(metadata_store)]

    def generate(self, message: str, session_data: dict) -> str:
        detections = session_data.get("detections") or []
        stats      = session_data.get("statistics") or {}
        retrieved  = self.retrieve(message, top_k=3)
        lines      = []
        msg        = message.lower()

        if any(w in msg for w in ["lime", "superpixel", "segment"]):
            for r in retrieved:
                if "lime" in r.lower():
                    lines.append(r); break
        elif any(w in msg for w in ["grad", "cam", "heatmap"]):
            for r in retrieved:
                if "grad" in r.lower():
                    lines.append(r); break
        elif retrieved:
            lines.append(retrieved[0])

        if detections:
            lines.append("\n Current Detections ")
            for i, d in enumerate(detections, 1):
                s = d['resnet_score']
                level = "HIGH" if s>=0.9 else "MEDIUM" if s>=0.7 else "LOW"
                x1,y1,x2,y2 = d['bbox']
                lines.append(f"Person {i}: {level} | ResNet {s*100:.1f}% | "
                             f"YOLO {d['yolo_score']*100:.1f}% | ({x1},{y1})({x2},{y2})")
        if stats:
            lines.append(f"\n Stats  Verified: {stats.get('verified_detections',0)}/"
                        f"{stats.get('yolo_detections',0)} | "
                        f"Filtered: {stats.get('filtered_out',0)} | "
                        f"Time: {stats.get('process_time',0):.2f}s")

        return "\n".join(lines) if lines else "Please run a scan first."


class HybridChatbot:
    def __init__(self):
        self.rules = RuleBasedResponder()
        self.rag   = LocalRAGResponder()

    def respond(self, message: str, session_data: dict):
        r = self.rules.check(message, session_data)
        if r:
            return r, "rule-based"
        return self.rag.generate(message, session_data), "rag"


chatbot = HybridChatbot()

# ============================================================================
# HELPERS
# ============================================================================

def build_session_context():
    return {"session_id": current_session_id, "detections": current_detections,
            "statistics": current_stats, "chat_history": current_chat_history}

def numpy_to_base64(img_array):
    _, buf = cv2.imencode('.png', img_array)
    return base64.b64encode(buf).decode('utf-8')

def store_detection_embeddings(detections, session_id):
    for idx, det in enumerate(detections, 1):
        x1,y1,x2,y2 = det['bbox']
        text = (f"Session {session_id}: Person {idx} at ({x1},{y1})({x2},{y2}), "
                f"ResNet: {det['resnet_score']:.2f}, YOLO: {det['yolo_score']:.2f}")
        emb = embed_model.encode([text])
        index.add(np.array(emb).astype('float32'))
        metadata_store.append(text)

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="XCAM-Detect API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
async def health():
    return {"status": "online", "device": str(device),
            "session_id": current_session_id,
            "detections": len(current_detections) if current_detections else 0}


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    global current_detections, current_stats, current_original_image
    global current_detected_image, current_session_id, current_xai_data

    contents = await file.read()
    img      = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    if current_session_id is None:
        current_session_id = storage.create_session()

    detections, stats, original_img, xai_data, all_detections = detector.detect(img)

    current_detections     = detections
    current_stats          = stats
    current_original_image = original_img
    current_detected_image = detector.visualize_detections(original_img, all_detections)
    current_xai_data       = xai_data

    storage.save_scan_results(
        session_id=current_session_id, original_image=current_original_image,
        detected_image=current_detected_image, detections=detections,
        statistics=stats, chat_history=current_chat_history
    )
    store_detection_embeddings(detections, current_session_id)

    # Build XAI payload: gradcampp + lime per person
    xai_payload = [{
        "person_id": e['person_id'],
        "gradcampp": numpy_to_base64(e['gradcampp']),
        "lime":      numpy_to_base64(e['lime']),
        "crop":      numpy_to_base64(e['crop']),
    } for e in xai_data]

    return {
        "session_id":     current_session_id,
        "detected_image": numpy_to_base64(current_detected_image),
        "xai_images":     xai_payload,
        "detections":     detections,
        "statistics":     stats
    }


@app.get("/api/explain")
async def explain():
    """Generate RAG-powered explanations for all detected persons"""
    if not current_xai_data:
        return JSONResponse(status_code=400,
                            content={"error": "No XAI data. Run detection first."})
    explanations = [generate_xai_explanation(e, i+1)
                    for i, e in enumerate(current_xai_data)]
    return {"explanations": explanations, "total": len(explanations),
            "session_id": current_session_id, "timestamp": datetime.now().isoformat()}


@app.get("/api/report")
async def report():
    """Generate full report data for download"""
    if not current_detections:
        return JSONResponse(status_code=400,
                            content={"error": "No detection data. Run a scan first."})
    explanations = [generate_xai_explanation(e, i+1)
                    for i, e in enumerate(current_xai_data)]
    return {
        "report_title":   "XCAM-Detect Surveillance Report",
        "generated_at":   datetime.now().isoformat(),
        "session_id":     current_session_id,
        "device":         str(device),
        "statistics":     current_stats,
        "detections":     current_detections,
        "explanations":   explanations,
        "detected_image": numpy_to_base64(current_detected_image),
        "chat_history":   current_chat_history,
        "summary": {
            "total_persons":  len(current_detections),
            "high_threat":    sum(1 for d in current_detections if d['resnet_score'] >= 0.9),
            "medium_threat":  sum(1 for d in current_detections if 0.7 <= d['resnet_score'] < 0.9),
            "low_threat":     sum(1 for d in current_detections if d['resnet_score'] < 0.7),
            "avg_confidence": round(
                sum(d['resnet_score'] for d in current_detections) /
                len(current_detections) * 100, 1) if current_detections else 0,
        }
    }


@app.post("/api/chat")
async def chat(payload: dict):
    global current_chat_history
    user_message = payload.get("message", "").strip()
    if not user_message:
        return JSONResponse(status_code=400, content={"error": "Empty message"})
    response, source = chatbot.respond(user_message, build_session_context())
    if current_session_id:
        storage.save_chat(session_id=current_session_id, user_message=user_message,
                          assistant_response=response, message_type=source)
    current_chat_history.append({"role": "user",      "content": user_message})
    current_chat_history.append({"role": "assistant",  "content": response})
    return {"response": response, "source": source}


@app.get("/api/session")
async def get_session():
    return {"session_id": current_session_id, "statistics": current_stats,
            "chat_history": current_chat_history,
            "detection_count": len(current_detections) if current_detections else 0}


@app.post("/api/new-session")
async def new_session():
    global current_session_id, current_chat_history, current_detections
    global current_stats, current_original_image, current_detected_image, current_xai_data
    current_session_id = storage.create_session()
    current_chat_history = []; current_detections = None
    current_stats = None; current_original_image = None
    current_detected_image = None; current_xai_data = []
    return {"session_id": current_session_id, "status": "new session created"}


@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": storage.list_sessions()}


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    return {"deleted": storage.delete_session(session_id), "session_id": session_id}


# ============================================================================
# ENTRY POINT - Auto-open browser
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    html_path = Path("front.html").resolve()

    def open_browser():
        time.sleep(2.5)
        if html_path.exists():
            webbrowser.open(f"file:///{str(html_path).replace(chr(92), '/')}")
            print(f"[Starting] Opened front.html in browser")
        else:
            webbrowser.open("http://127.0.0.1:8000/docs")
            print("[WARNING]  front.html not found  opened API docs instead")

    threading.Thread(target=open_browser, daemon=True).start()

    print("\n" + "=" * 70)
    print("[Starting] XCAM-Detect API Server Starting...")
    print("[API] API:        http://127.0.0.1:8000")
    print("[Docs] Docs:       http://127.0.0.1:8000/docs")
    print(f"[UI] Frontend:   {html_path}")
    print("[XAI] XAI:        GradCAM++ + LIME enabled")
    print("[Chatbot] Chatbot:    Rule-based + FAISS RAG")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)