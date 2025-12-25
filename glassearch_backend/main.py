import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import base64
import time

app = FastAPI()

# =========================
# KONFIGURASI YOLO11
# =========================
MODEL_PATH = "glassearch_backend\\best2.onnx"
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONF_THRESHOLD = 0.7   # Threshold keyakinan
NMS_THRESHOLD = 0.45    # Threshold tumpang tindih kotak

# =========================
# LOAD MODEL YOLO11n
# =========================
print("🔄 Loading YOLO11 Nano ONNX Model...")
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_meta = session.get_inputs()[0]
output_meta = session.get_outputs()[0]

print(f"📥 Input  : {input_meta.name} {input_meta.shape}")
print(f"📤 Output : {output_meta.name} {output_meta.shape}")
print("✅ YOLO11 Nano Ready\n")

# =========================
# PREPROCESS (LETTERBOX)
# =========================
def preprocess(img_base64):
    try:
        if "," in img_base64:
            img_base64 = img_base64.split(",")[1]

        nparr = np.frombuffer(base64.b64decode(img_base64), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return None, None, None

        h, w = img.shape[:2]
        
        # Letterbox: Menjaga aspek rasio agar kacamata tidak lonjong
        scale = min(INPUT_WIDTH / w, INPUT_HEIGHT / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h))
        canvas = np.full((INPUT_HEIGHT, INPUT_WIDTH, 3), 114, dtype=np.uint8)
        
        pad_x = (INPUT_WIDTH - new_w) // 2
        pad_y = (INPUT_HEIGHT - new_h) // 2
        canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

        # Normalisasi ke format CHW
        img_input = canvas.astype(np.float32) / 255.0
        img_input = img_input.transpose(2, 0, 1)
        img_input = np.expand_dims(img_input, axis=0)

        return img_input, (w, h), (scale, pad_x, pad_y)
    except Exception as e:
        print(f"❌ Error in Preprocess: {e}")
        return None, None, None

# =========================
# POSTPROCESS (YOLO11 SPECIFIC)
# =========================
def postprocess(outputs, orig_size, lb):
    start_time = time.time()
    
    # YOLO11 Output: [1, 5, 8400] -> squeeze & transpose ke [8400, 5]
    preds = np.squeeze(outputs[0]).T
    
    scale, pad_x, pad_y = lb
    boxes, scores = [], []

    for pred in preds:
        # Index 4 adalah confidence score untuk kelas 'kacamata'
        conf = float(pred[4]) 
        
        if conf > CONF_THRESHOLD:
            cx, cy, w, h = pred[:4]
            
            # Unletterbox: Mengembalikan koordinat ke dimensi asli HP
            # $$real\_cx = \frac{cx - pad\_x}{scale}$$
            real_cx = (cx - pad_x) / scale
            real_cy = (cy - pad_y) / scale
            real_w = w / scale
            real_h = h / scale

            # Konversi Center-XY ke Top-Left-XY untuk React Native
            x1 = int(real_cx - real_w / 2)
            y1 = int(real_cy - real_h / 2)

            boxes.append([x1, y1, int(real_w), int(real_h)])
            scores.append(conf)

    results = []
    if boxes:
        # NMS untuk eliminasi kotak ganda
        idxs = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)
        if len(idxs) > 0:
            for i in idxs.flatten():
                det = {
                    "box": boxes[i],
                    "confidence": round(scores[i], 2),
                    "class": "kacamata"
                }
                results.append(det)
                print(f"👓 Detected: {det['class']} ({det['confidence']*100}%)")

    print(f"⏱️ Postprocess {time.time()-start_time:.3f}s | Found: {len(results)}")
    return results

# =========================
# WEBSOCKET
# =========================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("🚀 YOLO11n Server: Client Connected\n")
    frame_idx = 0

    try:
        while True:
            frame_idx += 1
            data = await ws.receive_text()
            
            # Preprocess
            inp, orig, lb = preprocess(data)
            if inp is None:
                continue

            # Inference
            t0 = time.time()
            outputs = session.run(None, {input_meta.name: inp})
            inf_time = time.time() - t0

            # Postprocess
            dets = postprocess(outputs, orig, lb)

            # Logging ala main2.py
            print(f"🎞️ Frame {frame_idx} | ⚡ Inf: {inf_time:.3f}s")
            
            await ws.send_json({
                "frame": frame_idx,
                "detections": dets,
                "count": len(dets)
            })

    except WebSocketDisconnect:
        print("❌ Client disconnected")
    except Exception as e:
        print(f"⚠️ Runtime Error: {e}")

if __name__ == "__main__":
    import uvicorn
    print("🔥 YOLO11n GLASSEARCH BACKEND READY")
    uvicorn.run(app, host="0.0.0.0", port=8000)