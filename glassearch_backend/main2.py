import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import base64
import time

app = FastAPI()

# =========================
# KONFIGURASI
# =========================
MODEL_PATH = "glassearch_backend/yolo12n.onnx"
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONF_THRESHOLD = 0.15   # ⬅️ TURUNIN BIAR OBJEK JAUH KEDETECT
NMS_THRESHOLD = 0.45

COCO_CLASS_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

# =========================
# LOAD MODEL
# =========================
print("🔄 Loading YOLOv12n ONNX...")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_meta = session.get_inputs()[0]
output_meta = session.get_outputs()[0]

print(f"📥 Input  : {input_meta.name} {input_meta.shape}")
print(f"📤 Output : {output_meta.name} {output_meta.shape}")
print("✅ Model ready\n")

# =========================
# PREPROCESS (LETTERBOX)
# =========================
def preprocess(img_base64):
    if "," in img_base64:
        img_base64 = img_base64.split(",")[1]

    img = cv2.imdecode(
        np.frombuffer(base64.b64decode(img_base64), np.uint8),
        cv2.IMREAD_COLOR
    )

    if img is None:
        return None, None, None

    h, w = img.shape[:2]
    print(f"📷 Frame size: {w}x{h}")

    scale = min(INPUT_WIDTH / w, INPUT_HEIGHT / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.full((INPUT_HEIGHT, INPUT_WIDTH, 3), 114, dtype=np.uint8)

    pad_x = (INPUT_WIDTH - new_w) // 2
    pad_y = (INPUT_HEIGHT - new_h) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

    img_input = canvas.astype(np.float32) / 255.0
    img_input = img_input.transpose(2, 0, 1)
    img_input = np.expand_dims(img_input, axis=0)

    print(f"📐 scale={scale:.4f} pad_x={pad_x} pad_y={pad_y}")
    return img_input, (w, h), (scale, pad_x, pad_y)

# =========================
# POSTPROCESS (YOLOv12 + UNLETTERBOX)
# =========================
def postprocess(outputs, orig_size, lb):
    start = time.time()
    preds = np.squeeze(outputs[0]).T
    print(f"🔍 Raw preds: {preds.shape}")

    orig_w, orig_h = orig_size
    scale, pad_x, pad_y = lb

    boxes, scores, class_ids = [], [], []
    best = (None, 0)

    for pred in preds:
        class_probs = pred[4:84]
        cid = int(np.argmax(class_probs))
        conf = float(class_probs[cid])

        if conf > best[1]:
            best = (cid, conf)

        if conf < CONF_THRESHOLD:
            continue

        cx, cy, w, h = pred[:4]
        cx = (cx - pad_x) / scale
        cy = (cy - pad_y) / scale
        w /= scale
        h /= scale

        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)

        boxes.append([x1, y1, int(w), int(h)])
        scores.append(conf)
        class_ids.append(cid)

    if best[0] is not None:
        print(f"🏆 Top: {COCO_CLASS_NAMES[best[0]]} {best[1]:.3f}")

    results = []
    if boxes:
        idxs = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)
        if len(idxs) > 0:
            for i in idxs.flatten():
                det = {
                    "box": boxes[i],
                    "confidence": round(scores[i], 3),
                    "class": COCO_CLASS_NAMES[class_ids[i]],
                    "class_id": class_ids[i]
                }
                results.append(det)
                print(f"✅ {det['class']} {det['confidence']}")

    print(f"⏱️ Postprocess {time.time()-start:.3f}s | Detected {len(results)}\n")
    return results

# =========================
# WEBSOCKET
# =========================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("🚀 Client connected\n")
    frame = 0

    try:
        while True:
            frame += 1
            print("="*50)
            print(f"🎞️ FRAME {frame}")

            data = await ws.receive_text()
            inp, orig, lb = preprocess(data)
            if inp is None:
                await ws.send_json({"frame": frame, "detections": []})
                continue

            t0 = time.time()
            outputs = session.run(None, {input_meta.name: inp})
            print(f"⚡ Inference {time.time()-t0:.3f}s")

            dets = postprocess(outputs, orig, lb)

            await ws.send_json({
                "frame": frame,
                "count": len(dets),
                "detections": dets
            })

    except WebSocketDisconnect:
        print("❌ Client disconnected")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    import uvicorn
    print("🔥 YOLOv12n SERVER READY")
    uvicorn.run(app, host="0.0.0.0", port=8000)
