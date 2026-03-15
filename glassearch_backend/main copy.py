# ==========================================================
# IMPORT LIBRARY: Mengambil alat-alat yang dibutuhkan
# ==========================================================
import cv2  
import numpy as np  
import onnxruntime as ort  
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  
import base64  
import time  

app = FastAPI()  

# ==========================================================
# KONFIGURASI: Pengaturan dasar AI
# ==========================================================
MODEL_PATH = "models\\yolo11n.onnx"  
INPUT_WIDTH = 640  
INPUT_HEIGHT = 640  
CONF_THRESHOLD = 0.3   
NMS_THRESHOLD = 0.45    

# ==========================================================
# LOAD MODEL
# ==========================================================
print("🔄 [ENTRY] Memuat Model YOLO11 Nano ONNX...")
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_meta = session.get_inputs()[0]  
output_meta = session.get_outputs()[0]  
print("✅ [EXIT] Model berhasil dimuat ke memori.\n")

# ==========================================================
# PREPROCESS: Merapikan gambar sebelum dilihat oleh AI
# ==========================================================
def preprocess(img_base64):
    print("🛠️ [ENTRY] Memulai tahap Preprocessing gambar...") 
    try:
        # STEP 1: Pembersihan string Base64 dari header data URI (jika ada)
        if "," in img_base64:
            img_base64 = img_base64.split(",")[1]

        # STEP 2: Dekoding teks Base64 menjadi array biner dan konversi ke format gambar OpenCV
        nparr = np.frombuffer(base64.b64decode(img_base64), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            print("❌ [EXIT] Gagal membaca gambar dari Base64.")
            return None, None, None

        h, w = img.shape[:2]
        print(f"   📸 Resolusi Asli: {w}x{h}") 
        
        # STEP 3: Kalkulasi skala Letterbox agar gambar tidak terdistorsi (tetap proporsional)
        scale = min(INPUT_WIDTH / w, INPUT_HEIGHT / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # STEP 4: Resize gambar ke ukuran target dan siapkan canvas kotak 640x640
        resized = cv2.resize(img, (new_w, new_h))
        canvas = np.full((INPUT_HEIGHT, INPUT_WIDTH, 3), 114, dtype=np.uint8)
        
        # STEP 5: Penambahan Padding (ruang kosong) agar gambar berada tepat di tengah canvas
        pad_x = (INPUT_WIDTH - new_w) // 2
        pad_y = (INPUT_HEIGHT - new_h) // 2
        canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

        # STEP 6: Normalisasi nilai pixel (0-255 menjadi 0-1) dan perubahan dimensi (HWC ke CHW)
        img_input = canvas.astype(np.float32) / 255.0
        img_input = img_input.transpose(2, 0, 1) # Mengubah urutan dimensi untuk model AI
        img_input = np.expand_dims(img_input, axis=0) # Menambah dimensi Batch

        print(f"   📏 Skala Letterbox: {scale:.4f} | Padding: ({pad_x}, {pad_y})")
        print("✅ [EXIT] Preprocessing selesai.") 
        return img_input, (w, h), (scale, pad_x, pad_y)
    except Exception as e:
        print(f"❌ [EXIT] Error di Preprocess: {e}")
        return None, None, None

# ==========================================================
# POSTPROCESS: Menerjemahkan jawaban AI
# ==========================================================
def postprocess(outputs, orig_size, lb):
    print("🔍 [ENTRY] Menganalisis hasil prediksi dari AI...") 
    start_time = time.time()
    
    # STEP 1: Penyederhanaan dimensi output (Squeeze) dan transposisi array
    preds = np.squeeze(outputs[0]).T
    scale, pad_x, pad_y = lb
    boxes, scores = [], []

    print(f"   📊 Jumlah prediksi mentah: {len(preds)}") 

    # STEP 2: Iterasi setiap hasil prediksi dan filter berdasarkan skor keyakinan (Confidence)
    for pred in preds:
        conf = float(pred[4]) 
        if conf > CONF_THRESHOLD:
            # STEP 3: Ekstraksi koordinat pusat (cx, cy) serta lebar dan tinggi (w, h)
            cx, cy, w, h = pred[:4]
            
            # STEP 4: Proses Unletterbox (Mengembalikan koordinat ke skala gambar asli HP)
            real_cx = (cx - pad_x) / scale
            real_cy = (cy - pad_y) / scale
            real_w = w / scale
            real_h = h / scale
            
            # STEP 5: Konversi koordinat pusat menjadi titik pojok kiri atas (format standard x1, y1)
            x1 = int(real_cx - real_w / 2)
            y1 = int(real_cy - real_h / 2)
            
            boxes.append([x1, y1, int(real_w), int(real_h)])
            scores.append(conf)

    orig_w, orig_h = orig_size 
    results = []
    
    if boxes:
        # STEP 6: Menggunakan NMS (Non-Maximum Suppression) untuk menghapus kotak ganda
        idxs = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)
        
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]
                
                # STEP 7: Normalisasi Koordinat (Mengubah piksel menjadi rentang 0.0 - 1.0)
                norm_x = x / orig_w
                norm_y = y / orig_h
                norm_w = w / orig_w
                norm_h = h / orig_h

                det = {
                    "box": [norm_x, norm_y, norm_w, norm_h], 
                    "confidence": round(scores[i], 2),
                    "class": "kacamata"
                }
                results.append(det)
                print(f"      👓 TERDETEKSI: {det['class']} ({det['confidence']*100}%)")

    print(f"✅ [EXIT] Postprocess Selesai dalam {time.time()-start_time:.4f} detik.") 
    return results

# ==========================================================
# WEBSOCKET: Jalur komunikasi real-time
# ==========================================================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    # USER ENTRY: Mencatat saat HP mencoba terhubung
    print("📡 [ENTRY] Mencoba melakukan jabat tangan (handshake) dengan HP...")
    await ws.accept() 
    print("🚀 [EXIT] Client Berhasil Terhubung! Menunggu aliran frame...\n")
    
    frame_idx = 0
    try:
        while True:
            frame_idx += 1
            # USER ENTRY: Menandakan penerimaan data baru
            data = await ws.receive_text()
            print(f"📥 [FRAME {frame_idx}] Data diterima ({len(data)} karakter)") 
            
            inp, orig, lb = preprocess(data)
            if inp is None:
                continue

            t0 = time.time()
            # USER ENTRY: Mencatat saat AI mulai berpikir
            print(f"🧠 [FRAME {frame_idx}] Sedang menjalankan Inferensi YOLO11...")
            outputs = session.run(None, {input_meta.name: inp})
            inf_time = time.time() - t0

            dets = postprocess(outputs, orig, lb)

            # USER ENTRY: Mengonfirmasi hasil dikirim balik ke HP
            print(f"📤 [FRAME {frame_idx}] Mengirim {len(dets)} deteksi kembali ke HP.")
            print(f"⚡ [FRAME {frame_idx}] Total Waktu Inferensi: {inf_time:.4f}s\n")
            
            await ws.send_json({
                "frame": frame_idx,
                "detections": dets,
                "count": len(dets)
            })

    except WebSocketDisconnect:
        # USER ENTRY: Mencatat jika aplikasi HP ditutup
        print("❌ [EXIT] Koneksi Terputus: HP menutup aplikasi atau kehilangan sinyal.") 
    except Exception as e:
        print(f"⚠️ [ERROR] Terjadi kegagalan sistem: {e}")

if __name__ == "__main__":
    import uvicorn
    print("🔥 BACKEND SIAP: Menjalankan server di http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)