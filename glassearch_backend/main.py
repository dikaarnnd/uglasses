# ==========================================================
# IMPORT LIBRARY: Mengambil alat-alat yang dibutuhkan
# ==========================================================
import cv2  # Library untuk mengolah gambar (resize, warna, gambar kotak)
import numpy as np  # Untuk perhitungan matematika dan mengolah data angka (array)
import onnxruntime as ort  # Mesin utama untuk menjalankan model kecerdasan buatan (AI)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # Framework untuk membuat server web cepat
import base64  # Untuk mengubah kode teks (Base64) dari HP kembali menjadi gambar asli
import time  # Untuk menghitung berapa lama proses deteksi berjalan

app = FastAPI()  # Membuat aplikasi web utama

# ==========================================================
# KONFIGURASI: Pengaturan dasar AI
# ==========================================================
MODEL_PATH = "models\\yolo11n.onnx"  # Lokasi file otak AI (model YOLO11)
INPUT_WIDTH = 640  # Ukuran lebar gambar yang diminta oleh AI
INPUT_HEIGHT = 640  # Ukuran tinggi gambar yang diminta oleh AI
CONF_THRESHOLD = 0.3   # AI hanya melapor jika yakin di atas 70% itu adalah kacamata
NMS_THRESHOLD = 0.45    # Untuk menghapus kotak deteksi yang tumpang tindih pada satu objek

# ==========================================================
# LOAD MODEL: Memasukkan otak AI ke dalam memori
# ==========================================================
print("🔄 Loading YOLO11 Nano ONNX Model...")
# Menjalankan model menggunakan CPU laptop
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_meta = session.get_inputs()[0]  # Mencari tahu apa yang harus dimasukkan ke AI
output_meta = session.get_outputs()[0]  # Mencari tahu format jawaban dari AI

print(f"📥 Input  : {input_meta.name} {input_meta.shape}")
print(f"📤 Output : {output_meta.name} {output_meta.shape}")
print("✅ YOLO11 Nano Ready\n")

# ==========================================================
# PREPROCESS: Merapikan gambar sebelum dilihat oleh AI
# ==========================================================
def preprocess(img_base64):
    try:
        # Menghapus teks tambahan jika ada di awal data Base64
        if "," in img_base64:
            img_base64 = img_base64.split(",")[1]

        # Mengubah teks Base64 menjadi deretan angka biner
        nparr = np.frombuffer(base64.b64decode(img_base64), np.uint8)
        # Mengubah data biner menjadi gambar berwarna yang dimengerti OpenCV
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return None, None, None

        h, w = img.shape[:2]  # Mengambil ukuran asli gambar dari kamera HP
        
        # Letterbox: Menghitung skala agar gambar tidak lonjong saat di-resize
        scale = min(INPUT_WIDTH / w, INPUT_HEIGHT / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Mengecilkan gambar sesuai skala
        resized = cv2.resize(img, (new_w, new_h))
        # Membuat canvas abu-abu kotak (640x640) sebagai latar belakang
        canvas = np.full((INPUT_HEIGHT, INPUT_WIDTH, 3), 114, dtype=np.uint8)
        
        # Menempelkan gambar yang sudah di-resize tepat di tengah canvas
        pad_x = (INPUT_WIDTH - new_w) // 2
        pad_y = (INPUT_HEIGHT - new_h) // 2
        canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

        # Mengubah angka warna (0-255) menjadi angka desimal (0-1) agar AI lebih mudah belajar
        img_input = canvas.astype(np.float32) / 255.0
        # Mengubah susunan data dari (Tinggi, Lebar, Warna) ke (Warna, Tinggi, Lebar)
        img_input = img_input.transpose(2, 0, 1)
        # Menambahkan satu dimensi lagi agar sesuai format AI (Batch Size)
        img_input = np.expand_dims(img_input, axis=0)

        return img_input, (w, h), (scale, pad_x, pad_y)
    except Exception as e:
        print(f"❌ Error in Preprocess: {e}")
        return None, None, None

# ==========================================================
# POSTPROCESS: Menerjemahkan jawaban angka dari AI menjadi lokasi objek
# ==========================================================
def postprocess(outputs, orig_size, lb):
    start_time = time.time()
    
    # Merapikan hasil jawaban AI yang tadinya berbentuk tensor rumit
    preds = np.squeeze(outputs[0]).T
    
    scale, pad_x, pad_y = lb
    boxes, scores = [], []

    for pred in preds:
        # Index ke-4 adalah tingkat keyakinan AI bahwa itu kacamata
        conf = float(pred[4]) 
        
        if conf > CONF_THRESHOLD:
            cx, cy, w, h = pred[:4] # Koordinat titik tengah, lebar, dan tinggi
            
            # Unletterbox: Mengembalikan lokasi kacamata ke ukuran asli layar HP
            real_cx = (cx - pad_x) / scale
            real_cy = (cy - pad_y) / scale
            real_w = w / scale
            real_h = h / scale

            # Mengubah koordinat titik tengah menjadi titik pojok kiri atas (format standard)
            x1 = int(real_cx - real_w / 2)
            y1 = int(real_cy - real_h / 2)

            boxes.append([x1, y1, int(real_w), int(real_h)])
            scores.append(conf)

    results = []
    if boxes:
        # NMS: Jika ada banyak kotak di kacamata yang sama, pilih satu yang paling yakin
        idxs = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)
        if len(idxs) > 0:
            for i in idxs.flatten():
                det = {
                    "box": boxes[i], # Lokasi kotak
                    "confidence": round(scores[i], 2), # Seberapa yakin AI
                    "class": "kacamata" # Nama benda yang ditemukan
                }
                results.append(det)
                print(f"👓 Detected: {det['class']} ({det['confidence']*100}%)")

    print(f"⏱️ Postprocess {time.time()-start_time:.3f}s | Found: {len(results)}")
    return results

# ==========================================================
# WEBSOCKET: Jalur komunikasi dua arah antara HP dan Laptop
# ==========================================================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept() # Menerima koneksi dari HP
    print("🚀 YOLO11n Server: Client Connected\n")
    frame_idx = 0

    try:
        while True:
            frame_idx += 1
            # Menerima data gambar berupa teks Base64 dari HP
            data = await ws.receive_text()
            
            # 1. Tahap Persiapan Gambar
            inp, orig, lb = preprocess(data)
            if inp is None:
                continue

            # 2. Tahap Inferensi: Menyerahkan gambar ke AI untuk ditebak
            t0 = time.time()
            outputs = session.run(None, {input_meta.name: inp})
            inf_time = time.time() - t0

            # 3. Tahap Penerjemahan Hasil AI
            dets = postprocess(outputs, orig, lb)

            # 4. Tahap Pengiriman Hasil: Mengirim data deteksi kembali ke HP
            print(f"🎞️ Frame {frame_idx} | ⚡ Inf: {inf_time:.3f}s")
            await ws.send_json({
                "frame": frame_idx,
                "detections": dets,
                "count": len(dets)
            })

    except WebSocketDisconnect:
        print("❌ Client disconnected") # Terjadi jika aplikasi HP ditutup
    except Exception as e:
        print(f"⚠️ Runtime Error: {e}")

if __name__ == "__main__":
    import uvicorn
    # Menjalankan server pada IP laptop Anda di port 8000
    print("🔥 YOLO11n GLASSEARCH BACKEND READY")
    uvicorn.run(app, host="0.0.0.0", port=8000)