---

# U-Glasses - YOLO Glasses Detection App

Aplikasi mobile berbasis **YOLO model** untuk mendeteksi penggunaan kacamata pada pengguna rabun jauh. Model dijalankan melalui **WebSocket** untuk proses inferensi real-time, sehingga memungkinkan integrasi yang ringan dan responsif pada perangkat mobile.

---

## 📌 Fitur Utama
- Deteksi kacamata menggunakan **YOLO object detection**.
- Arsitektur inferensi berbasis **WebSocket** untuk komunikasi cepat antara client dan server.
- Optimasi model agar dapat berjalan pada perangkat mobile dengan performa baik.
- Antarmuka sederhana untuk menampilkan hasil deteksi secara real-time.

---

## 🛠️ Teknologi yang Digunakan
- **YOLO (You Only Look Once)** sebagai backbone deteksi objek.
- **Python** untuk backend model serving.
- **WebSocket** sebagai protokol komunikasi inferensi.
- **React Native** sebagai antarmuka pengguna.

---

---

## 📡 Arsitektur Sistem
```
Mobile App (Camera Input) ---> WebSocket ---> YOLO Model Server ---> Detection Result ---> Mobile App (UI)
```


## 🤝 Kontribusi
Kontribusi sangat terbuka! Silakan buat **pull request** atau buka **issue** untuk diskusi lebih lanjut.

---
