import React, {useRef, useState} from 'react';
import {StyleSheet, View, Text, StatusBar} from 'react-native';
import {Camera, Canvas, Model, ImageUtil} from 'react-native-pytorch-core';

// Nama file model di folder assets
const MODEL_URL = 'models/yolov12n.torchscript';
const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

// Daftar label COCO (Ganti sesuai kebutuhan modelmu)
const COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
  'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
  'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];

export default function MainCamera() {
  const [label, setLabel] = useState('Memulai...');
  // const [prediction, setPrediction] = useState('Memuat Model...');
  const modelRef = useRef(null); // Gunakan ref untuk menyimpan instance model
  const isProcessing = useRef(false);

  // Langkah A: Muat model hanya sekali saat aplikasi dimulai
  useEffect(() => {
    async function loadModel() {
      try {
        const model = await MobileModel.load(MODEL_URL);
        modelRef.current = model;
        setPrediction('Model Siap. Arahkan Kamera...');
      } catch (e) {
        console.error("Gagal load model:", e);
        setPrediction('Gagal memuat model');
      }
    }
    loadModel();
  }, []);
  
  // Fungsi yang dipanggil setiap frame kamera
  async function handleFrame(image) {
    if (!modelRef.current) {
      image.release();
      return;
    }
    try {
      // 2. Pre-processing: Ubah gambar kamera menjadi tensor
      // YOLO biasanya membutuhkan input size 640x640
      const {tensor} = await ImageUtil.toTensor(image, {
        width: 640,
        height: 640,
      });

      // 3. Inference
      const output = await modelRef.current.forward(tensor);

      // 4. Post-processing (Sederhana)
      // YOLOv12 mengembalikan tensor berisi [boxes, scores, class_ids]
      // Logika di bawah ini perlu disesuaikan dengan shape output model Anda
      const result = processYOLOv12(output); 
      setPrediction(result);

      // 5. Lepas memori image & tensor
      image.release();
    } catch (err) {
      console.log("Inference error:", err);
      image.release();
    } finally {
      isProcessing.current = false;
    }
  }

  // 3. Logika Post-Processing YOLOv12
  function processYOLOv12(output) {
    const data = output.data; 
    const numClasses = 80;
    const numPredictions = 8400; // Standar output YOLOv12 (640x640)

    let maxConfidence = 0;
    let detectedClassId = -1;

    // Kita mencari skor kepercayaan tertinggi dari seluruh prediksi
    for (let i = 0; i < numPredictions; i++) {
      for (let j = 0; j < numClasses; j++) {
        // Indexing tensor YOLOv12: [1, 84, 8400]
        const confidence = data[(4 + j) * numPredictions + i];
        
        if (confidence > maxConfidence) {
          maxConfidence = confidence;
          detectedClassId = j;
        }
      }
    }

    if (maxConfidence > 0.45) { // Threshold 45%
      const className = COCO_CLASSES[detectedClassId] || 'Unknown';
      return `${className.toUpperCase()} (${(maxConfidence * 100).toFixed(1)}%)`;
    }
    
    return "Mencari objek...";
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />
      <Camera
        style={styles.camera}
        onFrame={handleFrame}
        hideControls={true}
      />
      <Canvas style={styles.canvas} />
      <View style={styles.labelContainer}>
        <Text style={styles.labelText}>{prediction}</Text>
      </View>
    </View>
  );
}

// Fungsi pembantu untuk memproses output tensor (Contoh kasar)
function processOutput(output) {
  // Anda perlu membedah tensor output sesuai arsitektur YOLOv12
  // Biasanya menggunakan Softmax untuk mendapatkan class dengan probabilitas tertinggi
  return "Object Detected"; 
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: 'black' },
  camera: { flex: 1 },
  canvas: { position: 'absolute', width: '100%', height: '100%' },
  labelContainer: {
    position: 'absolute',
    bottom: 50,
    alignSelf: 'center',
    backgroundColor: 'rgba(0,0,0,0.7)',
    padding: 10,
    borderRadius: 8,
  },
  labelText: { color: 'white', fontSize: 18, fontWeight: 'bold' },
});