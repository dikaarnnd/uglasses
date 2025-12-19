import React, { useEffect, useState, useRef } from 'react';
import { View, Text, Dimensions, Platform, ActivityIndicator, StatusBar } from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
  Frame,
} from 'react-native-vision-camera';
import { runOnJS } from 'react-native-reanimated';

const { width: SCREEN_W, height: SCREEN_H } = Dimensions.get('window');

// Tipe untuk deteksi
type Detection = {
  x: number;
  y: number;
  w: number;
  h: number;
  confidence: number;
  classId: number;
  className?: string;
};

// Label untuk classId (sesuaikan dengan model YOLOv12n Anda)
const LABEL_MAP: Record<number, string> = {
  0: 'person',
  1: 'bicycle',
  2: 'car',
  3: 'motorcycle',
  4: 'airplane',
  5: 'bus',
  6: 'train',
  7: 'truck',
  8: 'boat',
  9: 'traffic light',
  10: 'fire hydrant',
  11: 'stop sign',
  12: 'parking meter',
  13: 'bench',
  14: 'bird',
  15: 'cat',
  16: 'dog',
  17: 'horse',
  18: 'sheep',
  19: 'cow',
  20: 'elephant',
  21: 'bear',
  22: 'zebra',
  23: 'giraffe',
  24: 'backpack',
  25: 'umbrella',
  26: 'handbag',
  27: 'tie',
  28: 'suitcase',
  29: 'frisbee',
  30: 'skis',
  31: 'snowboard',
  32: 'sports ball',
  33: 'kite',
  34: 'baseball bat',
  35: 'baseball glove',
  36: 'skateboard',
  37: 'surfboard',
  38: 'tennis racket',
  39: 'bottle',
  40: 'wine glass',
  41: 'cup',
  42: 'fork',
  43: 'knife',
  44: 'spoon',
  45: 'bowl',
  46: 'banana',
  47: 'apple',
  48: 'sandwich',
  49: 'orange',
  50: 'broccoli',
  51: 'carrot',
  52: 'hot dog',
  53: 'pizza',
  54: 'donut',
  55: 'cake',
  56: 'chair',
  57: 'couch',
  58: 'potted plant',
  59: 'bed',
  60: 'dining table',
  61: 'toilet',
  62: 'tv',
  63: 'laptop',
  64: 'mouse',
  65: 'remote',
  66: 'keyboard',
  67: 'cell phone',
  68: 'microwave',
  69: 'oven',
  70: 'toaster',
  71: 'sink',
  72: 'refrigerator',
  73: 'book',
  74: 'clock',
  75: 'vase',
  76: 'scissors',
  77: 'teddy bear',
  78: 'hair drier',
  79: 'toothbrush',
};

// Batas kepercayaan (confidence threshold)
const CONFIDENCE_THRESHOLD = 0.5;

export default function CameraScreen() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  const [detections, setDetections] = useState<Detection[]>([]);
  const [isModelReady, setIsModelReady] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [fps, setFps] = useState(0);
  const frameCountRef = useRef(0);
  const lastFpsTimeRef = useRef(Date.now());

  // Inisialisasi model YOLO
  useEffect(() => {
    const initializeModel = async () => {
      try {
        // Panggil native module untuk memuat model
        if (Platform.OS === 'android') {
          // Untuk Android
          const { YoloModule } = require('./NativeYoloModule');
          const success = await YoloModule.loadModel('yolov12n.torchscript.pt');
          if (success) {
            console.log('Model YOLOv12n loaded successfully');
            setIsModelReady(true);
          }
        } else if (Platform.OS === 'ios') {
          // Untuk iOS
          const { YoloModule } = require('./NativeYoloModule');
          const success = await YoloModule.loadModel('yolov12n.torchscript.pt');
          if (success) {
            console.log('Model YOLOv12n loaded successfully');
            setIsModelReady(true);
          }
        }
      } catch (error) {
        console.error('Failed to load model:', error);
        setIsModelReady(false);
      }
    };

    initializeModel();
  }, []);

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission]);

  // Fungsi untuk menjalankan inferensi YOLO
  const runYoloInference = async (frame: Frame): Promise<Detection[]> => {
    try {
      if (!isModelReady || isProcessing) {
        return [];
      }

      setIsProcessing(true);

      // Konversi frame ke format yang bisa diproses oleh native module
      // Di sini kita akan mengirim data frame ke native module
      const { YoloModule } = require('./NativeYoloModule');
      
      // Untuk Android, kita bisa mengirim data frame langsung
      if (Platform.OS === 'android') {
        const result = await YoloModule.detectFromFrame(
          frame,
          CONFIDENCE_THRESHOLD
        );
        
        // Parse hasil deteksi
        const detections: Detection[] = result.map((det: any) => ({
          x: det.x,
          y: det.y,
          w: det.width,
          h: det.height,
          confidence: det.confidence,
          classId: det.classId,
          className: LABEL_MAP[det.classId] || `Class ${det.classId}`,
        }));
        
        return detections;
      }
      
      // Untuk iOS, implementasi mungkin berbeda
      if (Platform.OS === 'ios') {
        const result = await YoloModule.detectFromFrame(
          frame,
          CONFIDENCE_THRESHOLD
        );
        
        const detections: Detection[] = result.map((det: any) => ({
          x: det.x,
          y: det.y,
          w: det.width,
          h: det.height,
          confidence: det.confidence,
          classId: det.classId,
          className: LABEL_MAP[det.classId] || `Class ${det.classId}`,
        }));
        
        return detections;
      }

      return [];
    } catch (error) {
      console.error('YOLO inference error:', error);
      return [];
    } finally {
      setIsProcessing(false);
    }
  };

  // Frame processor
  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';
    
    // Hitung FPS
    frameCountRef.current += 1;
    const now = Date.now();
    if (now - lastFpsTimeRef.current >= 1000) {
      runOnJS(setFps)(frameCountRef.current);
      frameCountRef.current = 0;
      lastFpsTimeRef.current = now;
    }
    
    // Jalankan inferensi setiap 3 frame (untuk performa)
    if (frameCountRef.current % 3 === 0 && isModelReady && !isProcessing) {
      // Jalankan inferensi di thread JavaScript
      runOnJS(async () => {
        const newDetections = await runYoloInference(frame);
        if (newDetections.length > 0) {
          setDetections(newDetections);
        }
      })();
    }
  }, [isModelReady, isProcessing]);

  if (!device || !hasPermission) {
    return (
      <View className="flex-1 items-center justify-center bg-black">
        <ActivityIndicator size="large" color="#4F46E5" />
        <Text className="text-white mt-4">Loading camera...</Text>
      </View>
    );
  }

  // Skala untuk menyesuaikan koordinat model dengan layar
  // Model YOLO biasanya bekerja dengan resolusi 640x640
  const modelWidth = 640;
  const modelHeight = 640;
  
  const scaleX = SCREEN_W / modelWidth;
  const scaleY = SCREEN_H / modelHeight;

  return (
    <View className="flex-1 bg-black">
      <Camera
        className="absolute inset-0"
        device={device}
        isActive={true}
        frameProcessor={frameProcessor}
        // frameProcessorFps={10}
        audio={false}
      />

      {/* Overlay untuk status */}
      <View className="absolute top-10 left-0 right-0 items-center">
        <View className="bg-black/70 px-4 py-2 rounded-lg">
          <Text className="text-white text-sm">
            {isModelReady ? 'Model Ready' : 'Loading Model...'} | FPS: {fps} | Detections: {detections.length}
          </Text>
        </View>
      </View>

      {/* Render bounding boxes */}
      {detections.map((det, index) => {
        // Filter berdasarkan confidence threshold
        if (det.confidence < CONFIDENCE_THRESHOLD) return null;
        
        const className = LABEL_MAP[det.classId] || `Class ${det.classId}`;
        
        return (
          <View
            key={`${index}-${det.classId}-${det.confidence}`}
            style={{
              position: 'absolute',
              left: det.x * scaleX,
              top: det.y * scaleY,
              width: det.w * scaleX,
              height: det.h * scaleY,
              borderWidth: 2,
              borderColor: '#00FF00',
              backgroundColor: 'rgba(0, 255, 0, 0.1)',
              borderRadius: 4,
            }}
          >
            <View className="absolute -top-6 left-0 bg-green-600 px-2 py-1 rounded">
              <Text className="text-white text-xs font-bold">
                {className} {(det.confidence * 100).toFixed(0)}%
              </Text>
            </View>
          </View>
        );
      })}

      {/* Processing indicator */}
      {isProcessing && (
        <View className="absolute bottom-10 left-0 right-0 items-center">
          <View className="flex-row items-center bg-black/70 px-4 py-2 rounded-lg">
            <ActivityIndicator size="small" color="#00FF00" />
            <Text className="text-white ml-2">Processing YOLO inference...</Text>
          </View>
        </View>
      )}
    </View>
  );
}