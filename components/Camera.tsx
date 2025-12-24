import React, { useEffect, useRef, useState, memo } from 'react';
import { View, Text, StyleSheet, Dimensions, TouchableOpacity } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';

const { width: SCREEN_W, height: SCREEN_H } = Dimensions.get('window');
const WS_URL = `ws://${process.env.EXPO_PUBLIC_SERVER_IP_ADDRESS}:8000/ws`;

/* =========================
   CAMERA PREVIEW (STATIS)
========================= */
const StaticCamera = memo(({ camRef, onReady }: any) => (
  <CameraView
    ref={camRef}
    style={StyleSheet.absoluteFillObject}
    facing="back"
    animateShutter={false}   // ⛔ MATIKAN KEDIP
    onCameraReady={onReady}
  />
));

export default function MainCamera() {
  const [permission, requestPermission] = useCameraPermissions();
  const [isConnected, setIsConnected] = useState(false);
  const camRef = useRef<CameraView>(null);
  const ws = useRef<WebSocket | null>(null);

  const isProcessing = useRef(false);
  const isCameraReady = useRef(false);

  const [detections, setDetections] = useState<any[]>([]);
  const [frameSize, setFrameSize] = useState({ w: 1, h: 1 });
  const [lastFrameTime, setLastFrameTime] = useState(Date.now());

  /* =========================
     WEBSOCKET
  ========================= */
  useEffect(() => {
    const connect = () => {
      ws.current = new WebSocket(WS_URL);
      ws.current.onopen = () => setIsConnected(true);
      ws.current.onmessage = e => {
        const res = JSON.parse(e.data);
  
        setLastFrameTime(Date.now());
        setDetections(res.detections || []);
        isProcessing.current = false;
      };
      ws.current.onclose = () => {
        setIsConnected(false);
        setTimeout(connect, 3000); // Reconnect otomatis
      };
    }
    connect();
    return () => ws.current?.close();
  }, []);

  /* =========================
     FRAME LOOP (ANTI KEDIP)
  ========================= */
  useEffect(() => {
    const timer = setInterval(async () => {
      if (
        !camRef.current ||
        !isCameraReady.current ||
        isProcessing.current ||
        ws.current?.readyState !== WebSocket.OPEN
      ) return;

      if (isConnected && isCameraReady.current && !isProcessing.current) {
        if (camRef.current && ws.current?.readyState === WebSocket.OPEN) {
          isProcessing.current = true;
          try {
            const photo = await camRef.current.takePictureAsync({
              quality: 0.7,       // 🔥 RENDAH = STABIL
              base64: true,
              skipProcessing: true,
            });
    
            if (photo?.base64) {
              setFrameSize({ w: photo.width, h: photo.height });
              ws.current.send(photo.base64);
            } else {
              isProcessing.current = false;
            }
          } catch {
            isProcessing.current = false;
          }
        }
      }
    }, 450); // ⏱️ JANGAN TERLALU CEPAT

    return () => clearInterval(timer);
  }, [isConnected]);

  /* =========================
     AUTO CLEAR BOX
  ========================= */
  useEffect(() => {
    const cleaner = setInterval(() => {
      if (Date.now() - lastFrameTime > 600) {
        setDetections([]);
      }
    }, 300);

    return () => clearInterval(cleaner);
  }, [lastFrameTime]);

  const scaleX = SCREEN_W / frameSize.w;
  const scaleY = SCREEN_H / frameSize.h;

  if (!permission) return <View style={styles.base}><Text style={styles.text}>Mengecek izin...</Text></View>;
    
  if (!permission.granted) {
    return (
      <View style={styles.base}>
        <TouchableOpacity onPress={requestPermission} style={styles.btn}>
          <Text style={styles.btnText}>IZINKAN KAMERA</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={{ flex: 1, backgroundColor: 'black' }}>
      <StaticCamera
        camRef={camRef}
        onReady={() => (isCameraReady.current = true)}
      />

      {/* ===== OVERLAY BOX ===== */}
      <View style={StyleSheet.absoluteFillObject} pointerEvents="none">
        {detections.map((det, i) => (
          <View
            key={`${det.class}-${det.box.join('-')}-${i}`}
            style={[
              styles.box,
              {
                left: det.box[0] * scaleX,
                top: det.box[1] * scaleY,
                width: det.box[2] * scaleX,
                height: det.box[3] * scaleY,
              },
            ]}
          >
            <Text style={styles.label}>
              {det.class} {(det.confidence * 100).toFixed(0)}%
            </Text>
          </View>
        ))}
      </View>

      {/* INDIKATOR STATUS */}
      <View style={styles.statusWrap}>
        <View style={[styles.statusBadge, { backgroundColor: isConnected ? '#10b981' : '#ef4444' }]}>
          <Text style={styles.statusText}>{isConnected ? 'LIVE' : 'RECONNECTING...'}</Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: 'black' },
  base: { flex: 1, backgroundColor: '#1E1E1E', justifyContent: 'center', alignItems: 'center' },
  overlay: { ...StyleSheet.absoluteFillObject, backgroundColor: 'transparent' },
  text: { color: 'white' },
  btn: { backgroundColor: '#3b82f6', paddingHorizontal: 20, paddingVertical: 12, borderRadius: 30 },
  btnText: { color: 'white', fontWeight: 'bold' },
  box: { position: 'absolute', borderWidth: 2, borderColor: '#10b981', borderRadius: 8 },
  labelWrapper: { backgroundColor: '#10b981', alignSelf: 'flex-start', paddingHorizontal: 4, borderRadius: 2 },
  label: { color: 'white', fontSize: 10, fontWeight: 'bold' },
  statusWrap: { position: 'absolute', top: 20, width: '100%', alignItems: 'center' },
  statusBadge: { paddingHorizontal: 15, paddingVertical: 5, borderRadius: 20, opacity: 0.8 },
  statusText: { color: 'white', fontWeight: 'bold', fontSize: 10 }
});
