import React, { useState, useEffect, useRef, memo } from 'react';
import { Text, View, Dimensions, StyleSheet, TouchableOpacity } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';

// Gunakan dimensi layar untuk kalkulasi posisi
const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');
const SERVER_IP = process.env.EXPO_PUBLIC_SERVER_IP_ADDRESS; // Pastikan IP sesuai dengan laptop Anda
const WS_URL = `ws://${SERVER_IP}:8000/ws`;

const StaticCamera = memo(({ cameraRef, onReady }: any) => (
  <CameraView
    style={StyleSheet.absoluteFillObject}
    ref={cameraRef}
    facing="back"
    onCameraReady={onReady}
    animateShutter={false} // Mengurangi flicker tambahan
  />
));

export default function MainCamera() {
  const [permission, requestPermission] = useCameraPermissions();
  const [isConnected, setIsConnected] = useState(false);
  const [detections, setDetections] = useState([]);
  
  const cameraRef = useRef<CameraView>(null);
  const ws = useRef<WebSocket | null>(null);
  
  const isProcessing = useRef(false);
  const isCameraReady = useRef(false);

  // 1. WebSocket Management
  useEffect(() => {
    const connect = () => {
      ws.current = new WebSocket(WS_URL);
      ws.current.onopen = () => setIsConnected(true);
      ws.current.onmessage = (e) => {
        const res = JSON.parse(e.data);
        if (res.detections) setDetections(res.detections);
        isProcessing.current = false;
      };
      ws.current.onclose = () => {
        setIsConnected(false);
        setTimeout(connect, 3000); // Reconnect otomatis
      };
    };
    connect();
    return () => ws.current?.close();
  }, []);

  // 2. Optimized Frame Loop
  useEffect(() => {
    const timer = setInterval(async () => {
      if (isConnected && isCameraReady.current && !isProcessing.current) {
        if (cameraRef.current && ws.current?.readyState === WebSocket.OPEN) {
          isProcessing.current = true;
          try {
            const photo = await cameraRef.current.takePictureAsync({
              quality: 0.1, // Sangat rendah agar cepat dikirim
              base64: true,
              skipProcessing: true,
            });
            if (photo?.base64) {
              ws.current.send(photo.base64);
            } else {
              isProcessing.current = false;
            }
          } catch (e) {
            isProcessing.current = false;
          }
        }
      }
    }, 450); // Interval sedikit dilonggarkan untuk stabilitas
    return () => clearInterval(timer);
  }, [isConnected]);

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
    <View style={styles.container}>
      <StaticCamera 
        cameraRef={cameraRef} 
        onReady={() => { isCameraReady.current = true; }} 
      />

      {/* LAYER OVERLAY: pointerEvents="none" agar tombol di bawah tetap bisa diklik */}
      <View style={styles.overlay} pointerEvents="none">
        {detections.map((det: any, i) => {
          const [x, y, w, h] = det.box;

          // LOGIKA SCALING: 
          // Pastikan pembagi (640) sesuai dengan resolusi resize di backend Anda
          const mappedX = (x * SCREEN_WIDTH) / SCREEN_WIDTH;
          const mappedY = (y * SCREEN_HEIGHT) / SCREEN_HEIGHT;
          const mappedW = (w * SCREEN_WIDTH) / 640;
          const mappedH = (h * SCREEN_HEIGHT) / 640;

          return (
            <View key={i} style={[styles.box, {
              left: mappedX,
              top: mappedY,
              width: mappedW,
              height: mappedH,
            }]}>
              <View style={styles.labelWrapper}>
                <Text style={styles.label}>
                  {det.class} {(det.confidence * 100).toFixed(0)}%
                </Text>
              </View>
            </View>
          );
        })}
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