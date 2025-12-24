import React, { useEffect, useRef, useState, memo } from 'react';
import { View, Text, StyleSheet, Dimensions, TouchableOpacity, Button } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';

const { width: SCREEN_W, height: SCREEN_H } = Dimensions.get('window');

const StaticCamera = memo(({ camRef, onReady }: any) => (
  <CameraView
    ref={camRef}
    style={StyleSheet.absoluteFillObject}
    facing="back"
    animateShutter={false}
    onCameraReady={onReady}
  />
));

export default function CameraScreen({ wsUrl, onOpenManual, onChangeServer }: any) {
  // 1. SEMUA HOOK HARUS DI ATAS EARLY RETURN
  const [permission, requestPermission] = useCameraPermissions();
  const [isConnected, setIsConnected] = useState(false);
  const [detections, setDetections] = useState<any[]>([]);
  const [frameSize, setFrameSize] = useState({ w: 1, h: 1 });
  const [lastFrameTime, setLastFrameTime] = useState(Date.now());

  const camRef = useRef<CameraView>(null);
  const ws = useRef<WebSocket | null>(null);
  const isProcessing = useRef(false);
  const isCameraReady = useRef(false);

  // 2. LOGIKA WEBSOCKET (Hanya satu useEffect yang memantau wsUrl)
  useEffect(() => {
    if (!wsUrl) return;

    const connect = () => {
      // console.log("Menghubungkan ke:", wsUrl);
      ws.current = new WebSocket(wsUrl);

      ws.current.onopen = () => setIsConnected(true);
      
      ws.current.onmessage = (e) => {
        const res = JSON.parse(e.data);
        setLastFrameTime(Date.now());
        setDetections(res.detections || []);
        isProcessing.current = false;
      };

      ws.current.onerror = (err) => {
        // console.log("WS error:", err);
        isProcessing.current = false;
      };

      ws.current.onclose = () => {
        setIsConnected(false);
        // Coba hubungkan kembali setelah 3 detik
        setTimeout(() => {
          if (ws.current?.readyState === WebSocket.CLOSED) connect();
        }, 3000);
      };
    };

    connect();

    return () => {
      ws.current?.close();
    };
  }, [wsUrl]); // Berjalan ulang jika wsUrl dari ServerConfig berubah

  // 3. FRAME LOOP
  useEffect(() => {
    const timer = setInterval(async () => {
      if (
        !camRef.current ||
        !isCameraReady.current ||
        isProcessing.current ||
        ws.current?.readyState !== WebSocket.OPEN
      ) return;

      isProcessing.current = true;
      try {
        const photo = await camRef.current.takePictureAsync({
          quality: 0.1, // Gunakan kualitas rendah (0.1) agar streaming lancar
          base64: true,
          skipProcessing: true,
        });

        if (photo?.base64) {
          setFrameSize({ w: photo.width, h: photo.height });
          ws.current.send(photo.base64);
        } else {
          isProcessing.current = false;
        }
      } catch (e) {
        isProcessing.current = false;
      }
    }, 450);

    return () => clearInterval(timer);
  }, [isConnected]);

  // 4. AUTO CLEAR BOX (Mencegah box "nyangkut" saat server lag)
  useEffect(() => {
    const cleaner = setInterval(() => {
      if (Date.now() - lastFrameTime > 600) {
        setDetections([]);
      }
    }, 300);
    return () => clearInterval(cleaner);
  }, [lastFrameTime]);

  // 5. EARLY RETURN SETELAH SEMUA HOOK TERDEFINISI
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

  const scaleX = SCREEN_W / frameSize.w;
  const scaleY = SCREEN_H / frameSize.h;

  return (
    <View style={{ flex: 1, backgroundColor: 'black' }}>
      <StaticCamera
        camRef={camRef}
        onReady={() => (isCameraReady.current = true)}
      />

      {/* OVERLAY BOX */}
      <View style={StyleSheet.absoluteFillObject} pointerEvents="none">
        {detections.map((det, i) => (
          <View
            key={i}
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
            <View style={styles.labelWrapper}>
              <Text style={styles.label}>
                {det.class} {(det.confidence * 100).toFixed(0)}%
              </Text>
            </View>
          </View>
        ))}
      </View>

      {/* HEADER CONTROLS */}
      <View style={styles.headerControls}>
        <TouchableOpacity style={styles.miniBtn} onPress={onOpenManual}>
          <Text style={styles.miniBtnText}>Manual</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.miniBtn} onPress={onChangeServer}>
          <Text style={styles.miniBtnText}>Server</Text>
        </TouchableOpacity>
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
  base: { flex: 1, backgroundColor: '#1E1E1E', justifyContent: 'center', alignItems: 'center' },
  text: { color: 'white' },
  btn: { backgroundColor: '#3b82f6', paddingHorizontal: 20, paddingVertical: 12, borderRadius: 30 },
  btnText: { color: 'white', fontWeight: 'bold' },
  box: { position: 'absolute', borderWidth: 2, borderColor: '#10b981', borderRadius: 8 },
  labelWrapper: { backgroundColor: '#10b981', alignSelf: 'flex-start', paddingHorizontal: 4 },
  label: { color: 'white', fontSize: 10, fontWeight: 'bold' },
  statusWrap: { position: 'absolute', top: 20, width: '100%', alignItems: 'center' },
  statusBadge: { paddingHorizontal: 15, paddingVertical: 5, borderRadius: 20, opacity: 0.8 },
  statusText: { color: 'white', fontWeight: 'bold', fontSize: 10 },
  headerControls: { position: 'absolute', top: 20, left: 20, right: 20, flexDirection: 'row', justifyContent: 'space-between' },
  miniBtn: { backgroundColor: 'rgba(0,0,0,0.5)', padding: 10, borderRadius: 10, borderWidth: 1, borderColor: 'white/20' },
  miniBtnText: { color: 'white', fontSize: 12, fontWeight: 'bold' }
});