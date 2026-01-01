import React, { useEffect, useRef, useState, memo } from 'react';
// Tambahkan Vibration di sini
import { View, Text, StyleSheet, TouchableOpacity, Vibration, Dimensions } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';

const StaticCamera = memo(({ camRef, onReady }: any) => (
  <CameraView
    ref={camRef}
    style={StyleSheet.absoluteFillObject}
    facing="back"
    animateShutter={false}
    onCameraReady={onReady}
  />
));

export default function CameraScreen({ wsUrl }: any) {
  const [permission, requestPermission] = useCameraPermissions();
  const [isConnected, setIsConnected] = useState(false);
  // Simpan status deteksi kacamata (true/false)
  const [isDetected, setIsDetected] = useState(false);
  const [lastFrameTime, setLastFrameTime] = useState(Date.now());
  const [boxes, setBoxes] = useState<any[]>([]);
  // const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');
  const [layout, setLayout] = useState({ width: 0, height: 0 });

  const camRef = useRef<CameraView>(null);
  const ws = useRef<WebSocket | null>(null);
  const isProcessing = useRef(false);
  const isCameraReady = useRef(false);

  useEffect(() => {
    if (!wsUrl) return;

    const connect = () => {
      ws.current = new WebSocket(wsUrl);
      ws.current.onopen = () => setIsConnected(true);
      
      ws.current.onmessage = (e) => {
        const res = JSON.parse(e.data);
        setLastFrameTime(Date.now());
        
        // Logika Baru: Cek jika ada kacamata yang terdeteksi
        if (res.detections && res.detections.length > 0) {
          setIsDetected(true);
          setBoxes(res.detections);
          // Getarkan HP selama 100ms saat terdeteksi
          Vibration.vibrate(100); 
        } else {
          setIsDetected(false);
          setBoxes([]);
        }
        
        isProcessing.current = false;
      };

      ws.current.onerror = () => { isProcessing.current = false; };
      ws.current.onclose = () => {
        setIsConnected(false);
        setTimeout(() => {
          if (ws.current?.readyState === WebSocket.CLOSED) connect();
        }, 3000);
      };
    };

    connect();
    return () => { ws.current?.close(); };
  }, [wsUrl]);

  // Logika Pengiriman Frame tetap sama
  useEffect(() => {
    const timer = setInterval(async () => {
      if (!camRef.current || !isCameraReady.current || isProcessing.current || ws.current?.readyState !== WebSocket.OPEN) return;

      isProcessing.current = true;
      try {
        const photo = await camRef.current.takePictureAsync({
          quality: 0.8, // Kualitas gambar
          base64: true,
          skipProcessing: true,
        });

        if (photo?.base64) {
          ws.current.send(photo.base64);
        } else {
          isProcessing.current = false;
        }
      } catch (e) { isProcessing.current = false; }
    }, 450);
    return () => clearInterval(timer);
  }, [isConnected]);

  // Auto clear status jika koneksi lag
  useEffect(() => {
    const cleaner = setInterval(() => {
      if (Date.now() - lastFrameTime > 600) {
        setIsDetected(false);
      }
    }, 300);
    return () => clearInterval(cleaner);
  }, [lastFrameTime]);

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
    <View 
      style={{ flex: 1, backgroundColor: 'black' }}
      onLayout={(event) => {
        const { width, height } = event.nativeEvent.layout;
        setLayout({ width, height });
      }}
    >
      <StaticCamera camRef={camRef} onReady={() => (isCameraReady.current = true)} />
      {boxes.map((det, index) => {
        const [nx, ny, nw, nh] = det.box; // Ini adalah angka 0.0 - 1.0 dari server

        const SCALE_FACTOR = 1.6; // (1.2 = +20%)
        const OFFSET_X = 0.035;
        const OFFSET_Y = 0;
  
        // Hitung lebar dan tinggi baru yang lebih besar
        const enlargedW = nw * SCALE_FACTOR;
        const enlargedH = nh * SCALE_FACTOR;

        // Geser titik X dan Y agar kotak tetap berada di tengah objek setelah diperbesar
        const adjustedX = (nx - (enlargedW - nw) / 2) + OFFSET_X;
        const adjustedY = (ny - (enlargedH - nh) / 2) + OFFSET_Y;

        const left = Math.round(adjustedX * layout.width);
        const top = Math.round(adjustedY * layout.height);
        const width = Math.round(enlargedW * layout.width);
        const height = Math.round(enlargedH * layout.height);

        return (
          <View
            key={index}
            style={{
              position: 'absolute',
              borderColor: '#10b981',
              borderWidth: 4,
              borderRadius: 4,
              // RUMUS UTAMA: Persentase dikali Ukuran Layar HP
              left: left,
              top: top,
              width: width,
              height: height,
              zIndex: 10,
            }}
          >
            <Text style={{ color: '#10b981', fontSize: 10, }}>
              {/* {det.class} {Math.round(det.confidence * 100)}% */}
              {Math.round(det.confidence * 100)}%
            </Text>
          </View>
        );
      })}
      <View style={styles.statusWrap}>
        {/* INDIKATOR LAMPU */}
        <View 
          style={[
            styles.lightIndicator, 
            { backgroundColor: isDetected ? '#10b981' : 'rgba(255, 0, 0, 0.4)' }
          ]} 
        />

        {/* INDIKATOR STATUS KONEKSI */}
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
  
  indicatorContainer: {
    position: 'absolute',
    top: 50,
    width: '100%',
    alignItems: 'center',
    justifyContent: 'center',
  },

  lightIndicator: {
    width: 20,
    height: 20,
    borderRadius: 40,
    borderWidth: 2,
    borderColor: 'white',
    // Efek cahaya (shadow) saat terdeteksi
    shadowColor: "#10b981",
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 20,
    elevation: 10,
  },
  detectionText: {
    fontSize: 18,
    fontWeight: 'bold',
    textShadowColor: 'black',
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 5,
  },

  statusWrap: { flexDirection: 'row', top: 10, width: '100%', justifyContent: 'center', alignItems: 'center', gap: 3 },
  statusBadge: { paddingHorizontal: 15, paddingVertical: 5, borderRadius: 20, opacity: 0.8 },
  statusText: { color: 'white', fontWeight: 'bold', fontSize: 10 },
  headerControls: { position: 'absolute', top: 20, left: 20, right: 20, flexDirection: 'row', justifyContent: 'space-between' },
  miniBtn: { backgroundColor: 'rgba(0,0,0,0.5)', padding: 10, borderRadius: 10, borderWidth: 1, borderColor: 'white/20' },
  miniBtnText: { color: 'white', fontSize: 12, fontWeight: 'bold' }
});