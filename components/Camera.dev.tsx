import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, View, Text } from 'react-native';
import { CameraView, useCameraPermissions, CameraType } from 'expo-camera';

type CameraScreenDevProps = {
  cameraMode?: 'back' | 'front';
};

export default function CameraScreenDev({ cameraMode = 'back' }: CameraScreenDevProps) {
  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing] = useState<CameraType>(cameraMode);

  useEffect(() => {
    setFacing(cameraMode);
  }, [cameraMode]);

  useEffect(() => {
    if (!permission?.granted) {
      requestPermission();
    }
  }, [permission]);

  if (!permission) {
    return <View style={styles.container}><Text>Loading...</Text></View>;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text>Camera permission is required</Text>
      </View>
    );
  }

  return (
    <CameraView style={styles.camera} facing={facing}>
      {/* Camera UI bisa ditambahkan di sini */}
    </CameraView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'black',
  },
  camera: {
    width: '100%',
    height: '100%',
  },
});