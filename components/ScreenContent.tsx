import { useState, useEffect } from 'react';
import { Text, View, TouchableOpacity, Button } from 'react-native';
import { SafeAreaView, SafeAreaProvider } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
// import { useNavigation } from '@react-navigation/native';

import CameraScreen from './Camera';
import CameraScreen2 from './Camera2';
import ServerConfig from './ServerConfig';
import UserManual from './UserManual';

type ScreenContentProps = {
  title: string;
  path: string;
  children?: React.ReactNode;
};

export const ScreenContent = ({ title, path, children }: ScreenContentProps) => {
  const [showCamera, setShowCamera] = useState(false);
  const [cameraMode, setCameraMode] = useState<'back' | 'front'>('back');
  const [wsUrl, setWsUrl] = useState<string | null>(null);
  const [page, setPage] = useState<'camera' | 'config' | 'manual'>('config');

  const toggleCamera = () => setShowCamera(!showCamera);
  const switchCamera = () => setCameraMode(cameraMode === 'back' ? 'front' : 'back');

  useEffect(() => {
    AsyncStorage.getItem('WS_URL').then(v => {
      if (v) {
        setWsUrl(v);
        setPage('camera');
      }
    });
  }, []);

  if (page === 'manual') {
    return <UserManual onBack={() => setPage('camera')} />;
  }

  if (!wsUrl || page === 'config') {
    return (
      <ServerConfig
        onSaved={(url: string) => {
          setWsUrl(url);
          setPage('camera');
        }}
      />
    );
  }

  return (
    <SafeAreaProvider>
      <SafeAreaView className="flex-1 justify-between px-5 bg-[#1E1E1E]">
        <View className='pt-8 gap-3'>
          <Text className='text-white text-3xl'>{title}</Text>
          <Text className='text-white font-thin'>Detect and find your perfect glasses </Text>
        </View>

        {!showCamera && (
          <View className="flex justify-center">
            {/* <View className='flex flex-row w-full justify-center pb-3'>
              <TouchableOpacity className='bg-[#242121] p-2 rounded-l w-1/2' onPress={() => navigation.navigate('ManualScreen')}>
                <Text className='text-white text-center'>Manual</Text>
              </TouchableOpacity>
              <TouchableOpacity className='bg-[#242121] p-2 rounded-r w-1/2'  onPress={() => navigation.navigate('ManualScreen')}>
                <Text className='text-white text-center'>Server</Text>
              </TouchableOpacity>
            </View> */}
            <TouchableOpacity
              onPress={toggleCamera}
              className="flex flex-row w-full items-center justify-center gap-2 bg-blue-500 py-3 rounded"
            >
              <MaterialIcons name="camera" size={30} color="white" />
              <Text className="text-white text-lg font-semibold">Open Camera</Text>
            </TouchableOpacity>
          </View>
        )}

        {showCamera && (
          // TAMBAHKAN flex-1 DI SINI agar kamera tidak berukuran 0
          <View className="flex-1 mt-4 rounded-xl overflow-hidden bg-black">
            <View className="flex-1 relative"> 
              <CameraScreen2
                wsUrl={wsUrl}
                onOpenManual={() => setPage('manual')}
                onChangeServer={() => setPage('config')}
              />

              {/* Kontrol Kamera */}
              <View className="absolute bottom-5 w-full flex-row justify-center items-center gap-5">
                <TouchableOpacity onPress={toggleCamera} className="bg-red-500 p-4 rounded-full">
                  <MaterialIcons name="close" size={24} color="white" />
                </TouchableOpacity>
                <TouchableOpacity onPress={switchCamera} className="bg-gray-800 p-4 rounded-full">
                  <MaterialIcons name="flip-camera-ios" size={24} color="white" />
                </TouchableOpacity>
              </View>
            </View>
          </View>
        )}

        {!showCamera && children}
      </SafeAreaView>
    </SafeAreaProvider>
  );
};
