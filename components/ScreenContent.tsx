import { useState, useEffect } from 'react';
import { Text, View, TouchableOpacity, Button } from 'react-native';
import { SafeAreaView, SafeAreaProvider } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
// import { useNavigation } from '@react-navigation/native';

import AntDesign from '@expo/vector-icons/AntDesign';
import MaterialCommunityIcons from '@expo/vector-icons/MaterialCommunityIcons';
import Entypo from '@expo/vector-icons/Entypo';

import CameraScreen from './Camera';
import CameraScreen2 from './Camera2';
import ServerConfig from './ServerConfig';
import PatchNote from './PatchNote';

type ScreenContentProps = {
  title: string;
  // path: string;
  children?: React.ReactNode;
};

export default function ScreenContent ({ title, children }: ScreenContentProps) {
  const [showCamera, setShowCamera] = useState(false);
  const [cameraMode, setCameraMode] = useState<'back' | 'front'>('back');
  const [wsUrl, setWsUrl] = useState<string | null>(null);
  const [page, setPage] = useState<'camera' | 'config' | 'manual'>('config');
  const toggleCamera = () => setShowCamera(!showCamera);
  // const switchCamera = () => setCameraMode(cameraMode === 'back' ? 'front' : 'back');
  const primaryIconSize = 30;
  const secondaryIconSize = 24;

  useEffect(() => {
    AsyncStorage.getItem('WS_URL').then(v => {
      if (v) {
        setWsUrl(v);
        setPage('camera');
      }
    });
  }, []);

  if (page === 'manual') {
    return <PatchNote onBack={() => setPage('camera')} />;
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
      <SafeAreaView className="flex-1 justify-between px-5 pb-2 bg-[#1E1E1E]">
        <View className='pt-8 gap-3'>
          <Text className='text-white text-3xl'>{title}</Text>
          <Text className='text-white font-thin'>Detect and find your glasses </Text>
        </View>

        {!showCamera && (
          <View className="flex flex-row justify-evenly items-center gap-2">
            <TouchableOpacity
              onPress={() => setPage('manual')} 
              activeOpacity={0.7}
              className='p-3 rounded-full'
            >
              <MaterialCommunityIcons 
                name="note-text-outline" 
                size={secondaryIconSize} color="white" 
              />
            </TouchableOpacity>
            <TouchableOpacity 
              onPress={toggleCamera} 
              activeOpacity={0.7}
              className='p-3 rounded-full bg-white'
            >
              <Entypo 
                name="magnifying-glass" 
                size={primaryIconSize} 
                color="black" 
              />
            </TouchableOpacity>
            <TouchableOpacity
              onPress={() => setPage('config')} 
              activeOpacity={0.7}
              className='p-3 rounded-full'
            >
              <AntDesign 
                name="setting" 
                size={secondaryIconSize} 
                color="white" 
              />
            </TouchableOpacity>
          </View>
        )}

        {showCamera && (
          // TAMBAHKAN flex-1 DI SINI agar kamera tidak berukuran 0
          <View className="flex-1 mt-4 rounded-xl overflow-hidden bg-black">
            <View className="flex-1 relative"> 
              <CameraScreen wsUrl={wsUrl} />

              {/* Kontrol Kamera */}
              <View className="absolute bottom-5 w-full flex-row justify-center items-center gap-5">
                <TouchableOpacity onPress={toggleCamera} className="bg-red-500 p-4 rounded-full">
                  <MaterialIcons name="close" size={24} color="white" />
                </TouchableOpacity>
                {/* <TouchableOpacity onPress={switchCamera} className="bg-gray-800 p-4 rounded-full">
                  <MaterialIcons name="flip-camera-ios" size={24} color="white" />
                </TouchableOpacity> */}
              </View>
            </View>
          </View>
        )}

        {!showCamera && children}
      </SafeAreaView>
    </SafeAreaProvider>
  );
};
