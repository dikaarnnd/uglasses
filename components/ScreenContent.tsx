import { useState } from 'react';
import { Text, View, TouchableOpacity  } from 'react-native';
import { SafeAreaView, SafeAreaProvider } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';

import MainCamera from './Camera';

type ScreenContentProps = {
  title: string;
  path: string;
  children?: React.ReactNode;
};

export const ScreenContent = ({ title, path, children }: ScreenContentProps) => {
  const [showCamera, setShowCamera] = useState(false);
  const [cameraMode, setCameraMode] = useState<'back' | 'front'>('back');

  const toggleCamera = () => setShowCamera(!showCamera);
  const switchCamera = () => setCameraMode(cameraMode === 'back' ? 'front' : 'back');

  return (
    <SafeAreaProvider>
      <SafeAreaView className="flex-1 px-5 bg-[#1E1E1E]">
        <View className='pt-8 gap-3'>
          <Text className='text-white text-3xl'>{title}</Text>
          <Text className='text-white font-thin'>Detect and find your perfect glasses </Text>
        </View>

        {!showCamera && (
          <View className="flex-1 justify-center">
            <TouchableOpacity
              onPress={toggleCamera}
              className="flex flex-row w-full items-center justify-center gap-2 bg-blue-500 py-3 rounded-full"
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
              <MainCamera />

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
