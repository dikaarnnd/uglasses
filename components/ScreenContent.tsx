import { useState } from 'react';
import { Text, View, TouchableOpacity  } from 'react-native';
import { SafeAreaView, SafeAreaProvider } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';

// import CameraScreen from './Camera.production';
// import CameraScreenDev from './Camera.dev';
import MainCamera from './Camera';

type ScreenContentProps = {
  title: string;
  path: string;
  children?: React.ReactNode;
};

export const ScreenContent = ({ title, path, children }: ScreenContentProps) => {
  const [showCamera, setShowCamera] = useState(false);
  const [cameraMode, setCameraMode] = useState<'back' | 'front'>('back');

  const toggleCamera = () => {
    setShowCamera(!showCamera);
  };

  const switchCamera = () => {
    setCameraMode(cameraMode === 'back' ? 'front' : 'back');
  };

  return (
    <SafeAreaProvider>
      <SafeAreaView className="flex-1 px-5 bg-[#1E1E1E]">
        <View className='pt-8 gap-3'>
          <Text className='text-white text-3xl'>{title}</Text>
          <Text className='text-white font-thin'>Detect and find your perfect glasses </Text>
        </View>
        {/* Tombol untuk membuka kamera */}
        {!showCamera && (
          <View className="flex-1 justify-center content-end">
            <TouchableOpacity
              onPress={toggleCamera}
              className="flex flex-row w-full items-center justify-center gap-2 bg-blue-500 py-2 px-4 rounded-full"
            >
              <MaterialIcons name="camera" size={30} color="white" />
              <Text className="text-white text-lg font-semibold">Open Camera</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* Area kamera */}
        {showCamera && (
          <View className="flex-1 mt-4 rounded-xl overflow-hidden">
            <View className="relative">
              {/* Kamera berdasarkan environment */}
              <MainCamera />
              {/* <CameraScreen /> */}
              {/* <CameraScreenDev cameraMode={cameraMode} /> */}

              {/* Overlay dengan kontrol kamera */}
              <View className="absolute bottom-5 w-full flex-row justify-center items-center gap-5">
                {/* Tombol tutup kamera */}
                <TouchableOpacity
                  onPress={toggleCamera}
                  className="bg-red-500 p-4 rounded-full"
                >
                  <MaterialIcons name="close" size={24} color="white" />
                </TouchableOpacity>

                {/* Tombol ambil foto */}
                {/* <TouchableOpacity
                  onPress={() => {
                    // Logika untuk mengambil foto akan ditambahkan
                    console.log('Take photo');
                  }}
                  className="bg-white p-6 rounded-full border-4 border-gray-300"
                >
                  <View className="w-8 h-8 bg-gray-800 rounded-full" />
                </TouchableOpacity> */}

                {/* Tombol ganti kamera depan/belakang */}
                <TouchableOpacity
                  onPress={switchCamera}
                  className="bg-gray-800 p-4 rounded-full"
                >
                  <MaterialIcons name="flip-camera-ios" size={24} color="white" />
                </TouchableOpacity>
              </View>
            </View>
          </View>
        )}

        {/* Konten tambahan */}
        {!showCamera && children}
      </SafeAreaView>
    </SafeAreaProvider>
  );
};
// const styles = {
//   container: `items-center flex-1 justify-center bg-white`,
//   separator: `h-[1px] my-7 w-4/5 bg-gray-200`,
//   title: `text-xl font-bold text-red-100`,
// };
