import React from 'react';
import { SafeAreaView, SafeAreaProvider } from 'react-native-safe-area-context';
import { View, Text, Button, ScrollView } from 'react-native';

export default function UserManual({ onBack }: any) {
  return (
    <SafeAreaProvider>
      <SafeAreaView className='flex-1 px-5 bg-[#1E1E1E] justify-between'>
        <View className='pl-3 gap-1'>
          <Text className='text-white text-xl font-bold'>
            User Manual
          </Text>
          <Text className='text-white text-xl'>
            1. Pastikan kamera mengarah ke objek
          </Text>
          <Text className='text-white text-xl'>
            2. Server harus aktif dan satu jaringan (jika ws://)
          </Text>
          <Text className='text-white text-xl'>
            3. Bounding box muncul otomatis saat objek terdeteksi
          </Text>
        </View>

        <View>
          <Button title="Kembali ke Kamera" onPress={onBack} />
        </View>
      </SafeAreaView>
    </SafeAreaProvider>
  );
}
