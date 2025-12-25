import React from 'react';
import { SafeAreaView, SafeAreaProvider } from 'react-native-safe-area-context';
import { View, Text, Button, ScrollView } from 'react-native';

export default function PatchNote({ onBack }: any) {
  const notes = [
    "Apk ini masih berantakan UI/UX-nya",
    "Proses deteksi masih rada delay, tpi overall okelah wkwk",
    "Pastiin kacamata yg dideteksi bentuk framenya yg rada bulet",
  ];

  return (
    <SafeAreaProvider>
      <SafeAreaView className='flex-1 px-5 py-2 bg-[#1E1E1E] justify-between'>
        <ScrollView className='pl-3 gap-1'>
          <Text className='text-white text-2xl font-bold mb-4'>
            Patch Note v1.0.0
          </Text>

          {notes.map((note, index) => (
            <View key={index} className="flex-row mb-2">
              <Text className='text-white text-xl mr-2'>
                {index + 1}.
              </Text>
              <Text className='text-white text-xl flex-1'>
                {note}
              </Text>
            </View>
          ))}
        </ScrollView>

        <View className="mb-4">
          <Button 
            title="Kembali" 
            onPress={onBack} 
            color="#3b82f6" 
          />
        </View>
      </SafeAreaView>
    </SafeAreaProvider>
  );
}