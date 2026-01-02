import React from 'react';
import { SafeAreaView, SafeAreaProvider } from 'react-native-safe-area-context';
import { View, Text, ScrollView, TouchableOpacity } from 'react-native';

import Ionicons from '@expo/vector-icons/Ionicons';

export default function PatchNote({ onBack }: any) {
  const updateNote = [
    "Dynamic Bounding Box: Penanda kotak visual yang membingkai kacamata secara otomatis, memudahkan Anda mengetahui posisi tepat objek yang terdeteksi di dalam frame kamera secara presisi.",
    "Real-time Detection: Integrasi kamera ponsel dengan server AI menggunakan protokol WebSocket untuk transmisi data gambar tanpa putus.",
    "Haptic Feedback: Ponsel akan memberikan getaran (vibration) instan sebagai sinyal taktil saat kacamata berhasil diidentifikasi.",
    "Dynamic UI Indicator: Lampu status di layar akan berubah warna (Merah ke Hijau) sebagai indikator konfirmasi deteksi objek.",
    "Efficient AI Inference: Penggunaan model YOLO11n (ONNX) yang dioptimalkan agar proses deteksi tetap cepat meski dijalankan pada perangkat standar."
  ];

  return (
    <SafeAreaProvider>
      <SafeAreaView className='flex-1 px-5 py-2 bg-[#1E1E1E] justify-between'>
        <ScrollView className='pl-3 gap-1'>
          <Text className='text-white text-2xl font-bold mb-4'>
            v1.0.0: 🚀 Apa yang Baru?
          </Text>

          {updateNote.map((note, index) => (
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

        <TouchableOpacity
          className='flex flex-row items-center justify-center bg-blue-600 rounded-md'
          onPress={onBack}
        >
          <Text className='text-xl text-white font-bold p-3'>Kembali</Text>
        </TouchableOpacity>
      </SafeAreaView>
    </SafeAreaProvider>
  );
}