import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, TouchableOpacity, Alert } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

export default function ServerConfig({ onSaved }: any) {
  const [ipAddress, setIpAddress] = useState('');

  useEffect(() => {
    // Saat memuat, kita coba ambil URL lama dan ekstrak IP-nya saja (opsional)
    AsyncStorage.getItem('WS_URL').then(v => {
      if (v) {
        // Ekstrak IP dari ws://IP:8000/ws
        const match = v.match(/\/\/([^:/]+)/);
        if (match) setIpAddress(match[1]);
      }
    });
  }, []);

  const save = async () => {
    // Validasi input minimal (memastikan tidak kosong)
    if (!ipAddress.trim()) {
      Alert.alert('Error', 'Silakan masukkan IP Address server Anda');
      return;
    }

    // PROSES BALIK LAYAR: Melengkapi URL
    // Format: ws:// + [IP] + :8000/ws
    const fullWsUrl = `ws://${ipAddress.trim()}:8000/ws`;

    await AsyncStorage.setItem('WS_URL', fullWsUrl);
    onSaved(fullWsUrl); // Mengirim URL lengkap ke ScreenContent
  };

  return (
    <View className="flex-1 bg-[#1E1E1E] justify-center px-8">
      <Text className="text-white text-2xl font-bold mb-2 text-center">
        Konfigurasi Server
      </Text>
      <Text className="text-gray-400 mb-8 text-center font-thin">
        Masukkan IP Address laptop/server Anda
      </Text>

      <View className="bg-white/5 p-6 rounded-3xl border border-white/10 shadow-xl">
        <Text className="text-blue-400 mb-2 font-bold text-xs uppercase ml-1">
          IP Address
        </Text>
        <TextInput
          value={ipAddress}
          onChangeText={setIpAddress}
          placeholder="Contoh: 192.168.1.20"
          placeholderTextColor="#555"
          keyboardType="numeric" // Memudahkan input angka di HP
          autoCapitalize="none"
          className="text-black text-lg p-4 bg-white rounded-xl border border-white/5 mb-6"
        />

        <TouchableOpacity 
          onPress={save}
          className="bg-blue-600 py-4 rounded-2xl items-center shadow-lg shadow-blue-500/20"
        >
          <Text className="text-white font-bold text-lg">Hubungkan Kamera</Text>
        </TouchableOpacity>
      </View>

      <Text className="text-gray-500 text-[10px] mt-8 text-center italic">
        Pastikan server FastAPI Anda berjalan di port 8000
      </Text>
    </View>
  );
}