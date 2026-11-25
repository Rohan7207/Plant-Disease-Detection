import { StatusBar } from 'expo-status-bar';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  SafeAreaView,
  Image,
  Alert,
  ScrollView,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { useState } from 'react';

export default function App() {
  const [image, setImage] = useState<string | null>(null);
  const [showResult, setShowResult] = useState(false);

  // Select image from gallery
  const selectImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      setShowResult(false);
    }
  };

  // Take photo using camera
  const takePhoto = async () => {
    const permission =
      await ImagePicker.requestCameraPermissionsAsync();

    if (!permission.granted) {
      Alert.alert(
        'Camera Permission',
        'Please allow camera access to take a plant photo.'
      );
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      setShowResult(false);
    }
  };

  // Temporary mock analysis
  const analyzePlant = () => {
    if (!image) {
      Alert.alert('No Image', 'Please select or take a plant photo first.');
      return;
    }

    setShowResult(true);
  };

  // If result screen is active
  if (showResult && image) {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar style="dark" />

        <ScrollView showsVerticalScrollIndicator={false}>
          <View style={styles.header}>
            <Text style={styles.emoji}>🌱</Text>

            <Text style={styles.title}>
              Detection Result
            </Text>

            <Text style={styles.subtitle}>
              Sample result for UI testing
            </Text>
          </View>

          {/* Plant Image */}
          <View style={styles.resultCard}>
            <Image
              source={{ uri: image }}
              style={styles.resultImage}
            />

            <Text style={styles.resultLabel}>
              Crop
            </Text>

            <Text style={styles.resultValue}>
              Tomato
            </Text>

            <Text style={styles.resultLabel}>
              Possible Disease
            </Text>

            <Text style={styles.diseaseValue}>
              Tomato Leaf Disease
            </Text>

            <Text style={styles.resultLabel}>
              Confidence
            </Text>

            <Text style={styles.confidence}>
              94%
            </Text>
          </View>

          {/* Information */}
          <View style={styles.infoCard}>
            <Text style={styles.sectionTitle}>
              📝 Symptoms
            </Text>

            <Text style={styles.sectionText}>
              • Yellow or brown spots may appear on the leaves.
            </Text>

            <Text style={styles.sectionText}>
              • Leaves may become dry or damaged.
            </Text>

            <Text style={styles.sectionText}>
              • Plant growth may be affected.
            </Text>
          </View>

          {/* Prevention */}
          <View style={styles.infoCard}>
            <Text style={styles.sectionTitle}>
              🛡️ Prevention
            </Text>

            <Text style={styles.sectionText}>
              • Keep the leaves dry and provide good air circulation.
            </Text>

            <Text style={styles.sectionText}>
              • Remove severely affected plant material.
            </Text>

            <Text style={styles.sectionText}>
              • Monitor plants regularly for new symptoms.
            </Text>
          </View>

          {/* Management */}
          <View style={styles.infoCard}>
            <Text style={styles.sectionTitle}>
              💡 Management
            </Text>

            <Text style={styles.sectionText}>
              • Follow locally appropriate plant disease management
                guidance.
            </Text>

            <Text style={styles.sectionText}>
              • Avoid spreading potentially infected plant material.
            </Text>

            <Text style={styles.sectionText}>
              • Consult a qualified agricultural professional for
                serious or uncertain cases.
            </Text>
          </View>

          {/* Back Button */}
          <TouchableOpacity
            style={styles.backButton}
            onPress={() => setShowResult(false)}
          >
            <Text style={styles.backButtonText}>
              ← Back to Scan
            </Text>
          </TouchableOpacity>

          <View style={styles.warning}>
            <Text style={styles.warningText}>
              ⚠️ This is a sample UI result. It is not a real AI
              diagnosis. Real prediction will be connected to the
              backend later.
            </Text>
          </View>
        </ScrollView>
      </SafeAreaView>
    );
  }

  // Main screen
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="dark" />

      <ScrollView showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.emoji}>🌱</Text>

          <Text style={styles.title}>
            Plant Disease Detection
          </Text>

          <Text style={styles.subtitle}>
            Detect plant diseases using AI
          </Text>
        </View>

        {/* Main Card */}
        <View style={styles.card}>
          {image ? (
            <Image
              source={{ uri: image }}
              style={styles.plantImage}
            />
          ) : (
            <Text style={styles.cardEmoji}>🍃</Text>
          )}

          <Text style={styles.cardTitle}>
            {image ? 'Plant Image Selected' : 'Check Your Plant'}
          </Text>

          <Text style={styles.cardText}>
            {image
              ? 'Your image is ready for AI analysis.'
              : 'Take a photo or select an image of your plant to identify possible diseases.'}
          </Text>

          {/* Gallery */}
          <TouchableOpacity
            style={styles.button}
            onPress={selectImage}
          >
            <Text style={styles.buttonText}>
              🖼️ Select Plant Image
            </Text>
          </TouchableOpacity>

          {/* Camera */}
          <TouchableOpacity
            style={styles.cameraButton}
            onPress={takePhoto}
          >
            <Text style={styles.cameraButtonText}>
              📷 Take Plant Photo
            </Text>
          </TouchableOpacity>

          {/* Analyze */}
          {image && (
            <TouchableOpacity
              style={styles.analyzeButton}
              onPress={analyzePlant}
            >
              <Text style={styles.analyzeButtonText}>
                🤖 Analyze Plant
              </Text>
            </TouchableOpacity>
          )}
        </View>

        {/* How it works */}
        <View style={styles.info}>
          <Text style={styles.infoTitle}>
            How it works
          </Text>

          <Text style={styles.infoText}>
            1. 📸 Upload a plant image
          </Text>

          <Text style={styles.infoText}>
            2. 🤖 AI analyzes the image
          </Text>

          <Text style={styles.infoText}>
            3. 🔍 Disease is identified
          </Text>

          <Text style={styles.infoText}>
            4. 💡 Get plant information
          </Text>
        </View>

        {/* Features */}
        <View style={styles.features}>
          <Text style={styles.featuresTitle}>
            Features
          </Text>

          <Text style={styles.featureText}>
            🌿 Plant disease detection
          </Text>

          <Text style={styles.featureText}>
            📷 Camera and gallery support
          </Text>

          <Text style={styles.featureText}>
            🤖 AI-based analysis
          </Text>

          <Text style={styles.featureText}>
            📊 Disease information
          </Text>

          <Text style={styles.featureText}>
            📋 Detection history
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F4F9F4',
    paddingHorizontal: 20,
  },

  header: {
    alignItems: 'center',
    marginTop: 30,
    marginBottom: 25,
  },

  emoji: {
    fontSize: 50,
    marginBottom: 8,
  },

  title: {
    fontSize: 26,
    fontWeight: 'bold',
    color: '#1B5E20',
    textAlign: 'center',
  },

  subtitle: {
    fontSize: 15,
    color: '#666',
    marginTop: 6,
    textAlign: 'center',
  },

  card: {
    backgroundColor: '#FFFFFF',
    borderRadius: 20,
    padding: 22,
    alignItems: 'center',
    elevation: 5,
  },

  cardEmoji: {
    fontSize: 55,
    marginBottom: 10,
  },

  plantImage: {
    width: 220,
    height: 220,
    borderRadius: 15,
    marginBottom: 15,
  },

  cardTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#2E7D32',
    marginBottom: 10,
  },

  cardText: {
    fontSize: 15,
    color: '#666',
    textAlign: 'center',
    lineHeight: 22,
    marginBottom: 20,
  },

  button: {
    backgroundColor: '#2E7D32',
    paddingVertical: 14,
    borderRadius: 12,
    width: '100%',
    alignItems: 'center',
    marginBottom: 10,
  },

  buttonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: 'bold',
  },

  cameraButton: {
    borderWidth: 2,
    borderColor: '#2E7D32',
    paddingVertical: 12,
    borderRadius: 12,
    width: '100%',
    alignItems: 'center',
  },

  cameraButtonText: {
    color: '#2E7D32',
    fontSize: 16,
    fontWeight: 'bold',
  },

  analyzeButton: {
    backgroundColor: '#1B5E20',
    paddingVertical: 14,
    borderRadius: 12,
    width: '100%',
    alignItems: 'center',
    marginTop: 10,
  },

  analyzeButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: 'bold',
  },

  info: {
    marginTop: 20,
    backgroundColor: '#E8F5E9',
    padding: 18,
    borderRadius: 15,
  },

  infoTitle: {
    fontSize: 19,
    fontWeight: 'bold',
    color: '#1B5E20',
    marginBottom: 10,
  },

  infoText: {
    fontSize: 14,
    color: '#444',
    marginBottom: 6,
  },

  features: {
    marginTop: 20,
    marginBottom: 30,
    backgroundColor: '#FFFFFF',
    padding: 18,
    borderRadius: 15,
    elevation: 3,
  },

  featuresTitle: {
    fontSize: 19,
    fontWeight: 'bold',
    color: '#1B5E20',
    marginBottom: 12,
  },

  featureText: {
    fontSize: 14,
    color: '#444',
    marginBottom: 8,
  },

  resultCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 20,
    padding: 20,
    alignItems: 'center',
    elevation: 5,
  },

  resultImage: {
    width: 220,
    height: 220,
    borderRadius: 15,
    marginBottom: 20,
  },

  resultLabel: {
    fontSize: 14,
    color: '#777',
    marginTop: 10,
  },

  resultValue: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#2E7D32',
    marginTop: 4,
  },

  diseaseValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#C62828',
    textAlign: 'center',
    marginTop: 4,
  },

  confidence: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#1B5E20',
    marginTop: 4,
  },

  infoCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 15,
    padding: 18,
    marginTop: 15,
    elevation: 2,
  },

  sectionTitle: {
    fontSize: 19,
    fontWeight: 'bold',
    color: '#1B5E20',
    marginBottom: 10,
  },

  sectionText: {
    fontSize: 14,
    color: '#444',
    lineHeight: 21,
    marginBottom: 6,
  },

  backButton: {
    backgroundColor: '#2E7D32',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 20,
  },

  backButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: 'bold',
  },

  warning: {
    backgroundColor: '#FFF3CD',
    padding: 15,
    borderRadius: 12,
    marginTop: 15,
    marginBottom: 30,
  },

  warningText: {
    fontSize: 13,
    color: '#664D03',
    textAlign: 'center',
    lineHeight: 19,
  },
});