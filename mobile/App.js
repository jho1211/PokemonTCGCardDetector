import { CameraView, useCameraPermissions } from "expo-camera";
import * as ImageManipulator from "expo-image-manipulator";
import { StatusBar } from "expo-status-bar";
import React, { useMemo, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Image,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";

import { identifyCard } from "./src/api/client";
import { addHistoryItem, getHistory } from "./src/storage/history";

export default function App() {
  const cameraRef = useRef(null);
  const [permission, requestPermission] = useCameraPermissions();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [screen, setScreen] = useState("camera");
  const [capturedImageUri, setCapturedImageUri] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);

  React.useEffect(() => {
    getHistory().then(setHistory).catch(() => {
      // Non-blocking storage read.
    });
  }, []);

  const marketPriceLabel = useMemo(() => {
    if (!result?.card) {
      return "-";
    }
    const price = result.card.market_price_usd;
    return typeof price === "number" ? `$${price.toFixed(2)}` : "Price unavailable";
  }, [result]);

  const hasCardMatch = Boolean(result?.success && result?.card);
  const hasNoMatch = Boolean(result && result.success === false);

  const resultTitle = useMemo(() => {
    if (hasCardMatch) {
      return result.card.name;
    }
    if (hasNoMatch) {
      return "No confident match found";
    }
    return "";
  }, [hasCardMatch, hasNoMatch, result]);

  const onCaptureAndIdentify = async () => {
    if (!cameraRef.current || isSubmitting || screen !== "camera") {
      return;
    }

    try {
      setIsSubmitting(true);
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.8 });
      const processed = await ImageManipulator.manipulateAsync(
        photo.uri,
        [{ resize: { width: 1080 } }],
        { compress: 0.85, format: ImageManipulator.SaveFormat.JPEG }
      );
      setCapturedImageUri(processed.uri);
      setScreen("processing");

      const apiResult = await identifyCard(processed.uri);
      setResult(apiResult);
      setScreen("result");

      console.log(apiResult);

      const historyItem = {
        id: `${apiResult.success ? apiResult.card?.id || "match" : "no-match"}-${Date.now()}`,
        scannedAt: new Date().toISOString(),
        result: apiResult,
      };
      const updated = await addHistoryItem(historyItem);
      setHistory(updated);
    } catch (error) {
      setScreen("camera");
      setCapturedImageUri(null);
      setResult(null);
      Alert.alert("Scan failed", error.message || "Unable to identify card.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const onRetakePhoto = () => {
    if (isSubmitting) {
      return;
    }

    setCapturedImageUri(null);
    setResult(null);
    setScreen("camera");
  };

  if (!permission) {
    return <SafeAreaView style={styles.center}><ActivityIndicator /></SafeAreaView>;
  }

  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.center}>
        <Text style={styles.heading}>Camera permission required</Text>
        <Pressable style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Camera Access</Text>
        </Pressable>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="dark" />
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {screen === "camera" ? (
          <>
            <Text style={styles.heading}>Pokemon TCG Identifier</Text>

            <View style={styles.cameraContainer}>
              <CameraView ref={cameraRef} style={styles.camera} facing="back" />
            </View>

            <Pressable style={[styles.button, isSubmitting && styles.buttonDisabled]} onPress={onCaptureAndIdentify} disabled={isSubmitting}>
              <Text style={styles.buttonText}>Identify Card</Text>
            </Pressable>

            <Pressable style={styles.secondaryButton} onPress={() => setShowHistory((v) => !v)}>
              <Text style={styles.secondaryButtonText}>{showHistory ? "Hide History" : "Show History"}</Text>
            </Pressable>

            {showHistory ? (
              <View style={styles.historyContainer}>
                {history.length === 0 ? <Text style={styles.muted}>No scans yet.</Text> : null}
                {history.map((item) => (
                  <View key={item.id} style={styles.historyItem}>
                    <View style={styles.historyHeaderRow}>
                      <Text style={styles.historyName}>
                        {item.result?.success && item.result?.card ? item.result.card.name : "No confident match"}
                      </Text>
                      <View
                        style={[
                          styles.historyBadge,
                          item.result?.success ? styles.historyBadgeMatched : styles.historyBadgeNoMatch,
                        ]}
                      >
                        <Text style={styles.historyBadgeText}>{item.result?.success ? "Matched" : "No Match"}</Text>
                      </View>
                    </View>
                    <Text style={styles.muted}>
                      {item.result?.success && item.result?.card
                        ? `${item.result.card.collection} • #${item.result.card.collector_number}`
                        : `Reason: ${item.result?.no_match?.reason || "unavailable"}`}
                    </Text>
                    <Text style={styles.muted}>{new Date(item.scannedAt).toLocaleString()}</Text>
                  </View>
                ))}
              </View>
            ) : null}
          </>
        ) : null}

        {screen === "processing" ? (
          <>
            <Text style={styles.heading}>Identifying Card...</Text>
            <View style={styles.cameraContainer}>
              {capturedImageUri ? <Image source={{ uri: capturedImageUri }} style={styles.camera} resizeMode="cover" /> : null}
            </View>
            <ActivityIndicator style={styles.loading} size="large" />
            <Text style={styles.progressText}>Analyzing image and matching card details. Please wait.</Text>
          </>
        ) : null}

        {screen === "result" ? (
          <>
            <Text style={styles.heading}>Identification Result</Text>

            <View style={styles.cameraContainer}>
              {capturedImageUri ? <Image source={{ uri: capturedImageUri }} style={styles.camera} resizeMode="cover" /> : null}
            </View>

            {result ? (
              <View style={styles.resultCard}>
                <Text style={styles.resultTitle}>{resultTitle}</Text>
                {hasCardMatch && result.card.image_url ? <Image source={{ uri: result.card.image_url }} style={styles.cardImage} resizeMode="cover" /> : null}
                {hasCardMatch ? (
                  <>
                    <Text style={styles.resultLine}>Collection: {result.card.collection}</Text>
                    <Text style={styles.resultLine}>Collector Number: {result.card.collector_number}</Text>
                    <Text style={styles.resultLine}>Market Price (USD): {marketPriceLabel}</Text>
                  </>
                ) : null}
                <Text style={styles.confidence}>Confidence: {(result.confidence * 100).toFixed(0)}% ({result.confidence_label})</Text>
                {result.warning ? <Text style={styles.warning}>{result.warning}</Text> : null}
                {hasNoMatch && result.no_match?.reason ? (
                  <Text style={styles.resultLine}>Reason: {result.no_match.reason.replace(/_/g, " ")}</Text>
                ) : null}
                {hasNoMatch && Array.isArray(result.no_match?.suggestions) && result.no_match.suggestions.length > 0 ? (
                  <View style={styles.suggestionContainer}>
                    {result.no_match.suggestions.map((tip, idx) => (
                      <Text key={`${tip}-${idx}`} style={styles.suggestionLine}>- {tip}</Text>
                    ))}
                  </View>
                ) : null}
              </View>
            ) : null}

            <Pressable style={[styles.button, isSubmitting && styles.buttonDisabled]} onPress={onRetakePhoto} disabled={isSubmitting}>
              <Text style={styles.buttonText}>Retake</Text>
            </Pressable>
          </>
        ) : null}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f4efe5",
  },
  scrollContent: {
    padding: 16,
    paddingBottom: 28,
    gap: 12,
  },
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#f4efe5",
    padding: 24,
  },
  heading: {
    fontSize: 28,
    fontWeight: "800",
    color: "#1f2f3f",
    marginBottom: 10,
  },
  cameraContainer: {
    borderRadius: 14,
    overflow: "hidden",
    borderWidth: 2,
    borderColor: "#1f2f3f",
  },
  camera: {
    width: "100%",
    height: 300,
  },
  button: {
    backgroundColor: "#ce2f2f",
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: "center",
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  buttonText: {
    color: "white",
    fontWeight: "700",
    fontSize: 16,
  },
  secondaryButton: {
    backgroundColor: "#203040",
    paddingVertical: 12,
    borderRadius: 12,
    alignItems: "center",
  },
  secondaryButtonText: {
    color: "white",
    fontWeight: "700",
    fontSize: 15,
  },
  loading: {
    marginVertical: 8,
  },
  progressText: {
    color: "#32485e",
    fontSize: 15,
    textAlign: "center",
    marginTop: 4,
  },
  resultCard: {
    backgroundColor: "#ffffff",
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: "#d4d8dc",
    gap: 6,
  },
  resultTitle: {
    fontSize: 22,
    fontWeight: "800",
    color: "#1f2f3f",
  },
  cardImage: {
    width: "100%",
    height: 320,
    borderRadius: 10,
    marginVertical: 8,
  },
  resultLine: {
    fontSize: 16,
    color: "#213142",
  },
  confidence: {
    marginTop: 6,
    fontSize: 14,
    fontWeight: "700",
    color: "#1f2f3f",
  },
  warning: {
    marginTop: 4,
    color: "#9f1c1c",
    fontWeight: "700",
  },
  suggestionContainer: {
    marginTop: 6,
    gap: 4,
  },
  suggestionLine: {
    color: "#32485e",
    fontSize: 14,
  },
  historyContainer: {
    gap: 10,
  },
  historyItem: {
    backgroundColor: "#fff",
    borderRadius: 10,
    padding: 10,
    borderWidth: 1,
    borderColor: "#d4d8dc",
    gap: 4,
  },
  historyHeaderRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  historyName: {
    fontSize: 16,
    fontWeight: "700",
    color: "#1f2f3f",
  },
  historyBadge: {
    borderRadius: 999,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  historyBadgeMatched: {
    backgroundColor: "#d8f3dc",
  },
  historyBadgeNoMatch: {
    backgroundColor: "#ffe3e3",
  },
  historyBadgeText: {
    fontSize: 12,
    fontWeight: "700",
    color: "#2f3e4e",
  },
  muted: {
    color: "#5f6b76",
  },
});
