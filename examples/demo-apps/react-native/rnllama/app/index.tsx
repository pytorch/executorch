import React, { useState, useEffect, useRef } from 'react';
import {
  TextInput,
  Text,
  View,
  ScrollView,
  Alert,
  TouchableOpacity,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
  Pressable,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as DocumentPicker from 'expo-document-picker';
import * as Haptics from 'expo-haptics';
import LLaMABridge from '@/bridge/LlamaBridge';
import { Ionicons } from '@expo/vector-icons';

export default function Index() {
  const [prompt, setPrompt] = useState('');
  const [currentOutput, setCurrentOutput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isStopped, setIsStopped] = useState(false);
  const [modelPath, setModelPath] = useState('');
  const [modelName, setModelName] = useState('');
  const [tokenizerPath, setTokenizerPath] = useState('');
  const [tokenizerName, setTokenizerName] = useState('');
  const [isInitialized, setIsInitialized] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);
  const [history, setHistory] = useState<Array<{ input: boolean, text: string }>>([]);
  const scrollViewRef = useRef();

  useEffect(() => {
    const unsubscribe = LLaMABridge.onToken((token) => {
      if (!isStopped) {
        // Natural stop
        if (token === "<|eot_id|>") {
          setIsGenerating(false);
          setCurrentOutput(prev => {
            if (prev.trim()) {
              setHistory(prevHistory => [...prevHistory, { input: false, text: prev.trim() }]);
            }
            return '';
          });
          return;
        }
  
        // Skip template tokens
        if (token === formatPrompt('') ||
            token.includes("<|begin_of_text|>") ||
            token.includes("<|start_header_id|>") ||
            token.includes("<|end_header_id|>") ||
            token.includes("assistant")) {
          return;
        }
  
        // Add token without leading newlines
        setCurrentOutput(prev => prev + token.replace(/^\n+/, ''));
      }
    });
  
    return () => unsubscribe();
  }, [isStopped, currentOutput]);


  const formatPrompt = (text: string) => {
    return `<|begin_of_text|><|start_header_id|>user<|end_header_id|>${text.trim()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>`;
  };

  const handleGenerate = async () => {
    if (!isInitialized || !prompt.trim()) {
      return;
    }

    setIsStopped(false);
    const newPrompt = prompt.trim();
    setPrompt('');
    setIsGenerating(true);
    setCurrentOutput('');

    // Add the user message immediately
    const userMessage = { input: true, text: newPrompt };
    setHistory(prev => [...prev, userMessage]);

    try {
      const formattedPrompt = formatPrompt(newPrompt);
      await LLaMABridge.generate(formattedPrompt, 768);
    } catch (error) {
      console.error(error);
      Alert.alert('Error', 'Generation failed');
      setIsGenerating(false);
    }
  };

  const handleStop = () => {
    if (!isGenerating) return;
    
    setIsStopped(true);
    LLaMABridge.stop();
    
    if (currentOutput) {
      setHistory(prev => [...prev, { input: false, text: currentOutput.trim() }]);
    }
    setCurrentOutput('');
    setIsGenerating(false);
  };

  const selectModel = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync();
      if (result.assets && result.assets[0]) {
        setModelPath(result.assets[0].uri.replace('file://', ''));
        setModelName(result.assets[0].name);
        setIsInitialized(false);
      }
    } catch (err) {
      if (!DocumentPicker.isCancel(err)) {
        Alert.alert('Error', err.message);
      }
    }
  };

  const selectTokenizer = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync();
      if (result.assets && result.assets[0]) {
        setTokenizerPath(result.assets[0].uri.replace('file://', ''));
        setTokenizerName(result.assets[0].name);
        setIsInitialized(false);
      }
    } catch (err) {
      if (!DocumentPicker.isCancel(err)) {
        Alert.alert('Error', 'Failed to select tokenizer file');
      }
    }
  };

  const initializeLLaMA = async () => {
    // If already initialized, reset everything
    if (isInitialized) {
      setModelPath('');
      setModelName('');
      setTokenizerPath('');
      setTokenizerName('');
      setIsInitialized(false);
      setHistory([]);
      setCurrentOutput('');
      return;
    }
  
    if (!modelPath || !tokenizerPath) {
      Alert.alert('Error', 'Please select both model and tokenizer files first');
      return;
    }
  
    setIsInitializing(true);
    try {
      await LLaMABridge.initialize(modelPath, tokenizerPath);
      setIsInitialized(true);
      Alert.alert('Success', 'LLaMA initialized successfully');
    } catch (error) {
      console.error('Failed to initialize LLaMA:', error);
      Alert.alert('Error', 'Failed to initialize LLaMA');
      setModelPath('');
      setModelName('');
      setTokenizerPath('');
      setTokenizerName('');
    } finally {
      setIsInitializing(false);
    }
  };


  const handleClearHistory = () => {
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    setHistory([]);
    setCurrentOutput('');
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>rnllama</Text>
      </View>

      <View style={styles.setupBar}>
        <View style={styles.setupControls}>
          <TouchableOpacity
            style={[styles.setupButton, modelPath ? styles.setupComplete : styles.setupIncomplete]}
            onPress={selectModel}
          >
            <Ionicons name="cube-outline" size={20} color="#fff" />
            <Text style={styles.setupText}>
              {modelName ? modelName.substring(0, 15) + '...' : "Select Model"}
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.setupButton, tokenizerPath ? styles.setupComplete : styles.setupIncomplete]}
            onPress={selectTokenizer}
          >
            <Ionicons name="key-outline" size={20} color="#fff" />
            <Text style={styles.setupText}>
              {tokenizerName ? tokenizerName.substring(0, 15) + '...' : "Select Tokenizer"}
            </Text>
          </TouchableOpacity>
        </View>
        <TouchableOpacity
          style={[
            styles.initButton,
            isInitialized ? styles.setupComplete : styles.setupIncomplete,
            (!modelPath || !tokenizerPath || isInitializing) && styles.buttonDisabled
          ]}
          onPress={initializeLLaMA}
          disabled={!modelPath || !tokenizerPath || isInitializing}
        >
          {isInitializing ? (
            <ActivityIndicator size="small" color="#fff" />
          ) : (
            <Ionicons
              name={isInitialized ? "checkmark-circle-outline" : "power-outline"}
              size={24}
              color="#fff"
            />
          )}
        </TouchableOpacity>
      </View>

      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        style={styles.content}
        keyboardVerticalOffset={Platform.OS === "ios" ? 90 : 0}
      >
        <ScrollView
          ref={scrollViewRef}
          style={styles.chatContainer}
          contentContainerStyle={styles.chatContent}
          onContentSizeChange={() => scrollViewRef.current?.scrollToEnd({ animated: true })}
        >
          {!isInitialized ? (
            <View style={styles.initPrompt}>
              <Text style={styles.initPromptText}>
                Please select model and tokenizer files, then initialize LLaMA to begin chatting
              </Text>
            </View>
          ) : history.length === 0 ? (
            <Pressable
              style={styles.emptyState}
              onLongPress={handleClearHistory}
            >
              <Text style={styles.emptyStateText}>Start a conversation</Text>
              <Text style={styles.emptyStateHint}>Long press to clear history</Text>
            </Pressable>
          ) : (
            <Pressable onLongPress={handleClearHistory}>
              {history.map((message, index) => (
                <View
                  key={index}
                  style={[
                    message.input ? styles.sentMessage : styles.receivedMessage
                  ]}
                >
                  <Text style={message.input ? styles.sentMessageText : styles.receivedMessageText}>
                    {message.text}
                  </Text>
                </View>
              ))}
              {currentOutput && (
                <View style={styles.receivedMessage}>
                  <Text style={styles.receivedMessageText}>{currentOutput}</Text>
                </View>
              )}
            </Pressable>
          )}
        </ScrollView>

        <View style={styles.inputContainer}>
          <TextInput
            value={prompt}
            onChangeText={setPrompt}
            placeholder={isInitialized ? "Message" : "Initialize LLaMA to begin chatting"}
            placeholderTextColor="#666"
            multiline
            style={[styles.input, !isInitialized && styles.inputDisabled]}
            editable={isInitialized}
          />
          <TouchableOpacity
            style={[styles.sendButton, (!isInitialized || (!isGenerating && !prompt.trim())) && styles.buttonDisabled]}
            onPress={isGenerating ? handleStop : handleGenerate}
            disabled={!isInitialized || (!prompt.trim() && !isGenerating)} // This was backwards
          >
            <Ionicons
              name={isGenerating ? "stop-outline" : "send-outline"}
              size={24}
              color="#fff"
            />
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  header: {
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
    alignItems: "start",
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
  },
  headerSubtitle: {
    fontSize: 12,
    color: '#666',
    marginTop: 4,
  },
  setupBar: {
    flexDirection: 'row',
    padding: 12,
    backgroundColor: '#1A1A1A',
    alignItems: 'center',
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  setupControls: {
    flex: 1,
    flexDirection: 'row',
    gap: 8,
  },
  setupButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    padding: 8,
    borderRadius: 8,
    gap: 8,
  },
  setupComplete: {
    backgroundColor: '#1a5c2c',
  },
  setupIncomplete: {
    backgroundColor: '#333',
  },
  setupText: {
    color: '#fff',
    fontSize: 12,
    flex: 1,
  },
  initButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    marginLeft: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  content: {
    flex: 1,
  },
  chatContainer: {
    flex: 1,
  },
  chatContent: {
    padding: 16,
  },
  emptyState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
    opacity: 0.5,
  },
  emptyStateText: {
    color: '#666',
    fontSize: 16,
    marginTop: 12,
  },
  emptyStateHint: {
    color: '#666',
    fontSize: 12,
    marginTop: 8,
  },
  initPrompt: {
    padding: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  initPromptText: {
    color: '#666',
    textAlign: 'center',
    fontSize: 16,
  },
  sentMessage: {
    backgroundColor: '#0084FF',
    alignSelf: 'flex-end',
    maxWidth: '80%',
    borderRadius: 20,
    marginBottom: 12,
    padding: 12,
  },
  sentMessageText: {
    color: '#fff',
    fontSize: 16,
  },
  receivedMessage: {
    backgroundColor: '#333',
    alignSelf: 'flex-start',
    maxWidth: '80%',
    borderRadius: 20,
    marginBottom: 12,
    padding: 12,
  },
  receivedMessageText: {
    color: '#fff',
    fontSize: 16,
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 12,
    backgroundColor: '#1A1A1A',
    alignItems: 'flex-end',
  },
  input: {
    flex: 1,
    backgroundColor: '#333',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingTop: 12,
    paddingBottom: 12,
    marginRight: 8,
    color: '#fff',
    fontSize: 16,
    maxHeight: 120,
  },
  inputDisabled: {
    opacity: 0.5,
  },
  sendButton: {
    backgroundColor: '#0084FF',
    borderRadius: 20,
    width: 44,
    height: 44,
    alignItems: 'center',
    justifyContent: 'center',
  },
  buttonDisabled: {
    opacity: 0.5,
  },
});