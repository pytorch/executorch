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

  const [modelPath, setModelPath] = useState('');
  const [modelName, setModelName] = useState('');
  const [tokenizerPath, setTokenizerPath] = useState('');
  const [tokenizerName, setTokenizerName] = useState('');

  const [isInitialized, setIsInitialized] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);

  const [history, setHistory] = useState<Array<{
    input: boolean, text: string, stats?: {
      tokens: number;
      time: number;
    };
  }>>([]);

  const [modelLoadTime, setModelLoadTime] = useState<number | null>(null);

  const [currentGenerationStartTime, setCurrentGenerationStartTime] = useState<number | null>(null);
  const [currentNumTokens, setCurrentNumTokens] = useState(0);

  const scrollViewRef = useRef();

  const handleGenerationStopped = () => {
    LLaMABridge.stop();
    const generationEndTime = Date.now();
    const stats = currentGenerationStartTime !== null ? {
      tokens: currentNumTokens,
      time: generationEndTime - currentGenerationStartTime
    } : undefined;

    setHistory(prevHistory => [...prevHistory, { input: false, text: currentOutput.trim(), stats }]);
    setIsGenerating(false);
    setCurrentOutput('');
    setCurrentNumTokens(0);
  }

  useEffect(() => {
    const unsubscribe = LLaMABridge.onToken((token) => {
      if (isGenerating) {
        // Natural stop
        if (token === "<|eot_id|>") {
          handleGenerationStopped();
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
        if (currentNumTokens === 0) {
          setCurrentOutput(prev => prev + token.replace(/^\n+/, ''));
        } else {
          setCurrentOutput(prev => prev + token);
        }

        setCurrentNumTokens(prev => prev + 1);
      }
    });

    return () => unsubscribe();
  }, [isGenerating, currentNumTokens]);


  const formatPrompt = (text: string) => {
    return `<|begin_of_text|><|start_header_id|>user<|end_header_id|>${text.trim()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>`;
  };

  const handleGenerate = async () => {
    if (!isInitialized || !prompt.trim()) {
      return;
    }

    setCurrentGenerationStartTime(Date.now());

    const newPrompt = prompt.trim();
    setPrompt('');
    setIsGenerating(true);

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
      const startTime = Date.now();
      await LLaMABridge.initialize(modelPath, tokenizerPath);
      const modelLoadTime = Date.now() - startTime;
      setModelLoadTime(modelLoadTime);

      setIsInitialized(true);
      Alert.alert('Success', `Model loaded in ${(modelLoadTime / 1000).toFixed(1)}s`);
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
        <View style={styles.initContainer}>

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
      {modelLoadTime && (
        <Text style={styles.emptyStateText}>
          Model loading took {(modelLoadTime / 1000).toFixed(1)}s
        </Text>
      )}
      <Text style={styles.emptyStateText}>Start a conversation</Text>
      <Text style={styles.emptyStateHint}>Long press to clear history</Text>
    </Pressable>
  ) : (
    <Pressable onLongPress={handleClearHistory}>
  {modelLoadTime && (
    <View style={styles.loadTimeContainer}>
      <Text style={styles.loadTimeMessage}>
        Model loading took {(modelLoadTime / 1000).toFixed(1)}s
      </Text>
    </View>
  )}
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
      {!message.input && message.stats && (
        <Text style={styles.tokensPerSecondText}>
          {`Tokens/sec: ${(message.stats.tokens / (message.stats.time / 1000)).toFixed(2)}`}
        </Text>
      )}
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
            onPress={isGenerating ? handleGenerationStopped : handleGenerate}
            disabled={!isInitialized || (!prompt.trim() && !isGenerating)}
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
  loadTimeContainer: {
    alignItems: 'center',
    marginBottom: 16,
    padding: 8,
    backgroundColor: '#1A1A1A',
    borderRadius: 8,
  },
  loadTimeMessage: {
    color: '#666',
    fontSize: 14,
  },
  tokensPerSecondText: {
    color: '#666',
    fontSize: 12,
    marginTop: 4,
  },
  header: {
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
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