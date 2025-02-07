import { NativeEventEmitter, NativeModules } from 'react-native';

const { LLaMABridge } = NativeModules;

if (!LLaMABridge) {
  throw new Error('LLaMABridge native module is not available');
}

const eventEmitter = new NativeEventEmitter(LLaMABridge);

export interface LLaMABridgeInterface {
  initialize(modelPath: string, tokenizerPath: string): Promise<boolean>;
  generate(prompt: string, sequenceLength: number): Promise<boolean>;
  stop(): void;
  onToken(callback: (token: string) => void): () => void;
}

const Bridge: LLaMABridgeInterface = {
  initialize: LLaMABridge.initialize,
  generate: LLaMABridge.generate,
  stop: LLaMABridge.stop,
  onToken(callback) {
    return eventEmitter.addListener('onToken', callback).remove;
  }
};

export default Bridge;