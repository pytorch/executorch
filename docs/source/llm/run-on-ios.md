# Running LLMs on iOS

ExecuTorch’s LLM-specific runtime components provide an experimental Objective-C and Swift components around the core C++ LLM runtime.

## Prerequisites

Make sure you have a model and tokenizer files ready, as described in the prerequisites section of the [Running LLMs with C++](run-with-c-plus-plus.md) guide.

## Runtime API

Once linked against the [`executorch_llm`](../using-executorch-ios.md) framework, you can import the necessary components.

### Importing

Objective-C:
```objectivec
#import <ExecuTorchLLM/ExecuTorchLLM.h>
```

Swift:
```swift
import ExecuTorchLLM
```

### TextLLMRunner

The `ExecuTorchLLMTextRunner` class (bridged to Swift as `TextLLMRunner`) provides a simple Objective-C/Swift interface for loading a text-generation model, configuring its tokenizer with custom special tokens, generating token streams, and stopping execution.
This API is experimental and subject to change.

#### Initialization

Create a runner by specifying paths to your serialized model (`.pte`) and tokenizer data, plus an array of special tokens to use during tokenization.
Initialization itself is lightweight and doesn’t load the program data immediately.

Objective-C:
```objectivec
NSString *modelPath     = [[NSBundle mainBundle] pathForResource:@"llama-3.2-instruct" ofType:@"pte"];
NSString *tokenizerPath = [[NSBundle mainBundle] pathForResource:@"tokenizer" ofType:@"model"];
NSArray<NSString *> *specialTokens = @[ @"<|bos|>", @"<|eos|>" ];

ExecuTorchLLMTextRunner *runner = [[ExecuTorchLLMTextRunner alloc] initWithModelPath:modelPath
                                                                       tokenizerPath:tokenizerPath
                                                                       specialTokens:specialTokens];
```

Swift:
```swift
let modelPath     = Bundle.main.path(forResource: "llama-3.2-instruct", ofType: "pte")!
let tokenizerPath = Bundle.main.path(forResource: "tokenizer", ofType: "model")!
let specialTokens = ["<|bos|>", "<|eos|>"]

let runner = TextLLMRunner(
  modelPath: modelPath,
  tokenizerPath: tokenizerPath,
  specialTokens: specialTokens
)
```

#### Loading

Explicitly load the model before generation to avoid paying the load cost during your first `generate` call.

Objective-C:
```objectivec
NSError *error = nil;
BOOL success = [runner loadWithError:&error];
if (!success) {
  NSLog(@"Failed to load: %@", error);
}
```

Swift:
```swift
do {
  try runner.load()
} catch {
  print("Failed to load: \(error)")
}
```

#### Generating

Generate tokens from an initial prompt, configured with an `ExecuTorchLLMConfig` object. The callback block is invoked once per token as it’s produced.

Objective-C:
```objectivec
ExecuTorchLLMConfig *config = [[ExecuTorchLLMConfig alloc] initWithBlock:^(ExecuTorchLLMConfig *c) {
  c.temperature = 0.8;
  c.sequenceLength = 2048;
}];

NSError *error = nil;
BOOL success = [runner generateWithPrompt:@"Once upon a time"
                                   config:config
                            tokenCallback:^(NSString *token) {
                              NSLog(@"Generated token: %@", token);
                            }
                                    error:&error];
if (!success) {
  NSLog(@"Generation failed: %@", error);
}
```

Swift:
```swift
do {
  try runner.generate("Once upon a time", Config {
    $0.temperature = 0.8
    $0.sequenceLength = 2048
  }) { token in
    print("Generated token:", token)
  }
} catch {
  print("Generation failed:", error)
}
```

#### Stopping Generation

If you need to interrupt a long‐running generation, call:

Objective-C:
```objectivec
[runner stop];
```

Swift:
```swift
runner.stop()
```

#### Resetting

To clear the prefilled tokens from the KV cache and reset generation stats, call:

Objective-C:
```objectivec
[runner reset];
```

Swift:
```swift
runner.reset()
```

### MultimodalRunner

The `ExecuTorchLLMMultimodalRunner` class (bridged to Swift as `MultimodalRunner`) provides an interface for loading and running multimodal models that can accept a sequence of text, image, and audio inputs.

#### Multimodal Inputs

Inputs are provided as an array of `ExecuTorchLLMMultimodalInput` (or `MultimodalInput` in Swift). You can create inputs from String for text, `ExecuTorchLLMImage` for images (`Image` in Swift), and `ExecuTorchLLMAudio` for audio features (`Audio`) in Swift.

Objective-C:
```objectivec
ExecuTorchLLMMultimodalInput *textInput = [ExecuTorchLLMMultimodalInput inputWithText:@"What's in this image?"];

NSData *imageData = ...; // Your raw image bytes
ExecuTorchLLMImage *image = [[ExecuTorchLLMImage alloc] initWithData:imageData width:336 height:336 channels:3];
ExecuTorchLLMMultimodalInput *imageInput = [ExecuTorchLLMMultimodalInput inputWithImage:image];
```

Swift:
```swift
let textInput = MultimodalInput("What's in this image?")

let imageData: Data = ... // Your raw image bytes
let image = Image(data: imageData, width: 336, height: 336, channels: 3)
let imageInput = MultimodalInput(image)

let audioFeatureData: Data = ... // Your raw audio feature bytes
let audio = Audio(float: audioFeatureData, batchSize: 1, bins: 128, frames: 3000)
let audioInput = MultimodalInput(audio)
```

#### Initialization

Create a runner by specifying the paths to your multimodal model and its tokenizer.

Objective-C:
```objectivec
NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"llava" ofType:@"pte"];
NSString *tokenizerPath = [[NSBundle mainBundle] pathForResource:@"llava_tokenizer" ofType:@"bin"];

ExecuTorchLLMMultimodalRunner *runner = [[ExecuTorchLLMMultimodalRunner alloc] initWithModelPath:modelPath
                                                                                   tokenizerPath:tokenizerPath];
```

Swift:
```swift
let modelPath = Bundle.main.path(forResource: "llava", ofType: "pte")!
let tokenizerPath = Bundle.main.path(forResource: "llava_tokenizer", ofType: "bin")!

let runner = MultimodalRunner(modelPath: modelPath, tokenizerPath: tokenizerPath)
```

#### Loading

Explicitly load the model before generation.

Objective-C:
```objectivec
NSError *error = nil;
BOOL success = [runner loadWithError:&error];
if (!success) {
  NSLog(@"Failed to load: %@", error);
}
```

Swift:
```swift
do {
  try runner.load()
} catch {
  print("Failed to load: \(error)")
}
```

#### Generating

Generate tokens from an ordered array of multimodal inputs.

Objective-C:
```objectivec
NSArray<ExecuTorchLLMMultimodalInput *> *inputs = @[textInput, imageInput];

ExecuTorchLLMConfig *config = [[ExecuTorchLLMConfig alloc] initWithBlock:^(ExecuTorchLLMConfig *c) {
  c.sequenceLength = 768;
}];

NSError *error = nil;
BOOL success = [runner generateWithInputs:inputs
                                   config:config
                            tokenCallback:^(NSString *token) {
                              NSLog(@"Generated token: %@", token);
                            }
                                    error:&error];
if (!success) {
  NSLog(@"Generation failed: %@", error);
}
```

Swift:
```swift
let inputs = [textInput, imageInput]

do {
  try runner.generate(inputs, Config {
    $0.sequenceLength = 768
  }) { token in
    print("Generated token:", token)
  }
} catch {
  print("Generation failed:", error)
}
```

#### Stopping and Resetting

The stop and reset methods for `MultimodalRunner` behave identically to those on `TextRunner`.

## Demo

Get hands-on with our [etLLM iOS Demo App](https://github.com/meta-pytorch/executorch-examples/tree/main/llm/apple) to see the LLM runtime APIs in action.
