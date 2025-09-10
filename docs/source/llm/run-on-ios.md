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

Generate up to a given number of tokens from an initial prompt. The callback block is invoked once per token as it’s produced.

Objective-C:
```objectivec
NSError *error = nil;
BOOL success = [runner generate:@"Once upon a time"
                 sequenceLength:50
              withTokenCallback:^(NSString *token) {
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
  try runner.generate("Once upon a time", sequenceLength: 50) { token in
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

## Demo

Get hands-on with our [LLaMA iOS Demo App](llama-demo-ios.md) to see the LLM runtime APIs in action.
