# ExecuTorch Llama Android Demo App

We’re excited to share that the newly revamped Android demo app is live and includes many new updates to provide a more intuitive and smoother user experience with a chat use case! The primary goal of this app is to showcase how easily ExecuTorch can be integrated into an Android demo app and how to exercise the many features ExecuTorch and Llama models have to offer.

This app serves as a valuable resource to inspire your creativity and provide foundational code that you can customize and adapt for your particular use case.

Please dive in and start exploring our demo app today! We look forward to any feedback and are excited to see your innovative ideas.


## Key Concepts
From this demo app, you will learn many key concepts such as:
* How to prepare Llama models, build the ExecuTorch library, and model inferencing across delegates
* Expose the ExecuTorch library via JNI layer
* Familiarity with current ExecuTorch app-facing capabilities

The goal is for you to see the type of support ExecuTorch provides and feel comfortable with leveraging it for your use cases.

## Supporting Models
As a whole, the models that this app supports are (varies by delegate):
* Llama 3.2 1B/3B
* Llama Guard 3 1B
* Llama 3.1 8B
* Llama 3 8B
* Llama 2 7B
* LLaVA-1.5 vision model (only XNNPACK)


## Building the APK
First it’s important to note that currently ExecuTorch provides support across 3 delegates. Once you identify the delegate of your choice, select the README link to get a complete end-to-end instructions for environment set-up to exporting the models to build ExecuTorch libraries and apps to run on device:

| Delegate      | Resource |
| ------------- | ------------- |
| XNNPACK (CPU-based library)  | [link](https://github.com/pytorch/executorch/blob/main/examples/demo-apps/android/LlamaDemo/docs/delegates/xnnpack_README.md) |
| QNN (Qualcomm AI Accelerators)  | [link](https://github.com/pytorch/executorch/blob/main/examples/demo-apps/android/LlamaDemo/docs/delegates/qualcomm_README.md) |
| MediaTek (MediaTek AI Accelerators)  | [link](https://github.com/pytorch/executorch/blob/main/examples/demo-apps/android/LlamaDemo/docs/delegates/mediatek_README.md)  |

**WARNING** NDK r27 will cause issues like:
  ```
  java.lang.UnsatisfiedLinkError: dlopen failed: cannot locate symbol "_ZTVNSt6__ndk114basic_ifstreamIcNS_11char_traitsIcEEEE" referenced by "/data/app/~~F5IwquaXUZPdLpSEYA-JGA==/com.example.executorchllamademo-FSyx80gEhsQCsxz7hvS2Ew==/lib/arm64/libexecutorch.so"...
  ```
  Please use NDK version 26.3.11579264.

## How to Use the App

This section will provide the main steps to use the app, along with a code snippet of the ExecuTorch API.

For loading the app, development, and running on device we recommend Android Studio:
1. Open Android Studio and select "Open an existing Android Studio project" to open examples/demo-apps/android/LlamaDemo.
2. Run the app (^R). This builds and launches the app on the phone.

### Opening the App

Below are the UI features for the app.

Select the settings widget to get started with picking a model, its parameters and any prompts.
<p align="center">
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/opening_the_app_details.png" style="width:800px">
</p>



### Select Models and Parameters

Once you've selected the model, tokenizer, and model type you are ready to click on "Load Model" to have the app load the model and go back to the main Chat activity.
<p align="center">
      <img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/settings_menu.png" style="width:300px">
</p>



Optional Parameters:
* Temperature: Defaulted to 0, you can adjust the temperature for the model as well. The model will reload upon any adjustments.
* System Prompt: Without any formatting, you can enter in a system prompt. For example, "you are a travel assistant" or "give me a response in a few sentences".
* User Prompt: More for the advanced user, if you would like to manually input a prompt then you can do so by modifying the `{{user prompt}}`. You can also modify the special tokens as well. Once changed then go back to the main Chat activity to send.

#### ExecuTorch App API

```java
// Upon returning to the Main Chat Activity
mModule = new LlamaModule(
            ModelUtils.getModelCategory(mCurrentSettingsFields.getModelType()),
            modelPath,
            tokenizerPath,
            temperature);
int loadResult = mModule.load();
```

* `modelCategory`: Indicate whether it’s a text-only or vision model
* `modePath`: path to the .pte file
* `tokenizerPath`: path to the tokenizer .bin file
* `temperature`: model parameter to adjust the randomness of the model’s output


### User Prompt
Once model is successfully loaded then enter any prompt and click the send (i.e. generate) button to send it to the model.
<p align="center">
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/load_complete_and_start_prompt.png" style="width:300px">
</p>

You can provide it more follow-up questions as well.
<p align="center">
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/chat.png" style="width:300px">
</p>

#### ExecuTorch App API

```java
mModule.generate(prompt,sequence_length, MainActivity.this);
```
* `prompt`: User formatted prompt
* `sequence_length`: Number of tokens to generate in response to a prompt
* `MainActivity.this`: Indicate that the callback functions (OnResult(), OnStats()) are present in this class.

[*LLaVA-1.5: Only for XNNPACK delegate*]

For LLaVA-1.5 implementation, select the exported LLaVA .pte and tokenizer file in the Settings menu and load the model. After this you can send an image from your gallery or take a live picture along with a text prompt to the model.

<p align="center">
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/llava_example.png" style="width:300px">
</p>


### Output Generated
To show completion of the follow-up question, here is the complete detailed response from the model.
<p align="center">
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/chat_response.png" style="width:300px">
</p>

#### ExecuTorch App API

Ensure you have the following functions in your callback class that you provided in the `mModule.generate()`. For this example, it is `MainActivity.this`.
```java
  @Override
  public void onResult(String result) {
    //...result contains token from response
    //.. onResult will continue to be invoked until response is complete
  }

  @Override
  public void onStats(float tps) {
    //...tps (tokens per second) stats is provided by framework
  }

```

## Reporting Issues
If you encountered any bugs or issues following this tutorial please file a bug/issue here on [Github](https://github.com/pytorch/executorch/issues/new).
