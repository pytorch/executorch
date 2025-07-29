/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

const et = Module;
et.onRuntimeInitialized = () => {
    const model_button = document.getElementById("upload_model_button");
    model_button.addEventListener("click", openFilePickerModel);

    const image_button = document.getElementById("upload_image_button");
    image_button.addEventListener("click", openFilePickerImage);
}

let module = null;

function loadModelFile(file) {
    const reader = new FileReader();
    reader.onload = function(event) {
        const buffer = event.target.result;

        const mod = et.Module.load(buffer);
        const modelText = document.getElementById("model_text");

        try {
            mod.loadMethod("forward");
        } catch (e) {
            modelText.textContent = "Failed to load forward method: " + e;
            return;
        }

        const methodMeta = mod.getMethodMeta("forward");
        if (methodMeta.inputTags.length != 1) {
            modelText.textContent = "Error: Expected input size of 1, got " + methodMeta.inputTags.length;
            modelText.style.color = "red";
            return;
        }

        if (methodMeta.inputTags[0] != et.Tag.Tensor) {
            modelText.textContent = "Error: Expected input type to be Tensor, got " + methodMeta.inputTags[0].name;
            modelText.style.color = "red";
            return;
        }

        const inputMeta = methodMeta.inputTensorMeta[0];

        if (inputMeta.sizes[0] != 1 || inputMeta.sizes[1] != 3 || inputMeta.sizes[2] != 224 || inputMeta.sizes[3] != 224) {
            modelText.textContent = "Error: Expected input shape to be [1, 3, 224, 224], got " + inputMeta.sizes;
            modelText.style.color = "red";
            return;
        }

        if (inputMeta.scalarType != et.ScalarType.Float) {
            modelText.textContent = "Error: Expected input type to be Float, got " + inputMeta.scalarType.name;
            modelText.style.color = "red";
            return;
        }

        if (methodMeta.outputTags.length != 1) {
            modelText.textContent = "Error: Expected output size of 1, got " + methodMeta.outputTags.length;
            modelText.style.color = "red";
            return;
        }

        if (methodMeta.outputTags[0] != et.Tag.Tensor) {
            modelText.textContent = "Error: Expected output type to be Tensor, got " + methodMeta.outputTags[0].name;
            modelText.style.color = "red";
            return;
        }

        const outputMeta = methodMeta.outputTensorMeta[0];

        if (outputMeta.sizes[0] != 1 || outputMeta.sizes[1] != 1000) {
            modelText.textContent = "Error: Expected output shape to be [1, 1000], got " + outputMeta.sizes;
            modelText.style.color = "red";
            return;
        }

        if (outputMeta.scalarType != et.ScalarType.Float) {
            modelText.textContent = "Error: Expected output type to be Float, got " + outputMeta.scalarType.name;
            modelText.style.color = "red";
            return;
        }

        module = mod;
        modelText.textContent = 'Uploaded model: ' + file.name;
        modelText.style.color = null;
        document.getElementById("upload_image_button").disabled = false;
    };
    reader.readAsArrayBuffer(file);
}

function* generateTensorData(data) {
  for (let j = 0; j < 3; j++) {
    for (let i = 0; i < data.length; i += 4) {
      yield data[i + j] / 255.0;
    }
  }
}

function loadImageFile(file) {
    const img = new Image();
    img.onload = function() {
      const canvas = document.getElementById("canvas");
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const inputTensor = et.Tensor.fromIter([1, 3, 224, 224], generateTensorData(imageData.data));
      const output = module.forward(inputTensor);
      const argmax = output[0].data.reduce((iMax, elem, i, arr) => elem > arr[iMax] ? i : iMax, 0);
      document.getElementById("class_text").textContent = _IMAGENET_CATEGORIES[argmax];
    }
    img.src = URL.createObjectURL(file);
}

async function openFilePickerModel() {
  try {
    const [fileHandle] = await window.showOpenFilePicker({
      types: [{
        description: 'Model Files',
        accept: {
          'application/octet-stream': ['.pte'],
        },
      }],
      multiple: false, // Set to true for multiple file selection
    });
    const file = await fileHandle.getFile();
    loadModelFile(file);
  } catch (err) {
    if (err.name === 'AbortError') {
      // Handle user abort silently
    } else {
      console.error('File picker error:', err);
    }
  }
}

async function openFilePickerImage() {
  try {
    const [fileHandle] = await window.showOpenFilePicker({
      types: [{
        description: "Images",
        accept: {
          "image/*": [".png", ".gif", ".jpeg", ".jpg"],
        },
      }],
      multiple: false, // Set to true for multiple file selection
    });
    const file = await fileHandle.getFile();
    loadImageFile(file);
  } catch (err) {
    if (err.name === 'AbortError') {
      // Handle user abort silently
    } else {
      console.error('File picker error:', err);
    }
  }
}
