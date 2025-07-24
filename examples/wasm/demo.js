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

        const mod = et.Module.load(new Uint8Array(buffer));
        try {
            mod.loadMethod("forward");
        } catch (e) {
            document.getElementById("model_text").textContent = "Failed to load forward method: " + e;
            return;
        }

        module = mod;
        document.getElementById("model_text").textContent = 'Uploaded model: ' + file.name;
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
