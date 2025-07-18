/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

const et = Module;
et.onRuntimeInitialized = () => {
    const button = document.getElementById("button");
    button.addEventListener("click", openFilePicker);
}

function loadFile(file) {
    const reader = new FileReader();
    reader.onload = function(event) {
        const buffer = event.target.result;

        const module = et.Module.load(new Uint8Array(buffer));
        const inputs = [];
        const methodMeta = module.getMethodMeta("forward");
        methodMeta.inputTensorMeta.forEach((tensorInfo) => {
            const input = et.Tensor.ones(tensorInfo.sizes, tensorInfo.scalarType);
            inputs.push(input);
        });
        const outputs = module.forward(inputs);
        const output = outputs[0].data;
        document.getElementById("p1").innerHTML = "<pre><code>" + output.reduce((acc, num, index) => {
            return acc + num.toFixed(5).padStart(9, ' ') + ((index + 1) % 20 === 0 ? '<br>' : ' ');
        }, '') + "</code></pre>";
        module.delete()
        inputs.forEach((input) => {
            input.delete();
        });
        outputs.forEach((output) => {
            output.delete();
        });
        module.delete();
    };
    reader.readAsArrayBuffer(file);
}

async function openFilePicker() {
  try {
    const [fileHandle] = await window.showOpenFilePicker({
      types: [{
        description: 'Text Files',
        accept: {
          'application/octet-stream': ['.pte'],
        },
      }],
      multiple: false, // Set to true for multiple file selection
    });
    const file = await fileHandle.getFile();
    document.getElementById("p2").textContent = 'Selected file: ' + file.name;
    loadFile(file);
  } catch (err) {
    console.error('File picker error:', err);
  }
}
