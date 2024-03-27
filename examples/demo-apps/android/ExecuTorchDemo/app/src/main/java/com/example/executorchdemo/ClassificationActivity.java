/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchdemo;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import java.io.IOException;
import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;

public class ClassificationActivity extends Activity implements Runnable {

  private void openSegmentationActivity() {
    Intent intent = new Intent(this, MainActivity.class);
    startActivity(intent);
  }

  private void populateBitmap(String file) {
    Bitmap bitmap = null;
    try {
      bitmap = BitmapFactory.decodeStream(getAssets().open(file));
      bitmap = Bitmap.createScaledBitmap(bitmap, 299, 299, true);
    } catch (IOException e) {
      Log.e("Classification", "Error reading assets", e);
      finish();
    }

    // showing image on UI
    ImageView imageView = findViewById(R.id.image);
    imageView.setImageBitmap(bitmap);
  }

  @Override
  public void run() {
    Bitmap bitmap = null;
    Module module = null;
    try {
      bitmap = BitmapFactory.decodeStream(getAssets().open("corgi2.jpg"));
      bitmap = Bitmap.createScaledBitmap(bitmap, 299, 299, true);
      module = Module.load(MainActivity.assetFilePath(this, "ic4_xnnpack_fp32.pte"));
    } catch (IOException e) {
      Log.e("PytorchHelloWorld", "Error reading assets", e);
      finish();
    }

    // showing image on UI
    ImageView imageView = findViewById(R.id.image);
    imageView.setImageBitmap(bitmap);

    // preparing input tensor
    final Tensor inputTensor =
        TensorImageUtils.bitmapToFloat32Tensor(
            bitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB);

    // running the model
    final Tensor outputTensor = module.forward(EValue.from(inputTensor))[0].toTensor();

    // getting tensor content as java array of floats
    final float[] scores = outputTensor.getDataAsFloatArray();

    // searching for the index with maximum score
    float maxScore = -Float.MAX_VALUE;
    int maxScoreIdx = -1;
    for (int i = 0; i < scores.length; i++) {
      if (scores[i] > maxScore) {
        maxScore = scores[i];
        maxScoreIdx = i;
      }
    }

    String className = ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx];

    // showing className on UI
    TextView textView = findViewById(R.id.text);
    textView.setText(className);
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_classification);

    final Button classificationDemoButton = findViewById(R.id.segmentationDemoButton);
    classificationDemoButton.setOnClickListener(
        new View.OnClickListener() {
          public void onClick(View v) {
            openSegmentationActivity();
          }
        });

    final Button forwardButton = findViewById(R.id.forward);
    forwardButton.setOnClickListener(
        new View.OnClickListener() {
          public void onClick(View v) {
            TextView textView = findViewById(R.id.text);
            textView.setText("Running");
            ClassificationActivity.this.run();
          }
        });

    populateBitmap("corgi2.jpg");
  }
}
