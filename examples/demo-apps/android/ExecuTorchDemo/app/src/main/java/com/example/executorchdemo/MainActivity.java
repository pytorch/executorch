/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchdemo;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.SystemClock;
import android.system.ErrnoException;
import android.system.Os;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import com.example.executorchdemo.executor.EValue;
import com.example.executorchdemo.executor.Module;
import com.example.executorchdemo.executor.Tensor;
import com.example.executorchdemo.executor.TensorImageUtils;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Objects;

public class MainActivity extends Activity implements Runnable {
  private ImageView mImageView;
  private Button mButtonSegment;
  private ProgressBar mProgressBar;
  private Bitmap mBitmap = null;
  private Module mModule = null;
  private String mImagename = "corgi.jpeg";

  // see http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/segexamples/index.html for the list of
  // classes with indexes
  private static final int CLASSNUM = 21;
  private static final int DOG = 12;
  private static final int PERSON = 15;
  private static final int SHEEP = 17;

  private void openClassificationActivity() {
    Intent intent = new Intent(this, ClassificationActivity.class);
    startActivity(intent);
  }

  public static String assetFilePath(Context context, String assetName) throws IOException {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    }
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    try {
      Os.setenv("ADSP_LIBRARY_PATH", getApplicationInfo().nativeLibraryDir, true);
    } catch (ErrnoException e) {
      Log.e("ExecuTorchDemo", "Cannot set ADSP_LIBRARY_PATH", e);
      finish();
    }

    try {
      mBitmap = BitmapFactory.decodeStream(getAssets().open(mImagename), null, null);
      mBitmap = Bitmap.createScaledBitmap(mBitmap, 224, 224, true);
    } catch (IOException e) {
      Log.e("ImageSegmentation", "Error reading assets", e);
      finish();
    }

    mImageView = findViewById(R.id.imageView);
    mImageView.setImageBitmap(mBitmap);

    final Button buttonRestart = findViewById(R.id.restartButton);
    buttonRestart.setOnClickListener(
        new View.OnClickListener() {
          public void onClick(View v) {
            if (Objects.equals(mImagename, "corgi.jpeg")) mImagename = "dog.jpg";
            else mImagename = "deeplab.jpg";
            try {
              mBitmap = BitmapFactory.decodeStream(getAssets().open(mImagename));
              mBitmap = Bitmap.createScaledBitmap(mBitmap, 224, 224, true);
              mImageView.setImageBitmap(mBitmap);
            } catch (IOException e) {
              Log.e("ImageSegmentation", "Error reading assets", e);
              finish();
            }
          }
        });

    final Button classificationDemoButton = findViewById(R.id.classificationDemoButton);
    classificationDemoButton.setOnClickListener(
        new View.OnClickListener() {
          public void onClick(View v) {
            openClassificationActivity();
          }
        });

    mButtonSegment = findViewById(R.id.segmentButton);
    mProgressBar = (ProgressBar) findViewById(R.id.progressBar);
    mButtonSegment.setOnClickListener(
        new View.OnClickListener() {
          public void onClick(View v) {
            mButtonSegment.setEnabled(false);
            mProgressBar.setVisibility(ProgressBar.VISIBLE);
            mButtonSegment.setText(getString(R.string.run_model));

            Thread thread = new Thread(MainActivity.this);
            thread.start();
          }
        });

    try {
      mModule =
          Module.load(MainActivity.assetFilePath(getApplicationContext(), "dl3_xnnpack_fp32.pte"));
    } catch (IOException e) {
      Log.e("ImageSegmentation", "Error reading assets", e);
      finish();
    }
  }

  @Override
  public void run() {
    final Tensor inputTensor =
        TensorImageUtils.bitmapToFloat32Tensor(
            mBitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB);
    final float[] inputs = inputTensor.getDataAsFloatArray();

    final long startTime = SystemClock.elapsedRealtime();
    Tensor outputTensor = mModule.forward(EValue.from(inputTensor)).toTensor();
    final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
    Log.d("ImageSegmentation", "inference time (ms): " + inferenceTime);

    final float[] scores = outputTensor.getDataAsFloatArray();
    int width = mBitmap.getWidth();
    int height = mBitmap.getHeight();

    int[] intValues = new int[width * height];
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        int maxi = 0, maxj = 0, maxk = 0;
        double maxnum = -Double.MAX_VALUE;
        for (int i = 0; i < CLASSNUM; i++) {
          float score = scores[i * (width * height) + j * width + k];
          if (score > maxnum) {
            maxnum = score;
            maxi = i;
            maxj = j;
            maxk = k;
          }
        }
        if (maxi == PERSON) intValues[maxj * width + maxk] = 0xFFFF0000; // R
        else if (maxi == DOG) intValues[maxj * width + maxk] = 0xFF00FF00; // G
        else if (maxi == SHEEP) intValues[maxj * width + maxk] = 0xFF0000FF; // B
        else intValues[maxj * width + maxk] = 0xFF000000;
      }
    }

    Bitmap bmpSegmentation = Bitmap.createScaledBitmap(mBitmap, width, height, true);
    Bitmap outputBitmap = bmpSegmentation.copy(bmpSegmentation.getConfig(), true);
    outputBitmap.setPixels(
        intValues,
        0,
        outputBitmap.getWidth(),
        0,
        0,
        outputBitmap.getWidth(),
        outputBitmap.getHeight());
    final Bitmap transferredBitmap =
        Bitmap.createScaledBitmap(outputBitmap, mBitmap.getWidth(), mBitmap.getHeight(), true);

    runOnUiThread(
        new Runnable() {
          @Override
          public void run() {
            mImageView.setImageBitmap(transferredBitmap);
            mButtonSegment.setEnabled(true);
            mButtonSegment.setText("segment");
            mProgressBar.setVisibility(ProgressBar.INVISIBLE);
          }
        });
  }
}
