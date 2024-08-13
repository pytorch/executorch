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
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Objects;
import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;

public class MainActivity extends Activity implements Runnable {
  private ImageView mImageView;
  private Button mButtonXnnpack;
  private Button mButtonHtp;
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

  private void populateImage() {
    try {
      mBitmap = BitmapFactory.decodeStream(getAssets().open(mImagename));
      mBitmap = Bitmap.createScaledBitmap(mBitmap, 224, 224, true);
      mImageView.setImageBitmap(mBitmap);
    } catch (IOException e) {
      Log.e("ImageSegmentation", "Error reading assets", e);
      finish();
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

    try {
      mModule =
          Module.load(MainActivity.assetFilePath(getApplicationContext(), "dl3_xnnpack_fp32.pte"));

    } catch (IOException e) {
      Log.e("ImageSegmentation", "Error reading assets", e);
      finish();
    }

    mImageView = findViewById(R.id.imageView);
    mImageView.setImageBitmap(mBitmap);

    final Button buttonNext = findViewById(R.id.nextButton);
    buttonNext.setOnClickListener(
        new View.OnClickListener() {
          public void onClick(View v) {
            if (Objects.equals(mImagename, "corgi.jpeg")) {
              mImagename = "dog.jpg";
            } else if (Objects.equals(mImagename, "dog.jpg")) {
              mImagename = "deeplab.jpg";
            } else {
              mImagename = "corgi.jpeg";
            }
            populateImage();
          }
        });

    mButtonXnnpack = findViewById(R.id.xnnpackButton);
    mButtonHtp = findViewById(R.id.htpButton);
    mProgressBar = (ProgressBar) findViewById(R.id.progressBar);
    mButtonXnnpack.setOnClickListener(
        new View.OnClickListener() {
          public void onClick(View v) {
            try {
              mModule.destroy();
              mModule =
                  Module.load(
                      MainActivity.assetFilePath(getApplicationContext(), "dl3_xnnpack_fp32.pte"));
            } catch (IOException e) {
              Log.e("ImageSegmentation", "Error reading assets", e);
              finish();
            }

            mButtonXnnpack.setEnabled(false);
            mProgressBar.setVisibility(ProgressBar.VISIBLE);
            mButtonXnnpack.setText(getString(R.string.run_model));

            Thread thread = new Thread(MainActivity.this);
            thread.start();
          }
        });

    mButtonHtp.setOnClickListener(
        new View.OnClickListener() {
          public void onClick(View v) {
            try {
              mModule.destroy();
              mModule =
                  Module.load(MainActivity.assetFilePath(getApplicationContext(), "dlv3_qnn.pte"));
            } catch (IOException e) {
              Log.e("ImageSegmentation", "Error reading assets", e);
              finish();
            }
            mButtonHtp.setEnabled(false);
            mProgressBar.setVisibility(ProgressBar.VISIBLE);
            mButtonHtp.setText(getString(R.string.run_model));

            Thread thread = new Thread(MainActivity.this);
            thread.start();
          }
        });

    final Button resetImage = findViewById(R.id.resetImage);
    resetImage.setOnClickListener(
        new View.OnClickListener() {
          public void onClick(View v) {
            populateImage();
          }
        });
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
    Tensor outputTensor = mModule.forward(EValue.from(inputTensor))[0].toTensor();
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
            mButtonXnnpack.setEnabled(true);
            mButtonXnnpack.setText(R.string.run_xnnpack);
            mButtonHtp.setEnabled(true);
            mButtonHtp.setText(R.string.run_htp);
            mProgressBar.setVisibility(ProgressBar.INVISIBLE);
          }
        });
  }
}
