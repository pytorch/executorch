/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import org.pytorch.executorch.LlamaCallback;
import org.pytorch.executorch.LlamaModule;

public class MainActivity extends Activity implements Runnable, LlamaCallback {
  private EditText mEditTextMessage;
  private TextView mTextViewChat;
  private Button mSendButton;
  private Button mStopButton;
  private LlamaModule mModule = null;
  private String mResult = null;

  private static String assetFilePath(Context context, String assetName) throws IOException {
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
  public void onResult(String result) {
    System.out.println("onResult: " + result);
    mResult = result;
    run();
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    mEditTextMessage = findViewById(R.id.editTextMessage);
    mTextViewChat = findViewById(R.id.textViewChat);
    mSendButton = findViewById(R.id.sendButton);
    mStopButton = findViewById(R.id.stopButton);

    mSendButton.setOnClickListener(
        view -> {
          String prompt = mEditTextMessage.getText().toString();
          mTextViewChat.append(prompt);
          mEditTextMessage.setText("");
          Runnable runnable =
              new Runnable() {
                @Override
                public void run() {
                  mModule.generate(prompt, MainActivity.this);
                }
              };
          new Thread(runnable).start();
        });

    mStopButton.setOnClickListener(
        view -> {
          mModule.stop();
        });

    try {
      String model = MainActivity.assetFilePath(getApplicationContext(), "model.pte");
      String tokenizer = MainActivity.assetFilePath(getApplicationContext(), "tokenizer.bin");
      mModule = new LlamaModule(model, tokenizer, 0.8f);
    } catch (IOException e) {
      finish();
    }
  }

  @Override
  public void run() {
    runOnUiThread(
        new Runnable() {
          @Override
          public void run() {
            mTextViewChat.append(mResult);
          }
        });
  }
}
