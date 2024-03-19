/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo;

import android.app.Activity;
import android.app.ActivityManager;
import android.app.AlertDialog;
import android.content.Context;
import android.os.Bundle;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ListView;
import java.io.File;
import org.pytorch.executorch.LlamaCallback;
import org.pytorch.executorch.LlamaModule;

public class MainActivity extends Activity implements Runnable, LlamaCallback {
  private EditText mEditTextMessage;
  private Button mSendButton;
  private ImageButton mModelButton;
  private ListView mMessagesView;
  private MessageAdapter mMessageAdapter;
  private LlamaModule mModule = null;
  private Message mResultMessage = null;

  private int mNumTokens = 0;
  private long mRunStartTime = 0;
  private String mModelFilePath = "";
  private String mTokenizerFilePath = "";

  @Override
  public void onResult(String result) {
    System.out.println("onResult: " + result);
    mResultMessage.appendText(result);
    mNumTokens++;
    run();
  }

  private static String[] listLocalFile(String path, String suffix) {
    File directory = new File(path);
    if (directory.exists() && directory.isDirectory()) {
      File[] files = directory.listFiles((dir, name) -> name.toLowerCase().endsWith(suffix));
      String[] result = new String[files.length];
      for (int i = 0; i < files.length; i++) {
        if (files[i].isFile() && files[i].getName().endsWith(suffix)) {
          result[i] = files[i].getAbsolutePath();
        }
      }
      return result;
    }
    return null;
  }

  private void setLocalModel(String modelPath, String tokenizerPath) {
    Message modelLoadingMessage = new Message("Loading model...", false);
    runOnUiThread(
        () -> {
          mSendButton.setEnabled(false);
          mMessageAdapter.add(modelLoadingMessage);
          mMessageAdapter.notifyDataSetChanged();
        });
    long runStartTime = System.currentTimeMillis();
    mModule = new LlamaModule(modelPath, tokenizerPath, 0.8f);
    int loadResult = mModule.load();
    if (loadResult != 0) {
      AlertDialog.Builder builder = new AlertDialog.Builder(this);
      builder.setTitle("Load failed: " + loadResult);
      AlertDialog alert = builder.create();
      runOnUiThread(
          () -> {
            alert.show();
          });
    }

    long runDuration = System.currentTimeMillis() - runStartTime;
    String modelInfo =
        "Model path: "
            + modelPath
            + "\nTokenizer path: "
            + tokenizerPath
            + "\nModel loaded time: "
            + runDuration
            + " ms";
    Message modelLoadedMessage = new Message(modelInfo, false);
    runOnUiThread(
        () -> {
          mSendButton.setEnabled(true);
          mMessageAdapter.remove(modelLoadingMessage);
          mMessageAdapter.add(modelLoadedMessage);
          mMessageAdapter.notifyDataSetChanged();
        });
  }

  private String memoryInfo() {
    final ActivityManager am = (ActivityManager) getSystemService(Context.ACTIVITY_SERVICE);
    ActivityManager.MemoryInfo memInfo = new ActivityManager.MemoryInfo();
    am.getMemoryInfo(memInfo);
    return "Total RAM: "
        + Math.floorDiv(memInfo.totalMem, 1000000)
        + " MB. Available RAM: "
        + Math.floorDiv(memInfo.availMem, 1000000)
        + " MB.";
  }

  private void modelDialog() {
    String[] pteFiles = listLocalFile("/data/local/tmp/llama/", ".pte");
    String[] binFiles = listLocalFile("/data/local/tmp/llama/", ".bin");
    AlertDialog.Builder modelPathBuilder = new AlertDialog.Builder(this);
    modelPathBuilder.setTitle("Select model path");
    AlertDialog.Builder tokenizerPathBuilder = new AlertDialog.Builder(this);
    tokenizerPathBuilder.setTitle("Select tokenizer path");
    modelPathBuilder.setSingleChoiceItems(
        binFiles,
        -1,
        (dialog, item) -> {
          mTokenizerFilePath = binFiles[item];
          mEditTextMessage.setText("");
          dialog.dismiss();
          tokenizerPathBuilder.create().show();
        });

    tokenizerPathBuilder.setSingleChoiceItems(
        pteFiles,
        -1,
        (dialog, item) -> {
          mModelFilePath = pteFiles[item];
          Runnable runnable =
              new Runnable() {
                @Override
                public void run() {
                  setLocalModel(mModelFilePath, mTokenizerFilePath);
                }
              };
          new Thread(runnable).start();
          dialog.dismiss();
        });

    modelPathBuilder.create().show();
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    mEditTextMessage = findViewById(R.id.editTextMessage);
    mSendButton = findViewById(R.id.sendButton);
    mSendButton.setEnabled(false);
    mModelButton = findViewById(R.id.modelButton);
    mMessagesView = findViewById(R.id.messages_view);
    mMessageAdapter = new MessageAdapter(this, R.layout.sent_message);
    mMessagesView.setAdapter(mMessageAdapter);
    mModelButton.setOnClickListener(
        view -> {
          mModule.stop();
          mMessageAdapter.clear();
          mMessageAdapter.notifyDataSetChanged();
          modelDialog();
        });

    onModelRunStopped();
    modelDialog();
  }

  private void onModelRunStarted() {
    mSendButton.setText("Stop");
    mSendButton.setOnClickListener(
        view -> {
          mModule.stop();
        });

    mRunStartTime = System.currentTimeMillis();
  }

  private void onModelRunStopped() {
    setTitle(memoryInfo());
    long runDuration = System.currentTimeMillis() - mRunStartTime;
    if (mResultMessage != null) {
      mResultMessage.setTokensPerSecond(1.0f * mNumTokens / (runDuration / 1000.0f));
    }
    mSendButton.setText("Generate");
    mSendButton.setOnClickListener(
        view -> {
          String prompt = mEditTextMessage.getText().toString();
          mMessageAdapter.add(new Message(prompt, true));
          mMessageAdapter.notifyDataSetChanged();
          mEditTextMessage.setText("");
          mResultMessage = new Message("", false);
          mMessageAdapter.add(mResultMessage);
          Runnable runnable =
              new Runnable() {
                @Override
                public void run() {
                  runOnUiThread(
                      new Runnable() {
                        @Override
                        public void run() {
                          onModelRunStarted();
                        }
                      });

                  mModule.generate(prompt, MainActivity.this);

                  runOnUiThread(
                      new Runnable() {
                        @Override
                        public void run() {
                          onModelRunStopped();
                        }
                      });
                }
              };
          new Thread(runnable).start();
        });
    mNumTokens = 0;
    mRunStartTime = 0;
    mMessageAdapter.notifyDataSetChanged();
  }

  @Override
  public void run() {
    runOnUiThread(
        new Runnable() {
          @Override
          public void run() {
            mMessageAdapter.notifyDataSetChanged();
            setTitle(memoryInfo());
          }
        });
  }
}
