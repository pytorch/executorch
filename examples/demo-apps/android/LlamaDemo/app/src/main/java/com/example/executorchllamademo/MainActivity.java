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
import android.system.ErrnoException;
import android.system.Os;
import android.util.Log;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ListView;
import android.widget.Toast;

import java.io.File;
import org.pytorch.executorch.LlamaCallback;
import org.pytorch.executorch.LlamaModule;

public class MainActivity extends Activity implements ModelRunnerCallback {
  private EditText mEditTextMessage;
  private Button mSendButton;
  private ImageButton mModelButton;
  private ListView mMessagesView;
  private MessageAdapter mMessageAdapter;
    private Message mResultMessage = null;
  private ModelRunner mModelRunner;
    private String mModelFilePath = "";
    private String mTokenizerFilePath = "";

    long mRunStartTime = 0;
    Message mModelLoadingMessage = null;

    @Override
    public void onGenerationStopped() {
        runOnUiThread(this::changeSendButtonToStart);
    }

    @Override
    public void onTokenGenerated(String token) {
        mResultMessage.appendText(token);
        runOnUiThread(()->{
            mMessageAdapter.notifyDataSetChanged();
        });
        Toast.makeText(this, "generated", Toast.LENGTH_LONG).show();
    }

    @Override
    public void onModelLoaded(int loadResult) {
        if (loadResult != 0) {
            AlertDialog.Builder builder = new AlertDialog.Builder(this);
            builder.setTitle("Load failed: " + loadResult);
            runOnUiThread(
                    () -> {
                        AlertDialog alert = builder.create();
                        alert.show();
                    });
        }

        long loadDuration = System.currentTimeMillis() - mRunStartTime;
        String modelInfo =
                "Model path: "
                        + mModelFilePath
                        + "\nTokenizer path: "
                        + mTokenizerFilePath
                        + "\nModel loaded time: "
                        + loadDuration
                        + " ms";
        Message modelLoadedMessage = new Message(modelInfo, false);
        runOnUiThread(
                () -> {
                    mSendButton.setEnabled(true);
                    mMessageAdapter.remove(mModelLoadingMessage);
                    mMessageAdapter.add(modelLoadedMessage);
                    mMessageAdapter.notifyDataSetChanged();
                });
    }


  @Override
  public void onStats(String tps) {
    runOnUiThread(
        () -> {
          if (mResultMessage != null) {
            mResultMessage.setTokensPerSecond(0f); // TODO: Use tps
            mMessageAdapter.notifyDataSetChanged();
          }
        });
  }

  static String[] listLocalFile(String path, String suffix) {
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
    return new String[0];
  }

  private void setLocalModel(String modelPath, String tokenizerPath) {
    mModelLoadingMessage = new Message("Loading model...", false);
    runOnUiThread(
        () -> {
          mSendButton.setEnabled(false);
          mMessageAdapter.add(mModelLoadingMessage);
          mMessageAdapter.notifyDataSetChanged();
        });
    mRunStartTime = System.currentTimeMillis();
    mModelRunner = new ModelRunner(modelPath, tokenizerPath, 0.8f, this);
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
    String[] modelFiles = listLocalFile("/data/local/tmp/llama/", ".model");
    String[] tokenizerFiles = new String[binFiles.length + modelFiles.length];
    System.arraycopy(binFiles, 0, tokenizerFiles, 0, binFiles.length);
    System.arraycopy(modelFiles, 0, tokenizerFiles, binFiles.length, modelFiles.length);
    AlertDialog.Builder modelPathBuilder = new AlertDialog.Builder(this);
    modelPathBuilder.setTitle("Select model path");
    AlertDialog.Builder tokenizerPathBuilder = new AlertDialog.Builder(this);
    tokenizerPathBuilder.setTitle("Select tokenizer path");
    modelPathBuilder.setSingleChoiceItems(
        pteFiles,
        -1,
        (dialog, item) -> {
          mModelFilePath = pteFiles[item];
          mEditTextMessage.setText("");
          dialog.dismiss();
          tokenizerPathBuilder.create().show();
        });

    tokenizerPathBuilder.setSingleChoiceItems(
        tokenizerFiles,
        -1,
        (dialog, item) -> {
          mTokenizerFilePath = tokenizerFiles[item];
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

    try {
      Os.setenv("ADSP_LIBRARY_PATH", getApplicationInfo().nativeLibraryDir, true);
    } catch (ErrnoException e) {
      finish();
    }

    mEditTextMessage = findViewById(R.id.editTextMessage);
    mSendButton = findViewById(R.id.sendButton);
    mSendButton.setEnabled(false);
    mModelButton = findViewById(R.id.modelButton);
    mMessagesView = findViewById(R.id.messages_view);
    mMessageAdapter = new MessageAdapter(this, R.layout.sent_message);
    mMessagesView.setAdapter(mMessageAdapter);
    mModelButton.setOnClickListener(
        view -> {
          mModelRunner.stop();
          mMessageAdapter.clear();
          mMessageAdapter.notifyDataSetChanged();
          modelDialog();
        });

      changeSendButtonToStart();
    modelDialog();
  }

  private void changeSendButtonToStop() {
    mSendButton.setText("Stop");
    mSendButton.setOnClickListener(
        view -> {
            onSendButtonStopClicked();
        });
  }

  private void changeSendButtonToStart() {
    setTitle(memoryInfo());
      mSendButton.setText("Generate");
    mSendButton.setOnClickListener(
        view -> {
            onSendButtonGenerateClicked();
        });
  }

  private void onSendButtonGenerateClicked() {
      String prompt = mEditTextMessage.getText().toString();
      mEditTextMessage.setText("");
      mMessageAdapter.add(new Message(prompt, true));
      mMessageAdapter.notifyDataSetChanged();
      mResultMessage = new Message("", false);
      mMessageAdapter.add(mResultMessage);
      mModelRunner.generate(prompt);
      changeSendButtonToStop();
  }

  private void onSendButtonStopClicked() {
        mModelRunner.stop();
  }
}
