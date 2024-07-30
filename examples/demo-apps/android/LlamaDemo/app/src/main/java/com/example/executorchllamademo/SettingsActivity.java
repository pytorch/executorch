/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo;

import android.app.AlertDialog;
import android.content.DialogInterface;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import com.google.gson.Gson;
import java.io.File;

public class SettingsActivity extends AppCompatActivity {

  private String mModelFilePath = "";
  private String mTokenizerFilePath = "";
  private TextView mModelTextView;
  private TextView mTokenizerTextView;
  private ImageButton mModelImageButton;
  private ImageButton mTokenizerImageButton;
  private EditText mSystemPromptEditText;
  private EditText mUserPromptEditText;
  private Button mLoadModelButton;
  private double mSetTemperature;
  private String mSystemPrompt;
  private String mUserPrompt;

  public SettingsFields mSettingsFields;

  private DemoSharedPreferences mDemoSharedPreferences;
  public static double TEMPERATURE_MIN_VALUE = 0.1;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_settings);
    ViewCompat.setOnApplyWindowInsetsListener(
        requireViewById(R.id.main),
        (v, insets) -> {
          Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
          v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
          return insets;
        });
    mDemoSharedPreferences = new DemoSharedPreferences(getBaseContext());
    mSettingsFields = new SettingsFields();
    setupSettings();
  }

  private void setupSettings() {
    mModelTextView = requireViewById(R.id.modelTextView);
    mTokenizerTextView = requireViewById(R.id.tokenizerTextView);
    mModelImageButton = requireViewById(R.id.modelImageButton);
    mTokenizerImageButton = requireViewById(R.id.tokenizerImageButton);
    mSystemPromptEditText = requireViewById(R.id.systemPromptText);
    mUserPromptEditText = requireViewById(R.id.userPromptText);
    loadSettings();

    // TODO: The two setOnClickListeners will be removed after file path issue is resolved
    mModelImageButton.setOnClickListener(
        view -> {
          setupModelSelectorDialog();
        });
    mTokenizerImageButton.setOnClickListener(
        view -> {
          setupTokenizerSelectorDialog();
        });
    mModelFilePath = mSettingsFields.getModelFilePath();
    if (!mModelFilePath.isEmpty()) {
      mModelTextView.setText(getFilenameFromPath(mModelFilePath));
    }
    mTokenizerFilePath = mSettingsFields.getTokenizerFilePath();
    if (!mTokenizerFilePath.isEmpty()) {
      mTokenizerTextView.setText(getFilenameFromPath(mTokenizerFilePath));
    }

    setupParameterSettings();
    setupPromptSettings();
    setupClearChatHistoryButton();
    setupLoadModelButton();
  }

  private void setupLoadModelButton() {
    mLoadModelButton = requireViewById(R.id.loadModelButton);
    mLoadModelButton.setEnabled(true);
    mLoadModelButton.setOnClickListener(
        view -> {
          new AlertDialog.Builder(this)
              .setTitle("Load Model")
              .setMessage("Do you really want to load the new model?")
              .setIcon(android.R.drawable.ic_dialog_alert)
              .setPositiveButton(
                  android.R.string.yes,
                  new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int whichButton) {
                      mSettingsFields.saveLoadModelAction(true);
                      mLoadModelButton.setEnabled(false);
                    }
                  })
              .setNegativeButton(android.R.string.no, null)
              .show();
        });
  }

  private void setupClearChatHistoryButton() {
    Button clearChatButton = requireViewById(R.id.clearChatButton);
    clearChatButton.setOnClickListener(
        view -> {
          new AlertDialog.Builder(this)
              .setTitle("Delete Chat History")
              .setMessage("Do you really want to delete chat history?")
              .setIcon(android.R.drawable.ic_dialog_alert)
              .setPositiveButton(
                  android.R.string.yes,
                  new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int whichButton) {
                      mSettingsFields.saveIsClearChatHistory(true);
                    }
                  })
              .setNegativeButton(android.R.string.no, null)
              .show();
        });
  }

  private void setupParameterSettings() {
    setupTemperatureSettings();
  }

  private void setupTemperatureSettings() {
    mSetTemperature = mSettingsFields.getTemperature();
    EditText temperatureEditText = requireViewById(R.id.temperatureEditText);
    temperatureEditText.setText(String.valueOf(mSetTemperature));
    temperatureEditText.addTextChangedListener(
        new TextWatcher() {
          @Override
          public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

          @Override
          public void onTextChanged(CharSequence s, int start, int before, int count) {}

          @Override
          public void afterTextChanged(Editable s) {
            mSetTemperature = Double.parseDouble(s.toString());
            // This is needed because temperature is changed together with model loading
            // Once temperature is no longer in LlamaModule constructor, we can remove this
            mSettingsFields.saveLoadModelAction(true);
            saveSettings();
          }
        });
  }

  private void setupPromptSettings() {
    setupSystemPromptSettings();
    setupUserPromptSettings();
  }

  private void setupSystemPromptSettings() {
    mSystemPrompt = mSettingsFields.getSystemPrompt();
    mSystemPromptEditText.setText(mSystemPrompt);
    mSystemPromptEditText.addTextChangedListener(
        new TextWatcher() {
          @Override
          public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

          @Override
          public void onTextChanged(CharSequence s, int start, int before, int count) {}

          @Override
          public void afterTextChanged(Editable s) {
            mSystemPrompt = s.toString();
          }
        });

    ImageButton resetSystemPrompt = requireViewById(R.id.resetSystemPrompt);
    resetSystemPrompt.setOnClickListener(
        view -> {
          new AlertDialog.Builder(this)
              .setTitle("Reset System Prompt")
              .setMessage("Do you really want to reset system prompt?")
              .setIcon(android.R.drawable.ic_dialog_alert)
              .setPositiveButton(
                  android.R.string.yes,
                  new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int whichButton) {
                      // Clear the messageAdapter and sharedPreference
                      mSystemPromptEditText.setText(mSettingsFields.getSystemPromptTemplate());
                    }
                  })
              .setNegativeButton(android.R.string.no, null)
              .show();
        });
  }

  private void setupUserPromptSettings() {
    mUserPrompt = mSettingsFields.getUserPrompt();
    mUserPromptEditText.setText(mUserPrompt);
    mUserPromptEditText.addTextChangedListener(
        new TextWatcher() {
          @Override
          public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

          @Override
          public void onTextChanged(CharSequence s, int start, int before, int count) {}

          @Override
          public void afterTextChanged(Editable s) {
            mUserPrompt = s.toString();
          }
        });

    ImageButton resetUserPrompt = requireViewById(R.id.resetUserPrompt);
    resetUserPrompt.setOnClickListener(
        view -> {
          new AlertDialog.Builder(this)
              .setTitle("Reset Prompt Template")
              .setMessage("Do you really want to reset the prompt template?")
              .setIcon(android.R.drawable.ic_dialog_alert)
              .setPositiveButton(
                  android.R.string.yes,
                  new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int whichButton) {
                      // Clear the messageAdapter and sharedPreference
                      mUserPromptEditText.setText(mSettingsFields.getUserPromptTemplate());
                    }
                  })
              .setNegativeButton(android.R.string.no, null)
              .show();
        });
  }

  private void setupModelSelectorDialog() {
    String[] pteFiles = listLocalFile("/data/local/tmp/llama/", ".pte");
    AlertDialog.Builder modelPathBuilder = new AlertDialog.Builder(this);
    modelPathBuilder.setTitle("Select model path");

    modelPathBuilder.setSingleChoiceItems(
        pteFiles,
        -1,
        (dialog, item) -> {
          mModelFilePath = pteFiles[item];
          mModelTextView.setText(getFilenameFromPath(mModelFilePath));
          mLoadModelButton.setEnabled(true);
          dialog.dismiss();
        });

    modelPathBuilder.create().show();
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

  private void setupTokenizerSelectorDialog() {
    String[] binFiles = listLocalFile("/data/local/tmp/llama/", ".bin");
    String[] tokenizerFiles = new String[binFiles.length];
    System.arraycopy(binFiles, 0, tokenizerFiles, 0, binFiles.length);
    AlertDialog.Builder tokenizerPathBuilder = new AlertDialog.Builder(this);
    tokenizerPathBuilder.setTitle("Select tokenizer path");
    tokenizerPathBuilder.setSingleChoiceItems(
        tokenizerFiles,
        -1,
        (dialog, item) -> {
          mTokenizerFilePath = tokenizerFiles[item];
          mTokenizerTextView.setText(getFilenameFromPath(mTokenizerFilePath));
          mLoadModelButton.setEnabled(true);
          dialog.dismiss();
        });

    tokenizerPathBuilder.create().show();
  }

  private String getFilenameFromPath(String uriFilePath) {
    String[] segments = uriFilePath.split("/");
    if (segments.length > 0) {
      return segments[segments.length - 1]; // get last element (aka filename)
    }
    return "";
  }

  private void loadSettings() {
    Gson gson = new Gson();
    String settingsFieldsJSON = mDemoSharedPreferences.getSettings();
    if (!settingsFieldsJSON.isEmpty()) {
      mSettingsFields = gson.fromJson(settingsFieldsJSON, SettingsFields.class);
    }
  }

  private void saveSettings() {
    mSettingsFields.saveModelPath(mModelFilePath);
    mSettingsFields.saveTokenizerPath(mTokenizerFilePath);
    mSettingsFields.saveParameters(mSetTemperature);
    mSettingsFields.savePrompts(mSystemPrompt, mUserPrompt);
    mDemoSharedPreferences.addSettings(mSettingsFields);
  }

  @Override
  public void onBackPressed() {
    super.onBackPressed();
    saveSettings();
  }
}
