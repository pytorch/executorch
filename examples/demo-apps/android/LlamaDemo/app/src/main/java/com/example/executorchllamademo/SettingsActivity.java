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
import android.os.Build;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import com.google.gson.Gson;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SettingsActivity extends AppCompatActivity {

  private String mModelFilePath = "";
  private String mTokenizerFilePath = "";
  private TextView mBackendTextView;
  private TextView mModelTextView;
  private TextView mTokenizerTextView;
  private TextView mModelTypeTextView;
  private EditText mSystemPromptEditText;
  private EditText mUserPromptEditText;
  private Button mLoadModelButton;
  private double mSetTemperature;
  private String mSystemPrompt;
  private String mUserPrompt;
  private BackendType mBackendType;
  private ModelType mModelType;
  public SettingsFields mSettingsFields;

  private DemoSharedPreferences mDemoSharedPreferences;
  public static double TEMPERATURE_MIN_VALUE = 0.0;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_settings);
    if (Build.VERSION.SDK_INT >= 21) {
      getWindow().setStatusBarColor(ContextCompat.getColor(this, R.color.status_bar));
      getWindow().setNavigationBarColor(ContextCompat.getColor(this, R.color.nav_bar));
    }
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
    mBackendTextView = requireViewById(R.id.backendTextView);
    mModelTextView = requireViewById(R.id.modelTextView);
    mTokenizerTextView = requireViewById(R.id.tokenizerTextView);
    mModelTypeTextView = requireViewById(R.id.modelTypeTextView);
    ImageButton backendImageButton = requireViewById(R.id.backendImageButton);
    ImageButton modelImageButton = requireViewById(R.id.modelImageButton);
    ImageButton tokenizerImageButton = requireViewById(R.id.tokenizerImageButton);
    ImageButton modelTypeImageButton = requireViewById(R.id.modelTypeImageButton);
    mSystemPromptEditText = requireViewById(R.id.systemPromptText);
    mUserPromptEditText = requireViewById(R.id.userPromptText);
    loadSettings();

    // TODO: The two setOnClickListeners will be removed after file path issue is resolved
    backendImageButton.setOnClickListener(
        view -> {
          setupBackendSelectorDialog();
        });
    modelImageButton.setOnClickListener(
        view -> {
          setupModelSelectorDialog();
        });
    tokenizerImageButton.setOnClickListener(
        view -> {
          setupTokenizerSelectorDialog();
        });
    modelTypeImageButton.setOnClickListener(
        view -> {
          setupModelTypeSelectorDialog();
        });
    mModelFilePath = mSettingsFields.getModelFilePath();
    if (!mModelFilePath.isEmpty()) {
      mModelTextView.setText(getFilenameFromPath(mModelFilePath));
    }
    mTokenizerFilePath = mSettingsFields.getTokenizerFilePath();
    if (!mTokenizerFilePath.isEmpty()) {
      mTokenizerTextView.setText(getFilenameFromPath(mTokenizerFilePath));
    }
    mModelType = mSettingsFields.getModelType();
    ETLogging.getInstance().log("mModelType from settings " + mModelType);
    if (mModelType != null) {
      mModelTypeTextView.setText(mModelType.toString());
    }
    mBackendType = mSettingsFields.getBackendType();
    ETLogging.getInstance().log("mBackendType from settings " + mBackendType);
    if (mBackendType != null) {
      mBackendTextView.setText(mBackendType.toString());
      setBackendSettingMode();
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
                      onBackPressed();
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
            // Once temperature is no longer in LlmModule constructor, we can remove this
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
                      mSystemPromptEditText.setText(PromptFormat.DEFAULT_SYSTEM_PROMPT);
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
            if (isValidUserPrompt(s.toString())) {
              mUserPrompt = s.toString();
            } else {
              showInvalidPromptDialog();
            }
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
                      mUserPromptEditText.setText(
                          PromptFormat.getUserPromptTemplate(mModelType, false));
                    }
                  })
              .setNegativeButton(android.R.string.no, null)
              .show();
        });
  }

  private boolean isValidUserPrompt(String userPrompt) {
    return userPrompt.contains(PromptFormat.USER_PLACEHOLDER);
  }

  private void showInvalidPromptDialog() {
    new AlertDialog.Builder(this)
        .setTitle("Invalid Prompt Format")
        .setMessage(
            "Prompt format must contain "
                + PromptFormat.USER_PLACEHOLDER
                + ". Do you want to reset prompt format?")
        .setIcon(android.R.drawable.ic_dialog_alert)
        .setPositiveButton(
            android.R.string.yes,
            (dialog, whichButton) -> {
              mUserPromptEditText.setText(PromptFormat.getUserPromptTemplate(mModelType, false));
            })
        .setNegativeButton(android.R.string.no, null)
        .show();
  }

  private void setupBackendSelectorDialog() {
    // Convert enum to list
    List<String> backendTypesList = new ArrayList<>();
    for (BackendType backendType : BackendType.values()) {
      backendTypesList.add(backendType.toString());
    }
    // Alert dialog builder takes in arr of string instead of list
    String[] backendTypes = backendTypesList.toArray(new String[0]);
    AlertDialog.Builder backendTypeBuilder = new AlertDialog.Builder(this);
    backendTypeBuilder.setTitle("Select backend type");
    backendTypeBuilder.setSingleChoiceItems(
        backendTypes,
        -1,
        (dialog, item) -> {
          mBackendTextView.setText(backendTypes[item]);
          mBackendType = BackendType.valueOf(backendTypes[item]);
          setBackendSettingMode();
          dialog.dismiss();
        });

    backendTypeBuilder.create().show();
  }

  private void setupModelSelectorDialog() {
    String[] pteFiles = listLocalFile("/data/local/tmp/llama/", new String[] {".pte"});
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

  private static boolean fileHasExtension(String file, String[] suffix) {
    return Arrays.stream(suffix).anyMatch(entry -> file.endsWith(entry));
  }

  private static String[] listLocalFile(String path, String[] suffix) {
    File directory = new File(path);
    if (directory.exists() && directory.isDirectory()) {
      File[] files = directory.listFiles((dir, name) -> (fileHasExtension(name, suffix)));
      String[] result = new String[files.length];
      for (int i = 0; i < files.length; i++) {
        if (files[i].isFile() && fileHasExtension(files[i].getName(), suffix)) {
          result[i] = files[i].getAbsolutePath();
        }
      }
      return result;
    }
    return new String[] {};
  }

  private void setupModelTypeSelectorDialog() {
    // Convert enum to list
    List<String> modelTypesList = new ArrayList<>();
    for (ModelType modelType : ModelType.values()) {
      modelTypesList.add(modelType.toString());
    }
    // Alert dialog builder takes in arr of string instead of list
    String[] modelTypes = modelTypesList.toArray(new String[0]);
    AlertDialog.Builder modelTypeBuilder = new AlertDialog.Builder(this);
    modelTypeBuilder.setTitle("Select model type");
    modelTypeBuilder.setSingleChoiceItems(
        modelTypes,
        -1,
        (dialog, item) -> {
          mModelTypeTextView.setText(modelTypes[item]);
          mModelType = ModelType.valueOf(modelTypes[item]);
          mUserPromptEditText.setText(PromptFormat.getUserPromptTemplate(mModelType, false));
          dialog.dismiss();
        });

    modelTypeBuilder.create().show();
  }

  private void setupTokenizerSelectorDialog() {
    String[] tokenizerFiles =
        listLocalFile("/data/local/tmp/llama/", new String[] {".bin", ".json", ".model"});
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

  private void setBackendSettingMode() {
    if (mBackendType.equals(BackendType.XNNPACK) || mBackendType.equals(BackendType.QUALCOMM)) {
      setXNNPACKSettingMode();
    } else if (mBackendType.equals(BackendType.MEDIATEK)) {
      setMediaTekSettingMode();
    }
  }

  private void setXNNPACKSettingMode() {
    requireViewById(R.id.modelLayout).setVisibility(View.VISIBLE);
    requireViewById(R.id.tokenizerLayout).setVisibility(View.VISIBLE);
    requireViewById(R.id.parametersView).setVisibility(View.VISIBLE);
    requireViewById(R.id.temperatureLayout).setVisibility(View.VISIBLE);
    mModelFilePath = "";
    mTokenizerFilePath = "";
  }

  private void setMediaTekSettingMode() {
    requireViewById(R.id.modelLayout).setVisibility(View.GONE);
    requireViewById(R.id.tokenizerLayout).setVisibility(View.GONE);
    requireViewById(R.id.parametersView).setVisibility(View.GONE);
    requireViewById(R.id.temperatureLayout).setVisibility(View.GONE);
    mModelFilePath = "/in/mtk/llama/runner";
    mTokenizerFilePath = "/in/mtk/llama/runner";
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
    mSettingsFields.saveModelType(mModelType);
    mSettingsFields.saveBackendType(mBackendType);
    mDemoSharedPreferences.addSettings(mSettingsFields);
  }

  @Override
  public void onBackPressed() {
    super.onBackPressed();
    saveSettings();
  }
}
