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
import android.widget.Toast;
import android.app.ProgressDialog;
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
import java.net.HttpURLConnection;
import java.io.FileOutputStream;
import java.io.FileInputStream;
import java.io.InputStream;
import java.net.URL;

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
    tokenizerImageButton.setEnabled(false);
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
                      mUserPromptEditText.setText(PromptFormat.getUserPromptTemplate(mModelType));
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
              mUserPromptEditText.setText(PromptFormat.getUserPromptTemplate(mModelType));
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

  private static class ModelInfo {
    String modelName;
    String tokenizerUrl;
    String modelUrl;
    String quantAttrsUrl;

    ModelInfo(String modelName, String tokenizerUrl, String modelUrl, String quantAttrsUrl) {
      this.modelName = modelName;
      this.tokenizerUrl = tokenizerUrl;
      this.modelUrl = modelUrl;
      this.quantAttrsUrl = quantAttrsUrl;
    }
  }

  // Construct the model info array
  private final ModelInfo[] modelInfoArray = new ModelInfo[] {
    new ModelInfo(
      "bitnet-b1.58-2B-4T",
      "https://huggingface.co/JY-W/test_model/resolve/main/tokenizer.json?download=true",
      "https://huggingface.co/JY-W/test_model/resolve/main/kv_llama_qnn.pte?download=true",
      "https://huggingface.co/JY-W/test_model/resolve/main/kv_llama_qnn_quant_attrs.txt?download=true"
    ),
    new ModelInfo(
      "llama-3.1-8B-Instruct",
      "https://huggingface.co/JY-W/test_model/resolve/main/llama_3_1_8b_bitdistiller_tokenizer.json?download=true",
      "https://huggingface.co/JY-W/test_model/resolve/main/llama_3_1_8b_instruct_bitdistiller.pte?download=true",
      "https://huggingface.co/JY-W/test_model/resolve/main/llama_3_1_8b_bitdistiller_quant_attrs.txt?download=true"
    ),
    new ModelInfo(
      "llama-3.1-8B-Instruct-LongContext",
      "https://huggingface.co/JY-W/test_model/resolve/main/llama_3_1_8b_bitdistiller_tokenizer.json?download=true",
      "https://huggingface.co/JY-W/test_model/resolve/main/llama_3_1_8b_instruct_bitdistiller_ctx1024.pte?download=true",
      "https://huggingface.co/JY-W/test_model/resolve/main/llama_3_1_8b_bitdistiller_ctx1024_quant_attrs.txt?download=true"
    ),
    new ModelInfo(
      "Qwen-3-8B",
      "https://huggingface.co/Qwen/Qwen3-8B/resolve/main/tokenizer.json?download=true",
      "https://huggingface.co/JY-W/test_model/resolve/main/qwen3_8b_bitdistiller.pte?download=true",
      "https://huggingface.co/JY-W/test_model/resolve/main/qwen3_8b_bitdistiller_quant_attrs.txt?download=true"
    )
  };

  private final String mOpPackageUrl = "https://huggingface.co/JY-W/test_model/resolve/main/libQnnTMANOpPackage.so?download=true";

  private void downloadFileFromUrl(String fileUrl, String fileName, boolean overwrite) {
    ProgressDialog progressDialog = new ProgressDialog(this);
    progressDialog.setTitle("Downloading " + fileName + "...");
    progressDialog.setMessage("Please wait...");
    progressDialog.setProgressStyle(ProgressDialog.STYLE_HORIZONTAL);
    progressDialog.setIndeterminate(false);
    progressDialog.setMax(100);
    progressDialog.setProgress(0);
    progressDialog.setCancelable(false);
    progressDialog.show();

    new Thread(() -> {
        try {
            File outputDir = getExternalFilesDir(null);
            File outputFile = new File(outputDir, fileName);
            if (!outputFile.exists() || overwrite) {
                URL url = new URL(fileUrl);
                HttpURLConnection connection = (HttpURLConnection) url.openConnection();
                connection.connect();

                if (connection.getResponseCode() != HttpURLConnection.HTTP_OK) {
                    runOnUiThread(() -> {
                        progressDialog.dismiss();
                        Toast.makeText(this, "Failed to download file", Toast.LENGTH_SHORT).show();
                    });
                    return;
                }

                long fileLength = connection.getContentLengthLong();
                InputStream input = connection.getInputStream();

                try (FileOutputStream output = new FileOutputStream(outputFile)) {
                    byte[] buffer = new byte[4096];
                    int bytesRead;
                    long totalBytesRead = 0;

                    while ((bytesRead = input.read(buffer)) != -1) {
                        totalBytesRead += bytesRead;
                        output.write(buffer, 0, bytesRead);

                        // Update progress
                        int progress = (int) (totalBytesRead * 100 / fileLength);
                        runOnUiThread(() -> progressDialog.setProgress(progress));
                    }
                }

                connection.disconnect();
            }

            runOnUiThread(() -> {
                progressDialog.dismiss();
                mLoadModelButton.setEnabled(true);
                Toast.makeText(this, "File downloaded to " + outputFile.getAbsolutePath(), Toast.LENGTH_SHORT).show();
            });
        } catch (Exception e) {
            runOnUiThread(() -> {
                progressDialog.dismiss();
                Toast.makeText(this, "Error: " + e.getMessage(), Toast.LENGTH_SHORT).show();
            });
        }
    }).start();
  }

  private void downloadModel(ModelInfo modelInfo) {
    String modelFileName = modelInfo.modelName + ".pte";
    String tokenizerFileName = modelInfo.modelName + "_tokenizer.json";
    String quantAttrsFileName = modelInfo.modelName + "_quant_attrs.txt";
    String opPackageFileName = "libQnnTMANOpPackage.so";

    File modelFile = new File(getExternalFilesDir(null), modelFileName);
    File tokenizerFile = new File(getExternalFilesDir(null), tokenizerFileName);
    File quantAttrsFile = new File(getExternalFilesDir(null), quantAttrsFileName);
    File opPackageFile = new File(getExternalFilesDir(null), opPackageFileName);

    if (modelFile.exists() || tokenizerFile.exists() || quantAttrsFile.exists() || opPackageFile.exists()) {
      runOnUiThread(() -> {
          new AlertDialog.Builder(this)
            .setTitle("Overwrite Existing Files")
            .setMessage("Some files for this model already exist. Do you want to overwrite them?")
            .setPositiveButton("Yes", (dialog, which) -> {
              downloadFileFromUrl(modelInfo.modelUrl, modelFileName, true);
              downloadFileFromUrl(modelInfo.tokenizerUrl, tokenizerFileName, true);
              downloadFileFromUrl(modelInfo.quantAttrsUrl, quantAttrsFileName, true);
              downloadFileFromUrl(mOpPackageUrl, opPackageFileName, true);
            })
            .setNegativeButton("No", (dialog, which) -> {
              downloadFileFromUrl(modelInfo.modelUrl, modelFileName, false);
              downloadFileFromUrl(modelInfo.tokenizerUrl, tokenizerFileName, false);
              downloadFileFromUrl(modelInfo.quantAttrsUrl, quantAttrsFileName, false);
              downloadFileFromUrl(mOpPackageUrl, opPackageFileName, false);
            })
            .show();
      });
    } else {
      downloadFileFromUrl(modelInfo.modelUrl, modelFileName, true);
      downloadFileFromUrl(modelInfo.tokenizerUrl, tokenizerFileName, true);
      downloadFileFromUrl(modelInfo.quantAttrsUrl, quantAttrsFileName, true);
      downloadFileFromUrl(mOpPackageUrl, opPackageFileName, true);
    }
    mModelFilePath = modelFile.getAbsolutePath();
    mModelTextView.setText(getFilenameFromPath(mModelFilePath));
    mTokenizerFilePath = tokenizerFile.getAbsolutePath();
    mTokenizerTextView.setText(getFilenameFromPath(mTokenizerFilePath));
  }

  private void setupModelSelectorDialog() {
    // set a map from model name to url
    AlertDialog.Builder modelPathBuilder = new AlertDialog.Builder(this);
    modelPathBuilder.setTitle("Select model");

    String[] modelNames = Arrays.stream(modelInfoArray)
                                .map(modelInfo -> modelInfo.modelName)
                                .toArray(String[]::new);

    modelPathBuilder.setSingleChoiceItems(
        modelNames,
        -1,
        (dialog, item) -> {
            ModelInfo selectedModel = modelInfoArray[item];
            downloadModel(selectedModel);
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
          mUserPromptEditText.setText(PromptFormat.getUserPromptTemplate(mModelType));
          dialog.dismiss();
        });

    modelTypeBuilder.create().show();
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
