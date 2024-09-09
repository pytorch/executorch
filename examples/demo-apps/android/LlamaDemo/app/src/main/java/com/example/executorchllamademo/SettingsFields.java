/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo;

public class SettingsFields {

  public String getModelFilePath() {
    return modelFilePath;
  }

  public String getTokenizerFilePath() {
    return tokenizerFilePath;
  }

  public double getTemperature() {
    return temperature;
  }

  public String getSystemPrompt() {
    return systemPrompt;
  }

  public ModelType getModelType() {
    return modelType;
  }

  public String getUserPrompt() {
    return userPrompt;
  }

  public String getFormattedSystemAndUserPrompt(String prompt) {
    return getFormattedSystemPrompt() + getFormattedUserPrompt(prompt);
  }

  public String getFormattedSystemPrompt() {
    return PromptFormat.getSystemPromptTemplate(modelType)
        .replace(PromptFormat.SYSTEM_PLACEHOLDER, systemPrompt);
  }

  public String getFormattedUserPrompt(String prompt) {
    return userPrompt.replace(PromptFormat.USER_PLACEHOLDER, prompt);
  }

  public boolean getIsClearChatHistory() {
    return isClearChatHistory;
  }

  public boolean getIsLoadModel() {
    return isLoadModel;
  }

  private String modelFilePath;
  private String tokenizerFilePath;
  private double temperature;
  private String systemPrompt;
  private String userPrompt;
  private boolean isClearChatHistory;
  private boolean isLoadModel;
  private ModelType modelType;

  public SettingsFields() {
    ModelType DEFAULT_MODEL = ModelType.LLAMA_3;

    modelFilePath = "";
    tokenizerFilePath = "";
    temperature = SettingsActivity.TEMPERATURE_MIN_VALUE;
    systemPrompt = "";
    userPrompt = PromptFormat.getUserPromptTemplate(DEFAULT_MODEL);
    isClearChatHistory = false;
    isLoadModel = false;
    modelType = DEFAULT_MODEL;
  }

  public SettingsFields(SettingsFields settingsFields) {
    this.modelFilePath = settingsFields.modelFilePath;
    this.tokenizerFilePath = settingsFields.tokenizerFilePath;
    this.temperature = settingsFields.temperature;
    this.systemPrompt = settingsFields.getSystemPrompt();
    this.userPrompt = settingsFields.getUserPrompt();
    this.isClearChatHistory = settingsFields.getIsClearChatHistory();
    this.isLoadModel = settingsFields.getIsLoadModel();
    this.modelType = settingsFields.modelType;
  }

  public void saveModelPath(String modelFilePath) {
    this.modelFilePath = modelFilePath;
  }

  public void saveTokenizerPath(String tokenizerFilePath) {
    this.tokenizerFilePath = tokenizerFilePath;
  }

  public void saveModelType(ModelType modelType) {
    this.modelType = modelType;
  }

  public void saveParameters(Double temperature) {
    this.temperature = temperature;
  }

  public void savePrompts(String systemPrompt, String userPrompt) {
    this.systemPrompt = systemPrompt;
    this.userPrompt = userPrompt;
  }

  public void saveIsClearChatHistory(boolean needToClear) {
    this.isClearChatHistory = needToClear;
  }

  public void saveLoadModelAction(boolean shouldLoadModel) {
    this.isLoadModel = shouldLoadModel;
  }

  public boolean equals(SettingsFields anotherSettingsFields) {
    if (this == anotherSettingsFields) return true;
    return modelFilePath.equals(anotherSettingsFields.modelFilePath)
        && tokenizerFilePath.equals(anotherSettingsFields.tokenizerFilePath)
        && temperature == anotherSettingsFields.temperature
        && systemPrompt.equals(anotherSettingsFields.systemPrompt)
        && userPrompt.equals(anotherSettingsFields.userPrompt)
        && isClearChatHistory == anotherSettingsFields.isClearChatHistory
        && isLoadModel == anotherSettingsFields.isLoadModel
        && modelType == anotherSettingsFields.modelType;
  }
}
