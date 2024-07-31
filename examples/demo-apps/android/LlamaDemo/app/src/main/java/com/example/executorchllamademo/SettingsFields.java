/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo;

public class SettingsFields {
  private static final String SYSTEM_PLACEHOLDER = "{{ system_prompt }}";
  private static final String USER_PLACEHOLDER = "{{ user_prompt }}";
  private static String SYSTEM_PROMPT_TEMPLATE =
      "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
          + SYSTEM_PLACEHOLDER
          + "<|eot_id|>";
  private static String USER_PROMPT_TEMPLATE =
      "<|start_header_id|>user<|end_header_id|>\n"
          + USER_PLACEHOLDER
          + "<|eot_id|>\n"
          + "<|start_header_id|>assistant<|end_header_id|>";

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

  public String getUserPrompt() {
    return userPrompt;
  }

  public String getEntirePrompt() {
    return systemPrompt + userPrompt;
  }

  public String getSystemPromptTemplate() {
    return SYSTEM_PROMPT_TEMPLATE;
  }

  public String getUserPromptTemplate() {
    return USER_PROMPT_TEMPLATE;
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

  public SettingsFields() {
    modelFilePath = "";
    tokenizerFilePath = "";
    temperature = SettingsActivity.TEMPERATURE_MIN_VALUE;
    systemPrompt = SYSTEM_PROMPT_TEMPLATE;
    userPrompt = USER_PROMPT_TEMPLATE;
    isClearChatHistory = false;
    isLoadModel = false;
  }

  public SettingsFields(SettingsFields settingsFields) {
    this.modelFilePath = settingsFields.modelFilePath;
    this.tokenizerFilePath = settingsFields.tokenizerFilePath;
    this.temperature = settingsFields.temperature;
    this.systemPrompt = settingsFields.getSystemPrompt();
    this.userPrompt = settingsFields.getUserPrompt();
    this.isClearChatHistory = settingsFields.getIsClearChatHistory();
    this.isLoadModel = settingsFields.getIsLoadModel();
  }

  public void saveModelPath(String modelFilePath) {
    this.modelFilePath = modelFilePath;
  }

  public void saveTokenizerPath(String tokenizerFilePath) {
    this.tokenizerFilePath = tokenizerFilePath;
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
        && isLoadModel == anotherSettingsFields.isLoadModel;
  }

  public boolean isSystemPromptChanged() {
    return !systemPrompt.contains(SYSTEM_PLACEHOLDER);
  }

  public boolean isUserPromptChanged() {
    return !userPrompt.contains(USER_PLACEHOLDER);
  }
}
