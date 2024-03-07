/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo;

public class Message {
  private String text;
  private boolean isSent;
  private float tokensPerSecond;

  public Message(String text, boolean isSent) {
    this.text = text;
    this.isSent = isSent;
  }

  public String getText() {
    return text;
  }

  public void appendText(String text) {
    this.text += text;
  }

  public boolean getIsSent() {
    return isSent;
  }

  public void setTokensPerSecond(float tokensPerSecond) {
    this.tokensPerSecond = tokensPerSecond;
  }

  public float getTokensPerSecond() {
    return tokensPerSecond;
  }
}
