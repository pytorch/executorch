/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.TextView;

public class MessageAdapter extends ArrayAdapter<Message> {
  public MessageAdapter(android.content.Context context, int resource) {
    super(context, resource);
  }

  @Override
  public View getView(int position, View convertView, ViewGroup parent) {
    Message currentMessage = getItem(position);

    int layoutIdForListItem =
        currentMessage.getIsSent() ? R.layout.sent_message : R.layout.received_message;
    View listItemView =
        LayoutInflater.from(getContext()).inflate(layoutIdForListItem, parent, false);
    TextView messageTextView = listItemView.findViewById(R.id.message_text);
    messageTextView.setText(currentMessage.getText());

    if (currentMessage.getTokensPerSecond() > 0) {
      TextView tokensView = listItemView.findViewById(R.id.tokens_per_second);
      tokensView.setText("" + currentMessage.getTokensPerSecond() + " t/s");
    }

    return listItemView;
  }
}
