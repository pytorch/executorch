/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.minibench;


import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import java.io.FileWriter;
import java.io.IOException;

public class LlmBenchmarkActivity extends Activity implements ModelRunnerCallback {
    ModelRunner mModelRunner;

    String mPrompt;
    StatsDump mStatsDump;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Intent intent = getIntent();

        String modelPath = intent.getStringExtra("model_path");
        String tokenizerPath = intent.getStringExtra("tokenizer_path");

        float temperature = intent.getFloatExtra("temperature", 0.8f);
        mPrompt = intent.getStringExtra("prompt");
        if (mPrompt == null) {
            mPrompt = "The ultimate answer";
        }

        mStatsDump = new StatsDump();
        mModelRunner = new ModelRunner(modelPath, tokenizerPath, temperature, this);
        mStatsDump.loadStart = System.currentTimeMillis();
    }

    @Override
    public void onModelLoaded(int status) {
        mStatsDump.loadEnd = System.currentTimeMillis();
        if (status != 0) {
            Log.e("LlmBenchmarkRunner", "Loaded failed: " + status);
            onGenerationStopped();
            return;
        }
        mStatsDump.generateStart = System.currentTimeMillis();
        int generateStatus = mModelRunner.generate(mPrompt);
    }

    @Override
    public void onTokenGenerated(String token) {
    }

    @Override
    public void onStats(String stats) {
        mStatsDump.tokens = stats;
    }

    @Override
    public void onGenerationStopped() {
        mStatsDump.generateEnd = System.currentTimeMillis();

        try (FileWriter writer = new FileWriter(getFilesDir() + "/benchmark_results.txt")) {
            writer.write(mStatsDump.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

class StatsDump {
    long loadStart;
    long loadEnd;
    long generateStart;
    long generateEnd;
    String tokens;

    @Override
    public String toString() {
        return "loadStart: "
                + loadStart
                + "\nloadEnd: "
                + loadEnd
                + "\ngenerateStart: "
                + generateStart
                + "\ngenerateEnd: "
                + generateEnd
                + "\n"
                + tokens;
    }
}