/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.fail;

import android.os.Environment;
import androidx.test.rule.GrantPermissionRule;
import android.Manifest;
import android.content.Context;
import org.junit.Test;
import org.junit.Before;
import org.junit.Rule;
import org.junit.runner.RunWith;
import java.io.InputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.List;
import java.util.ArrayList;
import java.io.IOException;
import java.io.File;
import java.io.FileOutputStream;
import org.junit.runners.JUnit4;
import org.apache.commons.io.FileUtils;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.InstrumentationRegistry;
import org.json.JSONException;
import org.json.JSONObject;
import org.pytorch.executorch.extension.llm.LlmCallback;
import org.pytorch.executorch.extension.llm.LlmModule;

/** Unit tests for {@link org.pytorch.executorch.extension.llm.LlmModule}. */
@RunWith(AndroidJUnit4.class)
public class LlmModuleInstrumentationTest implements LlmCallback {
    private static String TEST_FILE_NAME = "/stories.pte";
    private static String TOKENIZER_FILE_NAME = "/tokenizer.bin";
    private static String TEST_PROMPT = "Hello";
    private static int OK = 0x00;
    private static int SEQ_LEN = 32;

    private final List<String> results = new ArrayList<>();
    private final List<Float> tokensPerSecond = new ArrayList<>();
    private LlmModule mModule;

    private static String getTestFilePath(String fileName) {
        return InstrumentationRegistry.getInstrumentation().getTargetContext().getExternalCacheDir() + fileName;
    }

    @Before
    public void setUp() throws IOException {
        // copy zipped test resources to local device
        File addPteFile = new File(getTestFilePath(TEST_FILE_NAME));
        InputStream inputStream = getClass().getResourceAsStream(TEST_FILE_NAME);
        FileUtils.copyInputStreamToFile(inputStream, addPteFile);
        inputStream.close();

        File tokenizerFile = new File(getTestFilePath(TOKENIZER_FILE_NAME));
        inputStream = getClass().getResourceAsStream(TOKENIZER_FILE_NAME);
        FileUtils.copyInputStreamToFile(inputStream, tokenizerFile);
        inputStream.close();

        mModule = new LlmModule(getTestFilePath(TEST_FILE_NAME), getTestFilePath(TOKENIZER_FILE_NAME), 0.0f);
    }

    @Rule
    public GrantPermissionRule mRuntimePermissionRule = GrantPermissionRule.grant(Manifest.permission.READ_EXTERNAL_STORAGE);

    @Test
    public void testGenerate() throws IOException, URISyntaxException{
        int loadResult = mModule.load();
        // Check that the model can be load successfully
        assertEquals(OK, loadResult);

        mModule.generate(TEST_PROMPT, SEQ_LEN, LlmModuleInstrumentationTest.this);
        assertEquals(results.size(), SEQ_LEN);
        assertTrue(tokensPerSecond.get(tokensPerSecond.size() - 1) > 0);
    }

    @Test
    public void testGenerateAndStop() throws IOException, URISyntaxException{
        mModule.generate(TEST_PROMPT, SEQ_LEN, new LlmCallback() {
            @Override
            public void onResult(String result) {
                LlmModuleInstrumentationTest.this.onResult(result);
                mModule.stop();
            }

            @Override
            public void onStats(String stats) {
                LlmModuleInstrumentationTest.this.onStats(stats);
            }
        });

        int stoppedResultSize = results.size();
        assertTrue(stoppedResultSize < SEQ_LEN);
    }

    @Override
    public void onResult(String result) {
        results.add(result);
    }

    @Override
    public void onStats(String stats) {
        float tps = 0;
        try {
            JSONObject jsonObject = new JSONObject(stats);
            int numGeneratedTokens = jsonObject.getInt("generated_tokens");
            int inferenceEndMs = jsonObject.getInt("inference_end_ms");
            int promptEvalEndMs = jsonObject.getInt("prompt_eval_end_ms");
            tps = (float) numGeneratedTokens / (inferenceEndMs - promptEvalEndMs) * 1000;
            tokensPerSecond.add(tps);
        } catch (JSONException e) {
        }
    }
}
