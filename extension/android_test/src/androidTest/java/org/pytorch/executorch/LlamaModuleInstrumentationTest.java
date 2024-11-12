/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorch;

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
import org.pytorch.executorch.LlamaModule;
import org.pytorch.executorch.LlamaCallback;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Tensor;

/** Unit tests for {@link LlamaModule}. */
@RunWith(AndroidJUnit4.class)
public class LlamaModuleInstrumentationTest implements LlamaCallback {
    private static String TEST_FILE_NAME = "/tinyllama_portable_fp16_h.pte";
    private static String TOKENIZER_FILE_NAME = "/tokenizer.bin";
    private static String TEST_PROMPT = "Hello";
    private static int OK = 0x00;
    private static int SEQ_LEN = 32;

    private final List<String> results = new ArrayList<>();
    private final List<Float> tokensPerSecond = new ArrayList<>();
    private LlamaModule mModule;

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

        mModule = new LlamaModule(getTestFilePath(TEST_FILE_NAME), getTestFilePath(TOKENIZER_FILE_NAME), 0.0f);
    }

    @Rule
    public GrantPermissionRule mRuntimePermissionRule = GrantPermissionRule.grant(Manifest.permission.READ_EXTERNAL_STORAGE);

    @Test
    public void testGenerate() throws IOException, URISyntaxException{
        int loadResult = mModule.load();
        // Check that the model can be load successfully
        assertEquals(OK, loadResult);

        mModule.generate(TEST_PROMPT, SEQ_LEN, LlamaModuleInstrumentationTest.this);
        assertEquals(results.size(), SEQ_LEN);
        assertTrue(tokensPerSecond.get(tokensPerSecond.size() - 1) > 0);
    }

    @Test
    public void testGenerateAndStop() throws IOException, URISyntaxException{
        int seqLen = 32;
        mModule.generate(TEST_PROMPT, SEQ_LEN, new LlamaCallback() {
            @Override
            public void onResult(String result) {
                LlamaModuleInstrumentationTest.this.onResult(result);
                mModule.stop();
            }

            @Override
            public void onStats(float tps) {
                LlamaModuleInstrumentationTest.this.onStats(tps);
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
    public void onStats(float tps) {
        tokensPerSecond.add(tps);
    }
}
