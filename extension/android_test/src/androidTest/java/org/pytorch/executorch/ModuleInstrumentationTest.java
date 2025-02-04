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
import java.io.IOException;
import java.io.File;
import java.io.FileOutputStream;
import org.junit.runners.JUnit4;
import org.apache.commons.io.FileUtils;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.InstrumentationRegistry;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Tensor;

/** Unit tests for {@link Module}. */
@RunWith(AndroidJUnit4.class)
public class ModuleInstrumentationTest {
    private static String TEST_FILE_NAME = "/add.pte";
    private static String MISSING_FILE_NAME = "/missing.pte";
    private static String NON_PTE_FILE_NAME = "/test.txt";
    private static String FORWARD_METHOD = "forward";
    private static String NONE_METHOD = "none";
    private static int OK = 0x00;
    private static int INVALID_ARGUMENT = 0x12;
    private static int ACCESS_FAILED = 0x22;

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

        File nonPteFile = new File(getTestFilePath(NON_PTE_FILE_NAME));
        inputStream = getClass().getResourceAsStream(NON_PTE_FILE_NAME);
        FileUtils.copyInputStreamToFile(inputStream, nonPteFile);
        inputStream.close();
    }

    @Rule
    public GrantPermissionRule mRuntimePermissionRule = GrantPermissionRule.grant(Manifest.permission.READ_EXTERNAL_STORAGE);

    @Test
    public void testModuleLoadAndForward() throws IOException, URISyntaxException{
        Module module = Module.load(getTestFilePath(TEST_FILE_NAME));

        EValue[] results = module.forward();
        assertTrue(results[0].isTensor());
    }

    @Test
    public void testModuleLoadMethodAndForward() throws IOException{
        Module module = Module.load(getTestFilePath(TEST_FILE_NAME));

        int loadMethod = module.loadMethod(FORWARD_METHOD);
        assertEquals(loadMethod, OK);

        EValue[] results = module.forward();
        assertTrue(results[0].isTensor());
    }

    @Test
    public void testModuleLoadForwardExplicit() throws IOException{
        Module module = Module.load(getTestFilePath(TEST_FILE_NAME));

        EValue[] results = module.execute(FORWARD_METHOD);
        assertTrue(results[0].isTensor());
    }

    @Test
    public void testModuleLoadNonExistantFile() throws IOException{
        Module module = Module.load(getTestFilePath(MISSING_FILE_NAME));

        EValue[] results = module.forward();
        assertEquals(null, results);
    }

    @Test
    public void testModuleLoadMethodNonExistantFile() throws IOException{
        Module module = Module.load(getTestFilePath(MISSING_FILE_NAME));

        int loadMethod = module.loadMethod(FORWARD_METHOD);
        assertEquals(loadMethod, ACCESS_FAILED);
    }

    @Test
    public void testModuleLoadMethodNonExistantMethod() throws IOException{
        Module module = Module.load(getTestFilePath(TEST_FILE_NAME));

        int loadMethod = module.loadMethod(NONE_METHOD);
        assertEquals(loadMethod, INVALID_ARGUMENT);
    }

    @Test
    public void testNonPteFile() throws IOException{
        Module module = Module.load(getTestFilePath(NON_PTE_FILE_NAME));

        int loadMethod = module.loadMethod(FORWARD_METHOD);
        assertEquals(loadMethod, INVALID_ARGUMENT);
    }
}
