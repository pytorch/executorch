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
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.io.IOException;
import java.io.File;
import java.io.FileOutputStream;
import org.junit.runners.JUnit4;
import org.apache.commons.io.FileUtils;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.InstrumentationRegistry;

/** Unit tests for {@link Module}. */
@RunWith(AndroidJUnit4.class)
public class ModuleInstrumentationTest {
    private static String TEST_FILE_NAME = "/add.pte";
    private static String MISSING_FILE_NAME = "/missing.pte";
    private static String NON_PTE_FILE_NAME = "/test.txt";
    private static String FORWARD_METHOD = "forward";
    private static String NONE_METHOD = "none";
    private static int OK = 0x00;
    private static int INVALID_STATE = 0x2;
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

    @Test
    public void testLoadOnDestroyedModule() throws IOException{
        Module module = Module.load(getTestFilePath(TEST_FILE_NAME));

        module.destroy();

        int loadMethod = module.loadMethod(FORWARD_METHOD);
        assertEquals(loadMethod, INVALID_STATE);
    }

    @Test
    public void testForwardOnDestroyedModule() throws IOException{
        Module module = Module.load(getTestFilePath(TEST_FILE_NAME));

        int loadMethod = module.loadMethod(FORWARD_METHOD);
        assertEquals(loadMethod, OK);

        module.destroy();
        
        EValue[] results = module.forward();
        assertEquals(0, results.length);
    }
    
    @Test
    public void testForwardFromMultipleThreads() throws InterruptedException, IOException {
        Module module = Module.load(getTestFilePath(TEST_FILE_NAME));

        int numThreads = 100;
        CountDownLatch latch = new CountDownLatch(numThreads);
        AtomicInteger completed = new AtomicInteger(0);

        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                try {
                    latch.countDown();
                    latch.await(5000, java.util.concurrent.TimeUnit.MILLISECONDS);
                    EValue[] results = module.forward();
                    assertTrue(results[0].isTensor());
                    completed.incrementAndGet();
                } catch (InterruptedException e) {
        
                }
            }
        };

        Thread[] threads = new Thread[numThreads];
        for (int i = 0; i < numThreads; i++) {
            threads[i] = new Thread(runnable);
            threads[i].start();
        }

        for (int i = 0; i < numThreads; i++) {
            threads[i].join();
        }

        assertEquals(numThreads, completed.get());
    }
}
