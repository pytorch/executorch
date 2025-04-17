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
public class ModuleE2ETest {
    private static String getTestFilePath(String fileName) {
        return InstrumentationRegistry.getInstrumentation().getTargetContext().getExternalCacheDir() + fileName;
    }

    @Rule
    public GrantPermissionRule mRuntimePermissionRule = GrantPermissionRule.grant(Manifest.permission.READ_EXTERNAL_STORAGE);

    @Test
    public void testMv2Fp32() throws IOException, URISyntaxException{
        String filePath = "/mv2_xnnpack_fp32.pte";
        File pteFile = new File(getTestFilePath(filePath));
        InputStream inputStream = getClass().getResourceAsStream(filePath);
        FileUtils.copyInputStreamToFile(inputStream, pteFile);
        inputStream.close();

        Module module = Module.load(getTestFilePath(filePath));

        EValue[] results = module.forward();
        assertTrue(results[0].isTensor());
    }

}
