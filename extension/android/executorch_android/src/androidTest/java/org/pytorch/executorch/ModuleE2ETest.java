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

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
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

    static int argmax(float[] array) {
        if (array.length == 0) {
            throw new IllegalArgumentException("Array cannot be empty");
        }
        int maxIndex = 0;
        float maxValue = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public void testClassification(String filePath) throws IOException, URISyntaxException {
        File pteFile = new File(getTestFilePath(filePath));
        InputStream inputStream = getClass().getResourceAsStream(filePath);
        FileUtils.copyInputStreamToFile(inputStream, pteFile);
        inputStream.close();

        InputStream imgInputStream = getClass().getResourceAsStream("/banana.jpeg");
        Bitmap bitmap = BitmapFactory.decodeStream(imgInputStream);
        bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
        imgInputStream.close();

        Tensor inputTensor =
        TensorImageUtils.bitmapToFloat32Tensor(
            bitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB);

        Module module = Module.load(getTestFilePath(filePath));

        EValue[] results = module.forward(EValue.from(inputTensor));
        assertTrue(results[0].isTensor());
        float[] scores = results[0].toTensor().getDataAsFloatArray();

        int bananaClass = 954;  // From ImageNet 1K
        assertEquals(bananaClass, argmax(scores));
    }

    @Test
    public void testMv2Fp32() throws IOException, URISyntaxException {
        testClassification("/mv2_xnnpack_fp32.pte");
    }

    @Test
    public void testMv3Fp32() throws IOException, URISyntaxException {
        testClassification("/mv3_xnnpack_fp32.pte");
    }

    @Test
    public void testResnet50() throws IOException, URISyntaxException {
        testClassification("/resnet50_xnnpack_q8.pte");
    }
}
