/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

plugins { id("com.android.application") }

android {
  namespace = "org.pytorch.minibench"
  compileSdk = 34

  defaultConfig {
    applicationId = "org.pytorch.minibench"
    minSdk = 28
    targetSdk = 33
    versionCode = 1
    versionName = "1.0"

    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
  }

  buildTypes {
    release {
      isMinifyEnabled = false
      proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
    }
  }
  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_1_8
    targetCompatibility = JavaVersion.VERSION_1_8
  }
}

dependencies {
  implementation(files("libs/executorch.aar"))
  implementation("com.facebook.soloader:soloader:0.10.5")
  implementation("com.facebook.fbjni:fbjni:0.5.1")
  implementation("com.google.code.gson:gson:2.8.6")
  testImplementation("junit:junit:4.13.2")
  androidTestImplementation("androidx.test.ext:junit:1.2.1")
  androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
}
