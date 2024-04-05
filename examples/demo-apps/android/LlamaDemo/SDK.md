# Guide to set up Java/SDK/NDK for Android

## Set up Java 17
Download the archive from Oracle website:
- Linux: `wget https://download.oracle.com/java/17/archive/jdk-17.0.10_linux-x64_bin.tar.gz`
- macOS: `wget https://download.oracle.com/java/17/archive/jdk-17.0.10_macos-aarch64_bin.tar.gz`

Then unzip the archive. The directory named `jdk-17.0.10` is the Java root directory.

Export `JAVA_HOME` to that `jdk-17.0.10` directory.
```bash
export JAVA_HOME=<path-to>/jdk-17.0.10
export PATH="$JAVA_HOME/bin:$PATH"
```

Note: Oracle has tutorials for installing Java on
[Linux](https://docs.oracle.com/en/java/javase/17/install/installation-jdk-linux-platforms.html#GUID-4A6BD592-1840-4BB4-A758-4CD49E9EE88B)
and [macOS](https://docs.oracle.com/en/java/javase/17/install/installation-jdk-macos.html#GUID-E8A251B6-D9A9-4276-ABC8-CC0DAD62EA33).
Some Linux distributions has JDK package in package manager. For example, Debian users can install
openjdk-17-jdk package.

## Set up Android SDK/NDK
Android has a command line tool [sdkmanager](https://developer.android.com/tools/sdkmanager) which
helps users managing SDK and other tools related to Android development.

1. Go to https://developer.android.com/studio and download the archive from "Command line tools 
only" section. 
   * Make sure you have read and agree with the terms and conditions from the website, then
   * Linux: `wget https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip`
   * macOS: `wget https://dl.google.com/android/repository/commandlinetools-mac-11076708_latest.zip`
2. Unzip: (Linux) `unzip commandlinetools-linux-11076708_latest.zip`
3. Specify a root for Android SDK. For example, we can put it under `~/sdk`.

```
export ANDROID_HOME="$(realpath ~/sdk)"
# Install SDK 34
./cmdline-tools/bin/sdkmanager --sdk_root="${ANDROID_HOME}" --install "platforms;android-34"
# Install NDK
./cmdline-tools/bin/sdkmanager --sdk_root="${ANDROID_HOME}" --install "ndk;25.0.8775105"
# The NDK root is then under `ndk/<version>`.
export ANDROID_NDK="$ANDROID_HOME/ndk/25.0.8775105"
```
