# Guide to set up Java/SDK/NDK for Android

Follow this doc if you haven't set up Java/SDK/NDK for Android development
already.
This doc provides a CLI tutorial to set them up. Otherwise, you can do the same
thing with Android Studio GUI.

## Set up Java 17
1. Download the archive from Oracle website.
Make sure you have read and agree with the terms and conditions from the website before downloading.
```bash
export DEV_HOME=<path-to-dev>
cd $DEV_HOME
```
Linux:
```bash
curl https://download.oracle.com/java/17/archive/jdk-17.0.10_linux-x64_bin.tar.gz -o jdk-17.0.10.tar.gz
```
macOS:
```bash
curl https://download.oracle.com/java/17/archive/jdk-17.0.10_macos-aarch64_bin.tar.gz -o jdk-17.0.10.tar.gz
```
2. Unzip the archive. The directory named `jdk-17.0.10` is the Java root directory.
```bash
tar xf jdk-17.0.10.tar.gz
```
3. Set `JAVA_HOME` and update `PATH`.

Linux:
```bash
export JAVA_HOME="$DEV_HOME"/jdk-17.0.10
export PATH="$JAVA_HOME/bin:$PATH"
```
macOS:
```bash
export JAVA_HOME="$DEV_HOME"/jdk-17.0.10.jdk/Contents/Home
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
only" section. Make sure you have read and agree with the terms and conditions from the website.

Linux:
```bash
curl https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip -o commandlinetools.zip
```
macOS:
```bash
curl https://dl.google.com/android/repository/commandlinetools-mac-11076708_latest.zip -o commandlinetools.zip
```
2. Unzip.
```bash
unzip commandlinetools.zip
```
3. Specify a root for Android SDK. For example, we can put it under `$DEV_HOME/sdk`.

```
mkdir -p $DEV_HOME/sdk
export ANDROID_HOME="$(realpath $DEV_HOME/sdk)"
# Install SDK 34
./cmdline-tools/bin/sdkmanager --sdk_root="${ANDROID_HOME}" --install "platforms;android-34"
# Install NDK
./cmdline-tools/bin/sdkmanager --sdk_root="${ANDROID_HOME}" --install "ndk;25.0.8775105"
# The NDK root is then under `ndk/<version>`.
export ANDROID_NDK="$ANDROID_HOME/ndk/25.0.8775105"
```

### (Optional) Android Studio Setup
If you want to use Android Studio and never set up Java/SDK/NDK before, or if
you use the newly installed ones, follow these steps to set Android Studio to use
them.

Copy these output paths to be used by Android Studio
```bash
echo $ANDROID_HOME
echo $ANDROID_NDK
echo $JAVA_HOME
```

Open a project in Android Studio. In Project Structure (File -> Project
Structure, or `⌘;`) -> SDK Location,
* Set Android SDK Location to the path of $ANDROID_HOME
* Set Android NDK Location to the path of $ANDROID_NDK
* Set JDK location (Click Gradle Settings link) -> Gradle JDK -> Add JDK... to the path of $JAVA_HOME
