package com.example.qnntest;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.ScrollView;
import android.widget.LinearLayout;
import android.widget.TextView;

public class MainActivity extends Activity {

    private static final String TAG = "QnnHtpTest";
    private TextView logView;
    private Button testButton;

    static {
        System.loadLibrary("qnn_test_jni");
    }

    private native String runQnnHtpTest(String nativeLibDir);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(32, 32, 32, 32);

        testButton = new Button(this);
        testButton.setText("Run QNN HTP Test");
        testButton.setTextSize(18);
        layout.addView(testButton);

        ScrollView scrollView = new ScrollView(this);
        logView = new TextView(this);
        logView.setPadding(0, 24, 0, 0);
        logView.setTextSize(13);
        logView.setTextIsSelectable(true);
        scrollView.addView(logView);
        layout.addView(scrollView);

        setContentView(layout);

        String nativeLibDir = getApplicationInfo().nativeLibraryDir;

        testButton.setOnClickListener(v -> {
            testButton.setEnabled(false);
            testButton.setText("Running...");
            logView.setText("");
            appendLog("nativeLibraryDir: " + nativeLibDir + "\n");

            new Thread(() -> {
                String result = runQnnHtpTest(nativeLibDir);
                Log.i(TAG, result);
                runOnUiThread(() -> {
                    appendLog(result);
                    testButton.setEnabled(true);
                    testButton.setText("Run QNN HTP Test");
                });
            }).start();
        });
    }

    private void appendLog(String msg) {
        Log.i(TAG, msg);
        logView.append(msg + "\n");
    }
}
