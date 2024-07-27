package com.example.executorchllamademo;

public interface ModelRunnerCallback {

    void onModelLoaded(int status);
    void onTokenGenerated(String token);

    void onStats(String token);

    void onGeneratinStopped();
}
