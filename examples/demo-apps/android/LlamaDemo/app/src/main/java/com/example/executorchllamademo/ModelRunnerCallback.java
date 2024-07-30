package com.example.executorchllamademo;


/**
 * A helper interface within the app for MainActivity and Benchmarking to handle callback from
 * ModelRunner.
 */
public interface ModelRunnerCallback {

    void onModelLoaded(int status);

    void onTokenGenerated(String token);

    void onStats(String token);

    void onGenerationStopped();
}
