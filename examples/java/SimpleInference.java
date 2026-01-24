package com.example.executorch;

import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;
import org.pytorch.executorch.EValue;

public class SimpleInference {
    public static void main(String[] args) {
        System.out.println("SimpleInference: Starting...");
        try {
            // 1. Load the library
            // Note: System.loadLibrary("executorch_jni") is typically called inside Module.java static block via NativeLoader
            // But we need to ensure the library path is set correctly (-Djava.library.path=...)

            if (args.length < 1) {
                System.out.println("Usage: SimpleInference <path_to_model.pte>");
                return;
            }

            String modelPath = args[0];
            System.out.println("Loading model: " + modelPath);

            // 2. Load the Module
            Module module = Module.load(modelPath);
            System.out.println("Model loaded successfully.");

            // 3. Prepare inputs (Example: assumes model takes 1 float tensor)
            // Ideally we'd inspect the model metadata if possible, or just run 'forward' with dummy data matching the model.
            // For general verification, just loading might be enough if we don't have a specific model schema.
            // Let's try to run forward with no args or catch exception if it fails.
            
            System.out.println("Methods: ");
            for(String m : module.getMethods()) {
                System.out.println(" - " + m);
            }

            // Optional: Try a simple execution if we knew the input shape.
            // For now, success is "it loaded and didn't crash".

        } catch (Exception e) {
            System.err.println("Error occurred:");
            e.printStackTrace();
            System.exit(1);
        }
        System.out.println("SimpleInference: Finished.");
    }
}
