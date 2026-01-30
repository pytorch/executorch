/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorch;

import java.util.Scanner;
import org.pytorch.executorch.extension.llm.LlmCallback;
import org.pytorch.executorch.extension.llm.LlmModule;

/**
 * Interactive chat application using LlmModule for text generation.
 * 
 * Usage: LlamaChat <path_to_model.pte> <path_to_tokenizer>
 */
public class LlamaChat {
    private static final int SEQ_LEN = 512;
    private static final boolean ECHO = false;
    private static final float TEMPERATURE = 0.7f;

    public static void main(String[] args) {
        System.out.println("LlamaChat: Starting...");
        
        if (args.length < 2) {
            System.out.println("Usage: LlamaChat <path_to_model.pte> <path_to_tokenizer>");
            System.exit(1);
        }

        String ptePath = args[0];
        String tokenizerPath = args[1];

        try {
            System.out.println("Loading model: " + ptePath);
            System.out.println("Loading tokenizer: " + tokenizerPath);

            // Create the LlmModule
            LlmModule module = new LlmModule(LlmModule.MODEL_TYPE_TEXT, ptePath, tokenizerPath, TEMPERATURE);
            
            // Load the model
            int loadResult = module.load();
            if (loadResult != 0) {
                System.err.println("Failed to load model, error code: " + loadResult);
                System.exit(1);
            }
            System.out.println("Model loaded successfully.");
            System.out.println();

            // Start interactive chat loop
            Scanner scanner = new Scanner(System.in);
            System.out.println("=== LlamaChat ===");
            System.out.println("Type your message and press Enter. Type 'quit' or 'exit' to end.");
            System.out.println();

            while (true) {
                System.out.print("You: ");
                System.out.flush();
                
                String input = scanner.nextLine();
                
                if (input == null || input.trim().isEmpty()) {
                    continue;
                }
                
                String trimmedInput = input.trim().toLowerCase();
                if (trimmedInput.equals("quit") || trimmedInput.equals("exit")) {
                    System.out.println("Goodbye!");
                    break;
                }

                System.out.print("Assistant: ");
                System.out.flush();

                StringBuilder response = new StringBuilder();

                // Create callback to print tokens as they are generated
                LlmCallback callback = new LlmCallback() {
                    @Override
                    public void onResult(String result) {
                        response.append(result);

                    }

                    @Override
                    public void onStats(String stats) {
                        // Optionally print stats for debugging
                        // System.out.println("\n[Stats: " + stats + "]");
                    }
                };

                // Generate response
                int result = module.generate(input, SEQ_LEN, callback, ECHO, TEMPERATURE);
                
                if (result != 0) {
                    System.out.println("\n[Generation ended with code: " + result + "]");
                }
                
                System.out.println(response.toString());
                System.out.println();
            }

            // Clean up
            scanner.close();
            module.destroy();
            System.out.println("LlamaChat: Finished.");

        } catch (Exception e) {
            System.err.println("Error occurred:");
            e.printStackTrace();
            System.exit(1);
        }
    }
}
