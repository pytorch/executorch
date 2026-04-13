<!-- Copyright 2025-2026 Arm Limited and/or its affiliates. -->
# The following file contains all models that have been confirmed to be functional and tested for the Arm backend:
# Note: Deep AutoEncoder requires manual Linear+BatchNorm1d fusion as the quantizer does not yet support this pattern.
# Note: DS CNN requires AvgPool2d workaround for Ethos-U55 due to stride > 3 limitation.
- Conformer
- Deep AutoEncoder
- Deit Tiny
- DeepLab v3 (DL3)
- DS CNN
- Inception v3 (IC3)
- Llama
- Gemma3n
- Long Short-Term Memory (LSTM)
- MobileNet V1 0.25
- MobileNet v2 (MV2)
- MobileNet v3 (MV3)
- Some popular torch.nn.functional models (NN functional)
- Some popular torch.nn.modules models (NN modules)
- Some popular torch ops (Torch Functions)
- T5 (T5 for conditional generation)
- Neural Super Sampler (NSS)
- Phi-3
- ResNet 18
- ResNet-8
- Wav2Letter (W2L)
- Stable Diffusion:
    * CLIP Text Encoder (CLIP Text with Projection)
    * Stable Diffusion 3 Transformer (SD3 Transformer)
    * T5 Encoder
    * VAE Encoder/Decoder (VAE)
