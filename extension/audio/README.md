# Audio Processing with ExecuTorch

The file `mel_spectrogram.py` contains the class `WhisperAudioProcessor`, a module which converts a mono waveform audio input (as a 1D tensor) into Mel spectrograms. It applies a Short-Time Fourier Transform (via torch.stft) and a Mel filterbank. It is equivalent to the `WhisperFeatureExtractor` class in HuggingFace Transformers, but is implemented in PyTorch instead of NumPy. `WhisperFeatureExtractor` is used for Whisper, Voxtral, Qwen2 audio and Qwen2.5 omni. For example, the output Mel spectrograms can be fed directly into the Whisper model (encoder+decoder) exported from HF Transformers.

Since `WhisperAudioProcessor` is written in PyTorch, we can export it with ExecuTorch and run it on device. The defaults for `WhisperAudioProcessor` are 16kHz audio and 80 Mel spectrogram bins and audio chunks of 30 sec.

Run it as a script

``` python mel_spectrogram.py ```

to export `WhisperFeatureExtractor` (with default constructor arguments) as `whisper_preprocess.pte`, which can run on device (on CPU).
