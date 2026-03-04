"""Compare TRT vs eager encoder/decoder outputs on real audio."""

import torch
from executorch.runtime import Runtime
from examples.models.parakeet.export_parakeet_tdt import (
    EncoderWithProjection,
    DecoderStep,
    JointWithArgmax,
    load_audio,
    load_model,
)

PTE_PATH = "/home/dev/models/parakeet_trt_fp32/model.pte"
AUDIO_PATH = "/home/dev/models/parakeet_trt/output30s.wav"

print("Loading NeMo model...")
model = load_model()
model.eval()

print("Loading ExecuTorch program...")
runtime = Runtime.get()
program = runtime.load_program(open(PTE_PATH, "rb").read())


def et_to_torch(val):
    """Convert an ExecuTorch EValue to a PyTorch tensor."""
    try:
        t = val
        return torch.tensor(t.numpy()) if hasattr(t, 'numpy') else torch.tensor(t)
    except Exception:
        return val


with torch.no_grad():
    # --- Preprocessor ---
    audio = load_audio(AUDIO_PATH, sample_rate=16000)
    audio_1d = audio.squeeze(0)
    audio_len = torch.tensor([audio_1d.shape[0]], dtype=torch.int64)

    # Eager preprocessor
    mel_eager, mel_len_eager = model.preprocessor(
        input_signal=audio, length=torch.tensor([audio.shape[1]])
    )

    # ET preprocessor
    preproc_method = program.load_method("preprocessor")
    proc_result = preproc_method.execute([audio_1d, audio_len])
    mel_et = proc_result[0]
    mel_len_et = proc_result[1].item()

    print(f"\n=== Preprocessor ===")
    print(f"Eager mel shape: {mel_eager.shape}, len: {mel_len_eager.item()}")
    print(f"ET mel shape: {tuple(mel_et.shape)}, len: {mel_len_et}")
    mel_et_torch = et_to_torch(mel_et)
    diff = (mel_eager - mel_et_torch).abs()
    print(f"Mel diff: max={diff.max():.6f}, mean={diff.mean():.6f}")

    # --- Encoder ---
    encoder_with_proj = EncoderWithProjection(model.encoder, model.joint)
    encoder_with_proj.eval()
    f_proj_eager, enc_len_eager = encoder_with_proj(
        audio_signal=mel_eager, length=mel_len_eager
    )

    encoder_method = program.load_method("encoder")
    mel_len_tensor = torch.tensor([mel_len_et], dtype=torch.int64)
    # Pass the eager mel to the ET encoder so we isolate encoder differences
    enc_result = encoder_method.execute([mel_eager, mel_len_tensor])

    print(f"\n=== Encoder ===")
    print(f"Encoder returned {len(enc_result)} outputs")
    for i, val in enumerate(enc_result):
        print(f"  output[{i}]: type={type(val).__name__}, ", end="")
        try:
            print(f"shape={tuple(val.shape)}, dtype={val.dtype}")
        except Exception:
            try:
                print(f"value={val.item()}")
            except Exception:
                print(f"repr={val}")

    f_proj_et = et_to_torch(enc_result[0])
    enc_len_et = enc_result[1].item()

    print(f"Eager f_proj shape: {f_proj_eager.shape}, len: {enc_len_eager.item()}")
    print(f"ET f_proj shape: {tuple(f_proj_et.shape)}, len: {enc_len_et}")

    # Compare only valid frames
    valid = min(int(enc_len_eager.item()), enc_len_et)
    diff = (f_proj_eager[:, :valid, :] - f_proj_et[:, :valid, :]).abs()
    print(f"f_proj diff (valid frames): max={diff.max():.6f}, mean={diff.mean():.6f}")
    print(f"Eager f_proj[0,:3]: {f_proj_eager[0, 0, :3].tolist()}")
    print(f"ET    f_proj[0,:3]: {f_proj_et[0, 0, :3].tolist()}")

    # --- Decoder SOS step ---
    num_layers = model.decoder.pred_rnn_layers
    pred_hidden = model.decoder.pred_hidden
    blank_id = model.tokenizer.vocab_size

    decoder_step_eager = DecoderStep(model.decoder, model.joint)
    decoder_step_eager.eval()

    sos = torch.tensor([[blank_id]], dtype=torch.long)
    h0 = torch.zeros(num_layers, 1, pred_hidden)
    c0 = torch.zeros(num_layers, 1, pred_hidden)

    g_eager, h_eager, c_eager = decoder_step_eager(sos, h0, c0)

    decoder_method = program.load_method("decoder_step")
    dec_result = decoder_method.execute([sos, h0, c0])
    g_et = et_to_torch(dec_result[0])
    h_et = et_to_torch(dec_result[1])
    c_et = et_to_torch(dec_result[2])

    print(f"\n=== Decoder SOS step ===")
    g_diff = (g_eager - g_et).abs()
    print(f"g_proj diff: max={g_diff.max():.6f}, mean={g_diff.mean():.6f}")
    print(f"Eager g_proj[:3]: {g_eager[0, 0, :3].tolist()}")
    print(f"ET    g_proj[:3]: {g_et[0, 0, :3].tolist()}")

    # --- Joint with EAGER encoder+decoder outputs (isolate joint) ---
    num_token_classes = blank_id + 1
    joint_eager_mod = JointWithArgmax(model.joint, num_token_classes)
    joint_eager_mod.eval()

    joint_method = program.load_method("joint")

    print(f"\n=== Joint on EAGER inputs (first 10 frames) ===")
    for t in range(min(10, valid)):
        f_t = f_proj_eager[:, t:t+1, :].contiguous()
        eager_tok, eager_dur = joint_eager_mod(f_t, g_eager)
        et_result = joint_method.execute([f_t, g_eager])
        et_tok = et_result[0].item()
        et_dur = et_result[1].item()
        match = "OK" if eager_tok.item() == et_tok else "MISMATCH"
        print(f"  t={t}: eager_tok={eager_tok.item()}, et_tok={et_tok} [{match}], eager_dur={eager_dur.item()}, et_dur={et_dur}")

    # --- Joint with TRT encoder + TRT decoder outputs ---
    print(f"\n=== Joint on TRT inputs (first 10 frames) ===")
    for t in range(min(10, valid)):
        f_t_et = f_proj_et[:, t:t+1, :].contiguous()
        # eager joint with TRT inputs
        eager_tok, eager_dur = joint_eager_mod(f_t_et, g_et)
        # TRT joint with TRT inputs
        et_result = joint_method.execute([f_t_et, g_et])
        et_tok = et_result[0].item()
        et_dur = et_result[1].item()
        match = "OK" if eager_tok.item() == et_tok else "MISMATCH"
        print(f"  t={t}: eager_tok={eager_tok.item()}, et_tok={et_tok} [{match}]")

    # --- Full eager decode to verify model works ---
    print(f"\n=== Full eager greedy decode ===")
    from examples.models.parakeet.export_parakeet_tdt import greedy_decode_eager
    encoded, encoded_len = model.encoder(audio_signal=mel_eager, length=mel_len_eager)
    tokens = greedy_decode_eager(encoded, encoded_len, model)
    text = model.tokenizer.ids_to_text(tokens)
    print(f"Tokens: {len(tokens)}")
    print(f"Text: {text}")