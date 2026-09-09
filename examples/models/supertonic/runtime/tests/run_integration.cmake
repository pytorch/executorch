foreach(required RUNNER PTE ASSET_DIR STYLE OUTPUT)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "Missing integration variable: ${required}")
  endif()
endforeach()

if(NOT EXISTS "${PTE}"
   OR NOT EXISTS "${ASSET_DIR}/onnx/unicode_indexer.json"
   OR NOT EXISTS "${STYLE}"
)
  message("SKIP: Supertonic integration assets are unavailable")
  return()
endif()

execute_process(
  COMMAND
    "${RUNNER}" "--pte=${PTE}" "--asset_dir=${ASSET_DIR}"
    "--voice_style=${STYLE}" "--text=Hello." "--language=en" "--speed=1.05"
    "--seed=42" "--output=${OUTPUT}"
  RESULT_VARIABLE runner_result
  OUTPUT_VARIABLE runner_output
  ERROR_VARIABLE runner_error
)
if(NOT runner_result EQUAL 0)
  message(
    FATAL_ERROR
      "Supertonic runner failed (${runner_result})\n${runner_output}${runner_error}"
  )
endif()

file(SIZE "${OUTPUT}" output_size)
if(output_size LESS 46)
  message(FATAL_ERROR "Integration WAV is too small: ${output_size} bytes")
endif()
file(READ "${OUTPUT}" wav_hex HEX)
string(TOLOWER "${wav_hex}" wav_hex)
string(SUBSTRING "${wav_hex}" 0 8 riff_header)
string(SUBSTRING "${wav_hex}" 16 8 wave_header)
if(NOT riff_header STREQUAL "52494646" OR NOT wave_header STREQUAL "57415645")
  message(FATAL_ERROR "Integration WAV lacks RIFF/WAVE header")
endif()
string(SUBSTRING "${wav_hex}" 44 4 channel_hex)
string(SUBSTRING "${wav_hex}" 48 8 sample_rate_hex)
string(SUBSTRING "${wav_hex}" 68 4 bits_hex)
if(NOT channel_hex STREQUAL "0100"
   OR NOT sample_rate_hex STREQUAL "44ac0000"
   OR NOT bits_hex STREQUAL "1000"
)
  message(
    FATAL_ERROR
      "Integration WAV is not mono 44.1 kHz PCM16: channels=${channel_hex}, rate=${sample_rate_hex}, bits=${bits_hex}"
  )
endif()
string(SUBSTRING "${wav_hex}" 88 -1 pcm_hex)
if(pcm_hex MATCHES "^0*$")
  message(FATAL_ERROR "Integration WAV PCM payload is entirely zero")
endif()
message("${runner_output}")
