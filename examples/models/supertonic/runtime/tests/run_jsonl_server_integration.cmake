cmake_minimum_required(VERSION 3.24)

foreach(required RUNNER PTE ASSET_DIR STYLE WORK_DIR)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "Missing JSONL runner variable: ${required}")
  endif()
endforeach()

if(NOT EXISTS "${PTE}"
   OR NOT EXISTS "${ASSET_DIR}/onnx/unicode_indexer.json"
   OR NOT EXISTS "${STYLE}"
)
  message("SKIP: Supertonic JSONL server assets are unavailable")
  return()
endif()

file(MAKE_DIRECTORY "${WORK_DIR}")
set(output_one "${WORK_DIR}/000001.wav")
set(output_two "${WORK_DIR}/000002.wav")
file(REMOVE "${output_one}" "${output_two}")
set(requests "${WORK_DIR}/requests.jsonl")
file(
  WRITE "${requests}"
  "{\"type\":\"synthesize\",\"id\":1,\"text\":\"First warm request.\",\"output\":\"${output_one}\"}\n"
  "{bad json\n"
  "{\"type\":\"synthesize\",\"id\":1,\"text\":\"Duplicate request.\",\"output\":\"${WORK_DIR}/duplicate.wav\"}\n"
  "{\"type\":\"synthesize\",\"id\":2,\"text\":\"Existing output.\",\"output\":\"${output_one}\"}\n"
  "{\"type\":\"synthesize\",\"id\":3,\"text\":\"Second warm request.\",\"output\":\"${output_two}\"}\n"
  "{\"type\":\"shutdown\"}\n"
)

execute_process(
  COMMAND
    "${RUNNER}" "--pte=${PTE}" "--asset_dir=${ASSET_DIR}"
    "--voice_style=${STYLE}" "--server_jsonl=true" "--warmup_text=Warmup."
    "--language=en" "--speed=1.05" "--seed=42"
  INPUT_FILE "${requests}"
  RESULT_VARIABLE server_result
  OUTPUT_VARIABLE server_output
  ERROR_VARIABLE server_error
)
if(NOT server_result EQUAL 0)
  message(
    FATAL_ERROR
      "Supertonic JSONL runner failed (${server_result})\n${server_output}${server_error}"
  )
endif()

string(REPLACE "\n" ";" response_lines "${server_output}")
list(FILTER response_lines EXCLUDE REGEX "^$")
list(LENGTH response_lines response_count)
if(NOT response_count EQUAL 7)
  message(
    FATAL_ERROR
      "Expected seven JSONL responses, got ${response_count}: ${server_output}"
  )
endif()

set(expected_types
    ready
    result
    error
    error
    error
    result
    stopped
)
foreach(index RANGE 0 6)
  list(GET response_lines ${index} response)
  list(GET expected_types ${index} expected_type)
  string(
    JSON
    response_type
    ERROR_VARIABLE
    json_error
    GET
    "${response}"
    type
  )
  if(json_error OR NOT response_type STREQUAL expected_type)
    message(
      FATAL_ERROR
        "Response ${index} has type '${response_type}', expected '${expected_type}': ${json_error}"
    )
  endif()
endforeach()

list(GET response_lines 0 ready)
string(JSON ready_member_count LENGTH "${ready}")
set(ready_members)
if(ready_member_count GREATER 0)
  math(EXPR ready_last_member "${ready_member_count} - 1")
  foreach(index RANGE 0 ${ready_last_member})
    string(JSON ready_member MEMBER "${ready}" ${index})
    list(APPEND ready_members "${ready_member}")
  endforeach()
endif()
list(SORT ready_members)
set(expected_ready_members load_seconds protocol_version sample_rate type
                           warmup_seconds
)
list(SORT expected_ready_members)
string(JSON protocol_version GET "${ready}" protocol_version)
string(JSON sample_rate GET "${ready}" sample_rate)
string(JSON load_seconds GET "${ready}" load_seconds)
string(JSON warmup_seconds GET "${ready}" warmup_seconds)
if(NOT ready_members STREQUAL expected_ready_members
   OR NOT protocol_version EQUAL 1
   OR NOT sample_rate EQUAL 44100
   OR load_seconds LESS 0
   OR warmup_seconds LESS 0
)
  message(FATAL_ERROR "Malformed JSONL server ready event: ${ready}")
endif()

list(GET response_lines 1 first_result)
list(GET response_lines 2 malformed_error)
list(GET response_lines 3 duplicate_error)
list(GET response_lines 4 existing_error)
list(GET response_lines 5 second_result)
string(JSON first_id GET "${first_result}" id)
string(JSON first_output GET "${first_result}" output)
string(JSON first_samples GET "${first_result}" samples)
string(JSON first_audio_seconds GET "${first_result}" audio_seconds)
string(JSON first_synthesis_seconds GET "${first_result}" synthesis_seconds)
string(JSON first_rtf GET "${first_result}" rtf)
string(JSON second_id GET "${second_result}" id)
string(JSON second_output GET "${second_result}" output)
string(JSON second_samples GET "${second_result}" samples)
string(JSON second_audio_seconds GET "${second_result}" audio_seconds)
string(JSON second_synthesis_seconds GET "${second_result}" synthesis_seconds)
string(JSON second_rtf GET "${second_result}" rtf)
string(JSON malformed_id_type TYPE "${malformed_error}" id)
string(JSON duplicate_id GET "${duplicate_error}" id)
string(JSON duplicate_message GET "${duplicate_error}" message)
string(JSON existing_id GET "${existing_error}" id)
string(JSON existing_message GET "${existing_error}" message)
if(NOT malformed_id_type STREQUAL "NULL"
   OR NOT duplicate_id EQUAL 1
   OR NOT duplicate_message MATCHES "monotonically"
   OR NOT existing_id EQUAL 2
   OR NOT existing_message MATCHES "already exists"
   OR NOT first_id EQUAL 1
   OR NOT second_id EQUAL 3
   OR NOT first_output STREQUAL output_one
   OR NOT second_output STREQUAL output_two
   OR first_samples LESS_EQUAL 0
   OR second_samples LESS_EQUAL 0
   OR first_audio_seconds LESS_EQUAL 0
   OR second_audio_seconds LESS_EQUAL 0
   OR first_synthesis_seconds LESS 0
   OR second_synthesis_seconds LESS 0
   OR first_rtf LESS 0
   OR second_rtf LESS 0
)
  message(FATAL_ERROR "JSONL server result fields are invalid")
endif()

function(validate_wav path)
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR "JSONL server WAV does not exist: ${path}")
  endif()
  file(SIZE "${path}" output_size)
  if(output_size LESS 46)
    message(FATAL_ERROR "JSONL server WAV is too small: ${output_size} bytes")
  endif()
  file(READ "${path}" wav_hex HEX)
  string(TOLOWER "${wav_hex}" wav_hex)
  string(SUBSTRING "${wav_hex}" 0 8 riff_header)
  string(SUBSTRING "${wav_hex}" 16 8 wave_header)
  string(SUBSTRING "${wav_hex}" 44 4 channel_hex)
  string(SUBSTRING "${wav_hex}" 48 8 sample_rate_hex)
  string(SUBSTRING "${wav_hex}" 68 4 bits_hex)
  if(NOT riff_header STREQUAL "52494646"
     OR NOT wave_header STREQUAL "57415645"
     OR NOT channel_hex STREQUAL "0100"
     OR NOT sample_rate_hex STREQUAL "44ac0000"
     OR NOT bits_hex STREQUAL "1000"
  )
    message(FATAL_ERROR "JSONL server WAV format is invalid: ${path}")
  endif()
  string(SUBSTRING "${wav_hex}" 88 -1 pcm_hex)
  if(pcm_hex MATCHES "^0*$")
    message(
      FATAL_ERROR "JSONL server WAV PCM payload is entirely zero: ${path}"
    )
  endif()
endfunction()

validate_wav("${output_one}")
validate_wav("${output_two}")

set(empty_requests "${WORK_DIR}/empty.jsonl")
file(WRITE "${empty_requests}" "")
execute_process(
  COMMAND
    "${RUNNER}" "--pte=${PTE}" "--asset_dir=${ASSET_DIR}"
    "--voice_style=${STYLE}" "--server_jsonl=true" "--warmup_text=Warmup."
    "--language=en" "--speed=1.05" "--seed=42"
  INPUT_FILE "${empty_requests}"
  RESULT_VARIABLE eof_result
  OUTPUT_VARIABLE eof_output
  ERROR_VARIABLE eof_error
)
if(NOT eof_result EQUAL 0)
  message(
    FATAL_ERROR
      "Supertonic JSONL EOF exit failed (${eof_result})\n${eof_output}${eof_error}"
  )
endif()
string(REPLACE "\n" ";" eof_lines "${eof_output}")
list(FILTER eof_lines EXCLUDE REGEX "^$")
list(LENGTH eof_lines eof_count)
if(NOT eof_count EQUAL 1)
  message(FATAL_ERROR "EOF run must emit only ready: ${eof_output}")
endif()
list(GET eof_lines 0 eof_ready)
string(
  JSON
  eof_type
  ERROR_VARIABLE
  eof_json_error
  GET
  "${eof_ready}"
  type
)
if(eof_json_error OR NOT eof_type STREQUAL "ready")
  message(FATAL_ERROR "EOF run did not emit a valid ready record")
endif()

message("${server_output}")
