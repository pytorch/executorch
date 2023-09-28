cmake_minimum_required(VERSION 3.21)

cmake_policy(SET CMP0114 NEW)
  
project(external-curl)
include(ExternalProject)

set(CURL_VERSION 7.80.0)
string(REPLACE "." "_" CURL_VERSION_USCORE ${CURL_VERSION})

list(APPEND CMAKE_ARGS
    "-DCMAKE_TOOLCHAIN_FILE:PATH=${CMAKE_TOOLCHAIN_FILE}"
    "-DBUILD_SHARED_LIBS=OFF"
    "-DCMAKE_USE_OPENSSL=OFF"
    "-DBUILD_CURL_EXE=OFF"
    "-DBUILD_TESTING=FALSE"
    "-DHAVE_LIBIDN2=FALSE"
    "-DCURL_CA_PATH=none"
    "-DCURL_DISABLE_FTP=ON"
    "-DCURL_DISABLE_LDAP=ON"
    "-DCURL_DISABLE_LDAPS=ON"
    "-DCURL_DISABLE_TELNET=ON"
    "-DCURL_DISABLE_DICT=ON"
    "-DCURL_DISABLE_FILE=ON"
    "-DCURL_DISABLE_TFTP=ON"
    "-DCURL_DISABLE_RTSP=ON"
    "-DCURL_DISABLE_POP3=ON"
    "-DCURL_DISABLE_IMAP=ON"
    "-DCURL_DISABLE_SMTP=ON"
    "-DCURL_DISABLE_GOPHER=ON"
    "-DUSE_MANUAL=OFF"
)

if(IOS)
    list(APPEND CMAKE_ARGS
        "-DPLATFORM=${PLATFORM}"
    )
endif()

message(STATUS "Preparing external project \"curl\" with args:")
foreach(CMAKE_ARG ${CMAKE_ARGS})
    message(STATUS "-- ${CMAKE_ARG}")
endforeach()

ExternalProject_add(
    curl
    URL https://github.com/curl/curl/releases/download/curl-${CURL_VERSION_USCORE}/curl-${CURL_VERSION}.tar.gz
    PREFIX curl
    CMAKE_ARGS "${CMAKE_ARGS}"
    INSTALL_COMMAND cmake -E echo "Skipping install step."
    )
