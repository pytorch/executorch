/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdio.h>

#include "uart_stdout.h"

static int uart_putc(char ch, FILE* stream) {
  (void)stream;
  UartPutc(ch);
  return ch;
}

static FILE stdio_stream =
    FDEV_SETUP_STREAM(uart_putc, NULL, NULL, _FDEV_SETUP_WRITE);

FILE* const stdin = &stdio_stream;
__strong_reference(stdin, stdout);
__strong_reference(stdin, stderr);
