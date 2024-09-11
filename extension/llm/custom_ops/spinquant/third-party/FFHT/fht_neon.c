// @generated
#include "fht.h"
static inline void helper_float_1(float* buf);
static inline void helper_float_1(float* buf) {
  for (int j = 0; j < 2; j += 2) {
    for (int k = 0; k < 1; ++k) {
      float u = buf[j + k];
      float v = buf[j + k + 1];
      buf[j + k] = u + v;
      buf[j + k + 1] = u - v;
    }
  }
}
static inline void helper_float_2(float* buf);
static inline void helper_float_2(float* buf) {
  for (int j = 0; j < 4; j += 4) {
    __asm__ volatile(
        "LD1 {v0.4S}, [%0]\n"
        "TRN1 v16.4S, v0.4S, v0.4S\n"
        "FNEG v17.4S, v0.4S\n"
        "TRN2 v17.4S, v0.4S, v17.4S\n"
        "FADD v0.4S, v16.4S, v17.4S\n"
        "DUP v16.2D, v0.D[0]\n"
        "FNEG v17.4S, v0.4S\n"
        "INS v17.D[0], v0.D[1]\n"
        "FADD v0.4S, v16.4S, v17.4S\n"
        "ST1 {v0.4S}, [%0]\n" ::"r"(buf + j)
        : "%v0",
          "%v1",
          "%v2",
          "%v3",
          "%v4",
          "%v5",
          "%v6",
          "%v7",
          "%v8",
          "%v9",
          "%v10",
          "%v11",
          "%v12",
          "%v13",
          "%v14",
          "%v15",
          "%v16",
          "%v17",
          "%v18",
          "%v19",
          "%v20",
          "%v21",
          "%v22",
          "%v23",
          "%v24",
          "%v25",
          "%v26",
          "%v27",
          "%v28",
          "%v29",
          "%v30",
          "%v31",
          "memory");
  }
}
void helper_float_3_recursive(float* buf, int depth);
void helper_float_3_recursive(float* buf, int depth) {
  if (depth == 2) {
    helper_float_2(buf);
    return;
  }
  if (depth == 3) {
    helper_float_3_recursive(buf + 0, 2);
    helper_float_3_recursive(buf + 4, 2);
    for (int j = 0; j < 8; j += 8) {
      for (int k = 0; k < 4; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "ST1 {v16.4S}, [%0]\n"
            "ST1 {v17.4S}, [%1]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 4)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_3(float* buf);
void helper_float_3(float* buf) {
  helper_float_3_recursive(buf, 3);
}
void helper_float_4_recursive(float* buf, int depth);
void helper_float_4_recursive(float* buf, int depth) {
  if (depth == 3) {
    helper_float_3(buf);
    return;
  }
  if (depth == 4) {
    helper_float_4_recursive(buf + 0, 3);
    helper_float_4_recursive(buf + 8, 3);
    for (int j = 0; j < 16; j += 16) {
      for (int k = 0; k < 8; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "ST1 {v16.4S}, [%0]\n"
            "ST1 {v17.4S}, [%1]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 8)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_4(float* buf);
void helper_float_4(float* buf) {
  helper_float_4_recursive(buf, 4);
}
void helper_float_5_recursive(float* buf, int depth);
void helper_float_5_recursive(float* buf, int depth) {
  if (depth == 4) {
    helper_float_4(buf);
    return;
  }
  if (depth == 5) {
    helper_float_5_recursive(buf + 0, 4);
    helper_float_5_recursive(buf + 16, 4);
    for (int j = 0; j < 32; j += 32) {
      for (int k = 0; k < 16; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "ST1 {v16.4S}, [%0]\n"
            "ST1 {v17.4S}, [%1]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 16)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_5(float* buf);
void helper_float_5(float* buf) {
  helper_float_5_recursive(buf, 5);
}
void helper_float_6_recursive(float* buf, int depth);
void helper_float_6_recursive(float* buf, int depth) {
  if (depth == 3) {
    helper_float_3(buf);
    return;
  }
  if (depth == 6) {
    helper_float_6_recursive(buf + 0, 3);
    helper_float_6_recursive(buf + 8, 3);
    helper_float_6_recursive(buf + 16, 3);
    helper_float_6_recursive(buf + 24, 3);
    helper_float_6_recursive(buf + 32, 3);
    helper_float_6_recursive(buf + 40, 3);
    helper_float_6_recursive(buf + 48, 3);
    helper_float_6_recursive(buf + 56, 3);
    for (int j = 0; j < 64; j += 64) {
      for (int k = 0; k < 8; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "LD1 {v2.4S}, [%2]\n"
            "LD1 {v3.4S}, [%3]\n"
            "LD1 {v4.4S}, [%4]\n"
            "LD1 {v5.4S}, [%5]\n"
            "LD1 {v6.4S}, [%6]\n"
            "LD1 {v7.4S}, [%7]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "FADD v18.4S, v2.4S, v3.4S\n"
            "FSUB v19.4S, v2.4S, v3.4S\n"
            "FADD v20.4S, v4.4S, v5.4S\n"
            "FSUB v21.4S, v4.4S, v5.4S\n"
            "FADD v22.4S, v6.4S, v7.4S\n"
            "FSUB v23.4S, v6.4S, v7.4S\n"
            "FADD v0.4S, v16.4S, v18.4S\n"
            "FSUB v2.4S, v16.4S, v18.4S\n"
            "FADD v1.4S, v17.4S, v19.4S\n"
            "FSUB v3.4S, v17.4S, v19.4S\n"
            "FADD v4.4S, v20.4S, v22.4S\n"
            "FSUB v6.4S, v20.4S, v22.4S\n"
            "FADD v5.4S, v21.4S, v23.4S\n"
            "FSUB v7.4S, v21.4S, v23.4S\n"
            "FADD v16.4S, v0.4S, v4.4S\n"
            "FSUB v20.4S, v0.4S, v4.4S\n"
            "FADD v17.4S, v1.4S, v5.4S\n"
            "FSUB v21.4S, v1.4S, v5.4S\n"
            "FADD v18.4S, v2.4S, v6.4S\n"
            "FSUB v22.4S, v2.4S, v6.4S\n"
            "FADD v19.4S, v3.4S, v7.4S\n"
            "FSUB v23.4S, v3.4S, v7.4S\n"
            "ST1 {v16.4S}, [%0]\n"
            "ST1 {v17.4S}, [%1]\n"
            "ST1 {v18.4S}, [%2]\n"
            "ST1 {v19.4S}, [%3]\n"
            "ST1 {v20.4S}, [%4]\n"
            "ST1 {v21.4S}, [%5]\n"
            "ST1 {v22.4S}, [%6]\n"
            "ST1 {v23.4S}, [%7]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 8),
            "r"(buf + j + k + 16),
            "r"(buf + j + k + 24),
            "r"(buf + j + k + 32),
            "r"(buf + j + k + 40),
            "r"(buf + j + k + 48),
            "r"(buf + j + k + 56)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_6(float* buf);
void helper_float_6(float* buf) {
  helper_float_6_recursive(buf, 6);
}
void helper_float_7_recursive(float* buf, int depth);
void helper_float_7_recursive(float* buf, int depth) {
  if (depth == 3) {
    helper_float_3(buf);
    return;
  }
  if (depth == 7) {
    helper_float_7_recursive(buf + 0, 3);
    helper_float_7_recursive(buf + 8, 3);
    helper_float_7_recursive(buf + 16, 3);
    helper_float_7_recursive(buf + 24, 3);
    helper_float_7_recursive(buf + 32, 3);
    helper_float_7_recursive(buf + 40, 3);
    helper_float_7_recursive(buf + 48, 3);
    helper_float_7_recursive(buf + 56, 3);
    helper_float_7_recursive(buf + 64, 3);
    helper_float_7_recursive(buf + 72, 3);
    helper_float_7_recursive(buf + 80, 3);
    helper_float_7_recursive(buf + 88, 3);
    helper_float_7_recursive(buf + 96, 3);
    helper_float_7_recursive(buf + 104, 3);
    helper_float_7_recursive(buf + 112, 3);
    helper_float_7_recursive(buf + 120, 3);
    for (int j = 0; j < 128; j += 128) {
      for (int k = 0; k < 8; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "LD1 {v2.4S}, [%2]\n"
            "LD1 {v3.4S}, [%3]\n"
            "LD1 {v4.4S}, [%4]\n"
            "LD1 {v5.4S}, [%5]\n"
            "LD1 {v6.4S}, [%6]\n"
            "LD1 {v7.4S}, [%7]\n"
            "LD1 {v8.4S}, [%8]\n"
            "LD1 {v9.4S}, [%9]\n"
            "LD1 {v10.4S}, [%10]\n"
            "LD1 {v11.4S}, [%11]\n"
            "LD1 {v12.4S}, [%12]\n"
            "LD1 {v13.4S}, [%13]\n"
            "LD1 {v14.4S}, [%14]\n"
            "LD1 {v15.4S}, [%15]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "FADD v18.4S, v2.4S, v3.4S\n"
            "FSUB v19.4S, v2.4S, v3.4S\n"
            "FADD v20.4S, v4.4S, v5.4S\n"
            "FSUB v21.4S, v4.4S, v5.4S\n"
            "FADD v22.4S, v6.4S, v7.4S\n"
            "FSUB v23.4S, v6.4S, v7.4S\n"
            "FADD v24.4S, v8.4S, v9.4S\n"
            "FSUB v25.4S, v8.4S, v9.4S\n"
            "FADD v26.4S, v10.4S, v11.4S\n"
            "FSUB v27.4S, v10.4S, v11.4S\n"
            "FADD v28.4S, v12.4S, v13.4S\n"
            "FSUB v29.4S, v12.4S, v13.4S\n"
            "FADD v30.4S, v14.4S, v15.4S\n"
            "FSUB v31.4S, v14.4S, v15.4S\n"
            "FADD v0.4S, v16.4S, v18.4S\n"
            "FSUB v2.4S, v16.4S, v18.4S\n"
            "FADD v1.4S, v17.4S, v19.4S\n"
            "FSUB v3.4S, v17.4S, v19.4S\n"
            "FADD v4.4S, v20.4S, v22.4S\n"
            "FSUB v6.4S, v20.4S, v22.4S\n"
            "FADD v5.4S, v21.4S, v23.4S\n"
            "FSUB v7.4S, v21.4S, v23.4S\n"
            "FADD v8.4S, v24.4S, v26.4S\n"
            "FSUB v10.4S, v24.4S, v26.4S\n"
            "FADD v9.4S, v25.4S, v27.4S\n"
            "FSUB v11.4S, v25.4S, v27.4S\n"
            "FADD v12.4S, v28.4S, v30.4S\n"
            "FSUB v14.4S, v28.4S, v30.4S\n"
            "FADD v13.4S, v29.4S, v31.4S\n"
            "FSUB v15.4S, v29.4S, v31.4S\n"
            "FADD v16.4S, v0.4S, v4.4S\n"
            "FSUB v20.4S, v0.4S, v4.4S\n"
            "FADD v17.4S, v1.4S, v5.4S\n"
            "FSUB v21.4S, v1.4S, v5.4S\n"
            "FADD v18.4S, v2.4S, v6.4S\n"
            "FSUB v22.4S, v2.4S, v6.4S\n"
            "FADD v19.4S, v3.4S, v7.4S\n"
            "FSUB v23.4S, v3.4S, v7.4S\n"
            "FADD v24.4S, v8.4S, v12.4S\n"
            "FSUB v28.4S, v8.4S, v12.4S\n"
            "FADD v25.4S, v9.4S, v13.4S\n"
            "FSUB v29.4S, v9.4S, v13.4S\n"
            "FADD v26.4S, v10.4S, v14.4S\n"
            "FSUB v30.4S, v10.4S, v14.4S\n"
            "FADD v27.4S, v11.4S, v15.4S\n"
            "FSUB v31.4S, v11.4S, v15.4S\n"
            "FADD v0.4S, v16.4S, v24.4S\n"
            "FSUB v8.4S, v16.4S, v24.4S\n"
            "FADD v1.4S, v17.4S, v25.4S\n"
            "FSUB v9.4S, v17.4S, v25.4S\n"
            "FADD v2.4S, v18.4S, v26.4S\n"
            "FSUB v10.4S, v18.4S, v26.4S\n"
            "FADD v3.4S, v19.4S, v27.4S\n"
            "FSUB v11.4S, v19.4S, v27.4S\n"
            "FADD v4.4S, v20.4S, v28.4S\n"
            "FSUB v12.4S, v20.4S, v28.4S\n"
            "FADD v5.4S, v21.4S, v29.4S\n"
            "FSUB v13.4S, v21.4S, v29.4S\n"
            "FADD v6.4S, v22.4S, v30.4S\n"
            "FSUB v14.4S, v22.4S, v30.4S\n"
            "FADD v7.4S, v23.4S, v31.4S\n"
            "FSUB v15.4S, v23.4S, v31.4S\n"
            "ST1 {v0.4S}, [%0]\n"
            "ST1 {v1.4S}, [%1]\n"
            "ST1 {v2.4S}, [%2]\n"
            "ST1 {v3.4S}, [%3]\n"
            "ST1 {v4.4S}, [%4]\n"
            "ST1 {v5.4S}, [%5]\n"
            "ST1 {v6.4S}, [%6]\n"
            "ST1 {v7.4S}, [%7]\n"
            "ST1 {v8.4S}, [%8]\n"
            "ST1 {v9.4S}, [%9]\n"
            "ST1 {v10.4S}, [%10]\n"
            "ST1 {v11.4S}, [%11]\n"
            "ST1 {v12.4S}, [%12]\n"
            "ST1 {v13.4S}, [%13]\n"
            "ST1 {v14.4S}, [%14]\n"
            "ST1 {v15.4S}, [%15]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 8),
            "r"(buf + j + k + 16),
            "r"(buf + j + k + 24),
            "r"(buf + j + k + 32),
            "r"(buf + j + k + 40),
            "r"(buf + j + k + 48),
            "r"(buf + j + k + 56),
            "r"(buf + j + k + 64),
            "r"(buf + j + k + 72),
            "r"(buf + j + k + 80),
            "r"(buf + j + k + 88),
            "r"(buf + j + k + 96),
            "r"(buf + j + k + 104),
            "r"(buf + j + k + 112),
            "r"(buf + j + k + 120)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_7(float* buf);
void helper_float_7(float* buf) {
  helper_float_7_recursive(buf, 7);
}
static inline void helper_float_8(float* buf);
static inline void helper_float_8(float* buf) {
  for (int j = 0; j < 256; j += 64) {
    for (int k = 0; k < 4; k += 4) {
      __asm__ volatile(
          "LD1 {v0.4S}, [%0]\n"
          "LD1 {v1.4S}, [%1]\n"
          "LD1 {v2.4S}, [%2]\n"
          "LD1 {v3.4S}, [%3]\n"
          "LD1 {v4.4S}, [%4]\n"
          "LD1 {v5.4S}, [%5]\n"
          "LD1 {v6.4S}, [%6]\n"
          "LD1 {v7.4S}, [%7]\n"
          "LD1 {v8.4S}, [%8]\n"
          "LD1 {v9.4S}, [%9]\n"
          "LD1 {v10.4S}, [%10]\n"
          "LD1 {v11.4S}, [%11]\n"
          "LD1 {v12.4S}, [%12]\n"
          "LD1 {v13.4S}, [%13]\n"
          "LD1 {v14.4S}, [%14]\n"
          "LD1 {v15.4S}, [%15]\n"
          "TRN1 v16.4S, v0.4S, v0.4S\n"
          "FNEG v17.4S, v0.4S\n"
          "TRN2 v17.4S, v0.4S, v17.4S\n"
          "FADD v0.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v1.4S, v1.4S\n"
          "FNEG v17.4S, v1.4S\n"
          "TRN2 v17.4S, v1.4S, v17.4S\n"
          "FADD v1.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v2.4S, v2.4S\n"
          "FNEG v17.4S, v2.4S\n"
          "TRN2 v17.4S, v2.4S, v17.4S\n"
          "FADD v2.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v3.4S, v3.4S\n"
          "FNEG v17.4S, v3.4S\n"
          "TRN2 v17.4S, v3.4S, v17.4S\n"
          "FADD v3.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v4.4S, v4.4S\n"
          "FNEG v17.4S, v4.4S\n"
          "TRN2 v17.4S, v4.4S, v17.4S\n"
          "FADD v4.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v5.4S, v5.4S\n"
          "FNEG v17.4S, v5.4S\n"
          "TRN2 v17.4S, v5.4S, v17.4S\n"
          "FADD v5.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v6.4S, v6.4S\n"
          "FNEG v17.4S, v6.4S\n"
          "TRN2 v17.4S, v6.4S, v17.4S\n"
          "FADD v6.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v7.4S, v7.4S\n"
          "FNEG v17.4S, v7.4S\n"
          "TRN2 v17.4S, v7.4S, v17.4S\n"
          "FADD v7.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v8.4S, v8.4S\n"
          "FNEG v17.4S, v8.4S\n"
          "TRN2 v17.4S, v8.4S, v17.4S\n"
          "FADD v8.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v9.4S, v9.4S\n"
          "FNEG v17.4S, v9.4S\n"
          "TRN2 v17.4S, v9.4S, v17.4S\n"
          "FADD v9.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v10.4S, v10.4S\n"
          "FNEG v17.4S, v10.4S\n"
          "TRN2 v17.4S, v10.4S, v17.4S\n"
          "FADD v10.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v11.4S, v11.4S\n"
          "FNEG v17.4S, v11.4S\n"
          "TRN2 v17.4S, v11.4S, v17.4S\n"
          "FADD v11.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v12.4S, v12.4S\n"
          "FNEG v17.4S, v12.4S\n"
          "TRN2 v17.4S, v12.4S, v17.4S\n"
          "FADD v12.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v13.4S, v13.4S\n"
          "FNEG v17.4S, v13.4S\n"
          "TRN2 v17.4S, v13.4S, v17.4S\n"
          "FADD v13.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v14.4S, v14.4S\n"
          "FNEG v17.4S, v14.4S\n"
          "TRN2 v17.4S, v14.4S, v17.4S\n"
          "FADD v14.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v15.4S, v15.4S\n"
          "FNEG v17.4S, v15.4S\n"
          "TRN2 v17.4S, v15.4S, v17.4S\n"
          "FADD v15.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v0.D[0]\n"
          "FNEG v17.4S, v0.4S\n"
          "INS v17.D[0], v0.D[1]\n"
          "FADD v0.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v1.D[0]\n"
          "FNEG v17.4S, v1.4S\n"
          "INS v17.D[0], v1.D[1]\n"
          "FADD v1.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v2.D[0]\n"
          "FNEG v17.4S, v2.4S\n"
          "INS v17.D[0], v2.D[1]\n"
          "FADD v2.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v3.D[0]\n"
          "FNEG v17.4S, v3.4S\n"
          "INS v17.D[0], v3.D[1]\n"
          "FADD v3.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v4.D[0]\n"
          "FNEG v17.4S, v4.4S\n"
          "INS v17.D[0], v4.D[1]\n"
          "FADD v4.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v5.D[0]\n"
          "FNEG v17.4S, v5.4S\n"
          "INS v17.D[0], v5.D[1]\n"
          "FADD v5.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v6.D[0]\n"
          "FNEG v17.4S, v6.4S\n"
          "INS v17.D[0], v6.D[1]\n"
          "FADD v6.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v7.D[0]\n"
          "FNEG v17.4S, v7.4S\n"
          "INS v17.D[0], v7.D[1]\n"
          "FADD v7.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v8.D[0]\n"
          "FNEG v17.4S, v8.4S\n"
          "INS v17.D[0], v8.D[1]\n"
          "FADD v8.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v9.D[0]\n"
          "FNEG v17.4S, v9.4S\n"
          "INS v17.D[0], v9.D[1]\n"
          "FADD v9.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v10.D[0]\n"
          "FNEG v17.4S, v10.4S\n"
          "INS v17.D[0], v10.D[1]\n"
          "FADD v10.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v11.D[0]\n"
          "FNEG v17.4S, v11.4S\n"
          "INS v17.D[0], v11.D[1]\n"
          "FADD v11.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v12.D[0]\n"
          "FNEG v17.4S, v12.4S\n"
          "INS v17.D[0], v12.D[1]\n"
          "FADD v12.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v13.D[0]\n"
          "FNEG v17.4S, v13.4S\n"
          "INS v17.D[0], v13.D[1]\n"
          "FADD v13.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v14.D[0]\n"
          "FNEG v17.4S, v14.4S\n"
          "INS v17.D[0], v14.D[1]\n"
          "FADD v14.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v15.D[0]\n"
          "FNEG v17.4S, v15.4S\n"
          "INS v17.D[0], v15.D[1]\n"
          "FADD v15.4S, v16.4S, v17.4S\n"
          "FADD v16.4S, v0.4S, v1.4S\n"
          "FSUB v17.4S, v0.4S, v1.4S\n"
          "FADD v18.4S, v2.4S, v3.4S\n"
          "FSUB v19.4S, v2.4S, v3.4S\n"
          "FADD v20.4S, v4.4S, v5.4S\n"
          "FSUB v21.4S, v4.4S, v5.4S\n"
          "FADD v22.4S, v6.4S, v7.4S\n"
          "FSUB v23.4S, v6.4S, v7.4S\n"
          "FADD v24.4S, v8.4S, v9.4S\n"
          "FSUB v25.4S, v8.4S, v9.4S\n"
          "FADD v26.4S, v10.4S, v11.4S\n"
          "FSUB v27.4S, v10.4S, v11.4S\n"
          "FADD v28.4S, v12.4S, v13.4S\n"
          "FSUB v29.4S, v12.4S, v13.4S\n"
          "FADD v30.4S, v14.4S, v15.4S\n"
          "FSUB v31.4S, v14.4S, v15.4S\n"
          "FADD v0.4S, v16.4S, v18.4S\n"
          "FSUB v2.4S, v16.4S, v18.4S\n"
          "FADD v1.4S, v17.4S, v19.4S\n"
          "FSUB v3.4S, v17.4S, v19.4S\n"
          "FADD v4.4S, v20.4S, v22.4S\n"
          "FSUB v6.4S, v20.4S, v22.4S\n"
          "FADD v5.4S, v21.4S, v23.4S\n"
          "FSUB v7.4S, v21.4S, v23.4S\n"
          "FADD v8.4S, v24.4S, v26.4S\n"
          "FSUB v10.4S, v24.4S, v26.4S\n"
          "FADD v9.4S, v25.4S, v27.4S\n"
          "FSUB v11.4S, v25.4S, v27.4S\n"
          "FADD v12.4S, v28.4S, v30.4S\n"
          "FSUB v14.4S, v28.4S, v30.4S\n"
          "FADD v13.4S, v29.4S, v31.4S\n"
          "FSUB v15.4S, v29.4S, v31.4S\n"
          "FADD v16.4S, v0.4S, v4.4S\n"
          "FSUB v20.4S, v0.4S, v4.4S\n"
          "FADD v17.4S, v1.4S, v5.4S\n"
          "FSUB v21.4S, v1.4S, v5.4S\n"
          "FADD v18.4S, v2.4S, v6.4S\n"
          "FSUB v22.4S, v2.4S, v6.4S\n"
          "FADD v19.4S, v3.4S, v7.4S\n"
          "FSUB v23.4S, v3.4S, v7.4S\n"
          "FADD v24.4S, v8.4S, v12.4S\n"
          "FSUB v28.4S, v8.4S, v12.4S\n"
          "FADD v25.4S, v9.4S, v13.4S\n"
          "FSUB v29.4S, v9.4S, v13.4S\n"
          "FADD v26.4S, v10.4S, v14.4S\n"
          "FSUB v30.4S, v10.4S, v14.4S\n"
          "FADD v27.4S, v11.4S, v15.4S\n"
          "FSUB v31.4S, v11.4S, v15.4S\n"
          "FADD v0.4S, v16.4S, v24.4S\n"
          "FSUB v8.4S, v16.4S, v24.4S\n"
          "FADD v1.4S, v17.4S, v25.4S\n"
          "FSUB v9.4S, v17.4S, v25.4S\n"
          "FADD v2.4S, v18.4S, v26.4S\n"
          "FSUB v10.4S, v18.4S, v26.4S\n"
          "FADD v3.4S, v19.4S, v27.4S\n"
          "FSUB v11.4S, v19.4S, v27.4S\n"
          "FADD v4.4S, v20.4S, v28.4S\n"
          "FSUB v12.4S, v20.4S, v28.4S\n"
          "FADD v5.4S, v21.4S, v29.4S\n"
          "FSUB v13.4S, v21.4S, v29.4S\n"
          "FADD v6.4S, v22.4S, v30.4S\n"
          "FSUB v14.4S, v22.4S, v30.4S\n"
          "FADD v7.4S, v23.4S, v31.4S\n"
          "FSUB v15.4S, v23.4S, v31.4S\n"
          "ST1 {v0.4S}, [%0]\n"
          "ST1 {v1.4S}, [%1]\n"
          "ST1 {v2.4S}, [%2]\n"
          "ST1 {v3.4S}, [%3]\n"
          "ST1 {v4.4S}, [%4]\n"
          "ST1 {v5.4S}, [%5]\n"
          "ST1 {v6.4S}, [%6]\n"
          "ST1 {v7.4S}, [%7]\n"
          "ST1 {v8.4S}, [%8]\n"
          "ST1 {v9.4S}, [%9]\n"
          "ST1 {v10.4S}, [%10]\n"
          "ST1 {v11.4S}, [%11]\n"
          "ST1 {v12.4S}, [%12]\n"
          "ST1 {v13.4S}, [%13]\n"
          "ST1 {v14.4S}, [%14]\n"
          "ST1 {v15.4S}, [%15]\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 4),
          "r"(buf + j + k + 8),
          "r"(buf + j + k + 12),
          "r"(buf + j + k + 16),
          "r"(buf + j + k + 20),
          "r"(buf + j + k + 24),
          "r"(buf + j + k + 28),
          "r"(buf + j + k + 32),
          "r"(buf + j + k + 36),
          "r"(buf + j + k + 40),
          "r"(buf + j + k + 44),
          "r"(buf + j + k + 48),
          "r"(buf + j + k + 52),
          "r"(buf + j + k + 56),
          "r"(buf + j + k + 60)
          : "%v0",
            "%v1",
            "%v2",
            "%v3",
            "%v4",
            "%v5",
            "%v6",
            "%v7",
            "%v8",
            "%v9",
            "%v10",
            "%v11",
            "%v12",
            "%v13",
            "%v14",
            "%v15",
            "%v16",
            "%v17",
            "%v18",
            "%v19",
            "%v20",
            "%v21",
            "%v22",
            "%v23",
            "%v24",
            "%v25",
            "%v26",
            "%v27",
            "%v28",
            "%v29",
            "%v30",
            "%v31",
            "memory");
    }
  }
  for (int j = 0; j < 256; j += 256) {
    for (int k = 0; k < 64; k += 4) {
      __asm__ volatile(
          "LD1 {v0.4S}, [%0]\n"
          "LD1 {v1.4S}, [%1]\n"
          "LD1 {v2.4S}, [%2]\n"
          "LD1 {v3.4S}, [%3]\n"
          "FADD v16.4S, v0.4S, v1.4S\n"
          "FSUB v17.4S, v0.4S, v1.4S\n"
          "FADD v18.4S, v2.4S, v3.4S\n"
          "FSUB v19.4S, v2.4S, v3.4S\n"
          "FADD v0.4S, v16.4S, v18.4S\n"
          "FSUB v2.4S, v16.4S, v18.4S\n"
          "FADD v1.4S, v17.4S, v19.4S\n"
          "FSUB v3.4S, v17.4S, v19.4S\n"
          "ST1 {v0.4S}, [%0]\n"
          "ST1 {v1.4S}, [%1]\n"
          "ST1 {v2.4S}, [%2]\n"
          "ST1 {v3.4S}, [%3]\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 64),
          "r"(buf + j + k + 128),
          "r"(buf + j + k + 192)
          : "%v0",
            "%v1",
            "%v2",
            "%v3",
            "%v4",
            "%v5",
            "%v6",
            "%v7",
            "%v8",
            "%v9",
            "%v10",
            "%v11",
            "%v12",
            "%v13",
            "%v14",
            "%v15",
            "%v16",
            "%v17",
            "%v18",
            "%v19",
            "%v20",
            "%v21",
            "%v22",
            "%v23",
            "%v24",
            "%v25",
            "%v26",
            "%v27",
            "%v28",
            "%v29",
            "%v30",
            "%v31",
            "memory");
    }
  }
}
void helper_float_9_recursive(float* buf, int depth);
void helper_float_9_recursive(float* buf, int depth) {
  if (depth == 8) {
    helper_float_8(buf);
    return;
  }
  if (depth == 9) {
    helper_float_9_recursive(buf + 0, 8);
    helper_float_9_recursive(buf + 256, 8);
    for (int j = 0; j < 512; j += 512) {
      for (int k = 0; k < 256; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "ST1 {v16.4S}, [%0]\n"
            "ST1 {v17.4S}, [%1]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 256)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_9(float* buf);
void helper_float_9(float* buf) {
  helper_float_9_recursive(buf, 9);
}
void helper_float_10_recursive(float* buf, int depth);
void helper_float_10_recursive(float* buf, int depth) {
  if (depth == 8) {
    helper_float_8(buf);
    return;
  }
  if (depth == 10) {
    helper_float_10_recursive(buf + 0, 8);
    helper_float_10_recursive(buf + 256, 8);
    helper_float_10_recursive(buf + 512, 8);
    helper_float_10_recursive(buf + 768, 8);
    for (int j = 0; j < 1024; j += 1024) {
      for (int k = 0; k < 256; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "LD1 {v2.4S}, [%2]\n"
            "LD1 {v3.4S}, [%3]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "FADD v18.4S, v2.4S, v3.4S\n"
            "FSUB v19.4S, v2.4S, v3.4S\n"
            "FADD v0.4S, v16.4S, v18.4S\n"
            "FSUB v2.4S, v16.4S, v18.4S\n"
            "FADD v1.4S, v17.4S, v19.4S\n"
            "FSUB v3.4S, v17.4S, v19.4S\n"
            "ST1 {v0.4S}, [%0]\n"
            "ST1 {v1.4S}, [%1]\n"
            "ST1 {v2.4S}, [%2]\n"
            "ST1 {v3.4S}, [%3]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 256),
            "r"(buf + j + k + 512),
            "r"(buf + j + k + 768)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_10(float* buf);
void helper_float_10(float* buf) {
  helper_float_10_recursive(buf, 10);
}
void helper_float_11_recursive(float* buf, int depth);
void helper_float_11_recursive(float* buf, int depth) {
  if (depth == 10) {
    helper_float_10(buf);
    return;
  }
  if (depth == 11) {
    helper_float_11_recursive(buf + 0, 10);
    helper_float_11_recursive(buf + 1024, 10);
    for (int j = 0; j < 2048; j += 2048) {
      for (int k = 0; k < 1024; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "ST1 {v16.4S}, [%0]\n"
            "ST1 {v17.4S}, [%1]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 1024)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_11(float* buf);
void helper_float_11(float* buf) {
  helper_float_11_recursive(buf, 11);
}
void helper_float_12_recursive(float* buf, int depth);
void helper_float_12_recursive(float* buf, int depth) {
  if (depth == 10) {
    helper_float_10(buf);
    return;
  }
  if (depth == 12) {
    helper_float_12_recursive(buf + 0, 10);
    helper_float_12_recursive(buf + 1024, 10);
    helper_float_12_recursive(buf + 2048, 10);
    helper_float_12_recursive(buf + 3072, 10);
    for (int j = 0; j < 4096; j += 4096) {
      for (int k = 0; k < 1024; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "LD1 {v2.4S}, [%2]\n"
            "LD1 {v3.4S}, [%3]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "FADD v18.4S, v2.4S, v3.4S\n"
            "FSUB v19.4S, v2.4S, v3.4S\n"
            "FADD v0.4S, v16.4S, v18.4S\n"
            "FSUB v2.4S, v16.4S, v18.4S\n"
            "FADD v1.4S, v17.4S, v19.4S\n"
            "FSUB v3.4S, v17.4S, v19.4S\n"
            "ST1 {v0.4S}, [%0]\n"
            "ST1 {v1.4S}, [%1]\n"
            "ST1 {v2.4S}, [%2]\n"
            "ST1 {v3.4S}, [%3]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 1024),
            "r"(buf + j + k + 2048),
            "r"(buf + j + k + 3072)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_12(float* buf);
void helper_float_12(float* buf) {
  helper_float_12_recursive(buf, 12);
}
static inline void helper_float_13(float* buf);
static inline void helper_float_13(float* buf) {
  for (int j = 0; j < 8192; j += 64) {
    for (int k = 0; k < 4; k += 4) {
      __asm__ volatile(
          "LD1 {v0.4S}, [%0]\n"
          "LD1 {v1.4S}, [%1]\n"
          "LD1 {v2.4S}, [%2]\n"
          "LD1 {v3.4S}, [%3]\n"
          "LD1 {v4.4S}, [%4]\n"
          "LD1 {v5.4S}, [%5]\n"
          "LD1 {v6.4S}, [%6]\n"
          "LD1 {v7.4S}, [%7]\n"
          "LD1 {v8.4S}, [%8]\n"
          "LD1 {v9.4S}, [%9]\n"
          "LD1 {v10.4S}, [%10]\n"
          "LD1 {v11.4S}, [%11]\n"
          "LD1 {v12.4S}, [%12]\n"
          "LD1 {v13.4S}, [%13]\n"
          "LD1 {v14.4S}, [%14]\n"
          "LD1 {v15.4S}, [%15]\n"
          "TRN1 v16.4S, v0.4S, v0.4S\n"
          "FNEG v17.4S, v0.4S\n"
          "TRN2 v17.4S, v0.4S, v17.4S\n"
          "FADD v0.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v1.4S, v1.4S\n"
          "FNEG v17.4S, v1.4S\n"
          "TRN2 v17.4S, v1.4S, v17.4S\n"
          "FADD v1.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v2.4S, v2.4S\n"
          "FNEG v17.4S, v2.4S\n"
          "TRN2 v17.4S, v2.4S, v17.4S\n"
          "FADD v2.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v3.4S, v3.4S\n"
          "FNEG v17.4S, v3.4S\n"
          "TRN2 v17.4S, v3.4S, v17.4S\n"
          "FADD v3.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v4.4S, v4.4S\n"
          "FNEG v17.4S, v4.4S\n"
          "TRN2 v17.4S, v4.4S, v17.4S\n"
          "FADD v4.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v5.4S, v5.4S\n"
          "FNEG v17.4S, v5.4S\n"
          "TRN2 v17.4S, v5.4S, v17.4S\n"
          "FADD v5.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v6.4S, v6.4S\n"
          "FNEG v17.4S, v6.4S\n"
          "TRN2 v17.4S, v6.4S, v17.4S\n"
          "FADD v6.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v7.4S, v7.4S\n"
          "FNEG v17.4S, v7.4S\n"
          "TRN2 v17.4S, v7.4S, v17.4S\n"
          "FADD v7.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v8.4S, v8.4S\n"
          "FNEG v17.4S, v8.4S\n"
          "TRN2 v17.4S, v8.4S, v17.4S\n"
          "FADD v8.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v9.4S, v9.4S\n"
          "FNEG v17.4S, v9.4S\n"
          "TRN2 v17.4S, v9.4S, v17.4S\n"
          "FADD v9.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v10.4S, v10.4S\n"
          "FNEG v17.4S, v10.4S\n"
          "TRN2 v17.4S, v10.4S, v17.4S\n"
          "FADD v10.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v11.4S, v11.4S\n"
          "FNEG v17.4S, v11.4S\n"
          "TRN2 v17.4S, v11.4S, v17.4S\n"
          "FADD v11.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v12.4S, v12.4S\n"
          "FNEG v17.4S, v12.4S\n"
          "TRN2 v17.4S, v12.4S, v17.4S\n"
          "FADD v12.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v13.4S, v13.4S\n"
          "FNEG v17.4S, v13.4S\n"
          "TRN2 v17.4S, v13.4S, v17.4S\n"
          "FADD v13.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v14.4S, v14.4S\n"
          "FNEG v17.4S, v14.4S\n"
          "TRN2 v17.4S, v14.4S, v17.4S\n"
          "FADD v14.4S, v16.4S, v17.4S\n"
          "TRN1 v16.4S, v15.4S, v15.4S\n"
          "FNEG v17.4S, v15.4S\n"
          "TRN2 v17.4S, v15.4S, v17.4S\n"
          "FADD v15.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v0.D[0]\n"
          "FNEG v17.4S, v0.4S\n"
          "INS v17.D[0], v0.D[1]\n"
          "FADD v0.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v1.D[0]\n"
          "FNEG v17.4S, v1.4S\n"
          "INS v17.D[0], v1.D[1]\n"
          "FADD v1.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v2.D[0]\n"
          "FNEG v17.4S, v2.4S\n"
          "INS v17.D[0], v2.D[1]\n"
          "FADD v2.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v3.D[0]\n"
          "FNEG v17.4S, v3.4S\n"
          "INS v17.D[0], v3.D[1]\n"
          "FADD v3.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v4.D[0]\n"
          "FNEG v17.4S, v4.4S\n"
          "INS v17.D[0], v4.D[1]\n"
          "FADD v4.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v5.D[0]\n"
          "FNEG v17.4S, v5.4S\n"
          "INS v17.D[0], v5.D[1]\n"
          "FADD v5.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v6.D[0]\n"
          "FNEG v17.4S, v6.4S\n"
          "INS v17.D[0], v6.D[1]\n"
          "FADD v6.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v7.D[0]\n"
          "FNEG v17.4S, v7.4S\n"
          "INS v17.D[0], v7.D[1]\n"
          "FADD v7.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v8.D[0]\n"
          "FNEG v17.4S, v8.4S\n"
          "INS v17.D[0], v8.D[1]\n"
          "FADD v8.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v9.D[0]\n"
          "FNEG v17.4S, v9.4S\n"
          "INS v17.D[0], v9.D[1]\n"
          "FADD v9.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v10.D[0]\n"
          "FNEG v17.4S, v10.4S\n"
          "INS v17.D[0], v10.D[1]\n"
          "FADD v10.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v11.D[0]\n"
          "FNEG v17.4S, v11.4S\n"
          "INS v17.D[0], v11.D[1]\n"
          "FADD v11.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v12.D[0]\n"
          "FNEG v17.4S, v12.4S\n"
          "INS v17.D[0], v12.D[1]\n"
          "FADD v12.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v13.D[0]\n"
          "FNEG v17.4S, v13.4S\n"
          "INS v17.D[0], v13.D[1]\n"
          "FADD v13.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v14.D[0]\n"
          "FNEG v17.4S, v14.4S\n"
          "INS v17.D[0], v14.D[1]\n"
          "FADD v14.4S, v16.4S, v17.4S\n"
          "DUP v16.2D, v15.D[0]\n"
          "FNEG v17.4S, v15.4S\n"
          "INS v17.D[0], v15.D[1]\n"
          "FADD v15.4S, v16.4S, v17.4S\n"
          "FADD v16.4S, v0.4S, v1.4S\n"
          "FSUB v17.4S, v0.4S, v1.4S\n"
          "FADD v18.4S, v2.4S, v3.4S\n"
          "FSUB v19.4S, v2.4S, v3.4S\n"
          "FADD v20.4S, v4.4S, v5.4S\n"
          "FSUB v21.4S, v4.4S, v5.4S\n"
          "FADD v22.4S, v6.4S, v7.4S\n"
          "FSUB v23.4S, v6.4S, v7.4S\n"
          "FADD v24.4S, v8.4S, v9.4S\n"
          "FSUB v25.4S, v8.4S, v9.4S\n"
          "FADD v26.4S, v10.4S, v11.4S\n"
          "FSUB v27.4S, v10.4S, v11.4S\n"
          "FADD v28.4S, v12.4S, v13.4S\n"
          "FSUB v29.4S, v12.4S, v13.4S\n"
          "FADD v30.4S, v14.4S, v15.4S\n"
          "FSUB v31.4S, v14.4S, v15.4S\n"
          "FADD v0.4S, v16.4S, v18.4S\n"
          "FSUB v2.4S, v16.4S, v18.4S\n"
          "FADD v1.4S, v17.4S, v19.4S\n"
          "FSUB v3.4S, v17.4S, v19.4S\n"
          "FADD v4.4S, v20.4S, v22.4S\n"
          "FSUB v6.4S, v20.4S, v22.4S\n"
          "FADD v5.4S, v21.4S, v23.4S\n"
          "FSUB v7.4S, v21.4S, v23.4S\n"
          "FADD v8.4S, v24.4S, v26.4S\n"
          "FSUB v10.4S, v24.4S, v26.4S\n"
          "FADD v9.4S, v25.4S, v27.4S\n"
          "FSUB v11.4S, v25.4S, v27.4S\n"
          "FADD v12.4S, v28.4S, v30.4S\n"
          "FSUB v14.4S, v28.4S, v30.4S\n"
          "FADD v13.4S, v29.4S, v31.4S\n"
          "FSUB v15.4S, v29.4S, v31.4S\n"
          "FADD v16.4S, v0.4S, v4.4S\n"
          "FSUB v20.4S, v0.4S, v4.4S\n"
          "FADD v17.4S, v1.4S, v5.4S\n"
          "FSUB v21.4S, v1.4S, v5.4S\n"
          "FADD v18.4S, v2.4S, v6.4S\n"
          "FSUB v22.4S, v2.4S, v6.4S\n"
          "FADD v19.4S, v3.4S, v7.4S\n"
          "FSUB v23.4S, v3.4S, v7.4S\n"
          "FADD v24.4S, v8.4S, v12.4S\n"
          "FSUB v28.4S, v8.4S, v12.4S\n"
          "FADD v25.4S, v9.4S, v13.4S\n"
          "FSUB v29.4S, v9.4S, v13.4S\n"
          "FADD v26.4S, v10.4S, v14.4S\n"
          "FSUB v30.4S, v10.4S, v14.4S\n"
          "FADD v27.4S, v11.4S, v15.4S\n"
          "FSUB v31.4S, v11.4S, v15.4S\n"
          "FADD v0.4S, v16.4S, v24.4S\n"
          "FSUB v8.4S, v16.4S, v24.4S\n"
          "FADD v1.4S, v17.4S, v25.4S\n"
          "FSUB v9.4S, v17.4S, v25.4S\n"
          "FADD v2.4S, v18.4S, v26.4S\n"
          "FSUB v10.4S, v18.4S, v26.4S\n"
          "FADD v3.4S, v19.4S, v27.4S\n"
          "FSUB v11.4S, v19.4S, v27.4S\n"
          "FADD v4.4S, v20.4S, v28.4S\n"
          "FSUB v12.4S, v20.4S, v28.4S\n"
          "FADD v5.4S, v21.4S, v29.4S\n"
          "FSUB v13.4S, v21.4S, v29.4S\n"
          "FADD v6.4S, v22.4S, v30.4S\n"
          "FSUB v14.4S, v22.4S, v30.4S\n"
          "FADD v7.4S, v23.4S, v31.4S\n"
          "FSUB v15.4S, v23.4S, v31.4S\n"
          "ST1 {v0.4S}, [%0]\n"
          "ST1 {v1.4S}, [%1]\n"
          "ST1 {v2.4S}, [%2]\n"
          "ST1 {v3.4S}, [%3]\n"
          "ST1 {v4.4S}, [%4]\n"
          "ST1 {v5.4S}, [%5]\n"
          "ST1 {v6.4S}, [%6]\n"
          "ST1 {v7.4S}, [%7]\n"
          "ST1 {v8.4S}, [%8]\n"
          "ST1 {v9.4S}, [%9]\n"
          "ST1 {v10.4S}, [%10]\n"
          "ST1 {v11.4S}, [%11]\n"
          "ST1 {v12.4S}, [%12]\n"
          "ST1 {v13.4S}, [%13]\n"
          "ST1 {v14.4S}, [%14]\n"
          "ST1 {v15.4S}, [%15]\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 4),
          "r"(buf + j + k + 8),
          "r"(buf + j + k + 12),
          "r"(buf + j + k + 16),
          "r"(buf + j + k + 20),
          "r"(buf + j + k + 24),
          "r"(buf + j + k + 28),
          "r"(buf + j + k + 32),
          "r"(buf + j + k + 36),
          "r"(buf + j + k + 40),
          "r"(buf + j + k + 44),
          "r"(buf + j + k + 48),
          "r"(buf + j + k + 52),
          "r"(buf + j + k + 56),
          "r"(buf + j + k + 60)
          : "%v0",
            "%v1",
            "%v2",
            "%v3",
            "%v4",
            "%v5",
            "%v6",
            "%v7",
            "%v8",
            "%v9",
            "%v10",
            "%v11",
            "%v12",
            "%v13",
            "%v14",
            "%v15",
            "%v16",
            "%v17",
            "%v18",
            "%v19",
            "%v20",
            "%v21",
            "%v22",
            "%v23",
            "%v24",
            "%v25",
            "%v26",
            "%v27",
            "%v28",
            "%v29",
            "%v30",
            "%v31",
            "memory");
    }
  }
  for (int j = 0; j < 8192; j += 1024) {
    for (int k = 0; k < 64; k += 4) {
      __asm__ volatile(
          "LD1 {v0.4S}, [%0]\n"
          "LD1 {v1.4S}, [%1]\n"
          "LD1 {v2.4S}, [%2]\n"
          "LD1 {v3.4S}, [%3]\n"
          "LD1 {v4.4S}, [%4]\n"
          "LD1 {v5.4S}, [%5]\n"
          "LD1 {v6.4S}, [%6]\n"
          "LD1 {v7.4S}, [%7]\n"
          "LD1 {v8.4S}, [%8]\n"
          "LD1 {v9.4S}, [%9]\n"
          "LD1 {v10.4S}, [%10]\n"
          "LD1 {v11.4S}, [%11]\n"
          "LD1 {v12.4S}, [%12]\n"
          "LD1 {v13.4S}, [%13]\n"
          "LD1 {v14.4S}, [%14]\n"
          "LD1 {v15.4S}, [%15]\n"
          "FADD v16.4S, v0.4S, v1.4S\n"
          "FSUB v17.4S, v0.4S, v1.4S\n"
          "FADD v18.4S, v2.4S, v3.4S\n"
          "FSUB v19.4S, v2.4S, v3.4S\n"
          "FADD v20.4S, v4.4S, v5.4S\n"
          "FSUB v21.4S, v4.4S, v5.4S\n"
          "FADD v22.4S, v6.4S, v7.4S\n"
          "FSUB v23.4S, v6.4S, v7.4S\n"
          "FADD v24.4S, v8.4S, v9.4S\n"
          "FSUB v25.4S, v8.4S, v9.4S\n"
          "FADD v26.4S, v10.4S, v11.4S\n"
          "FSUB v27.4S, v10.4S, v11.4S\n"
          "FADD v28.4S, v12.4S, v13.4S\n"
          "FSUB v29.4S, v12.4S, v13.4S\n"
          "FADD v30.4S, v14.4S, v15.4S\n"
          "FSUB v31.4S, v14.4S, v15.4S\n"
          "FADD v0.4S, v16.4S, v18.4S\n"
          "FSUB v2.4S, v16.4S, v18.4S\n"
          "FADD v1.4S, v17.4S, v19.4S\n"
          "FSUB v3.4S, v17.4S, v19.4S\n"
          "FADD v4.4S, v20.4S, v22.4S\n"
          "FSUB v6.4S, v20.4S, v22.4S\n"
          "FADD v5.4S, v21.4S, v23.4S\n"
          "FSUB v7.4S, v21.4S, v23.4S\n"
          "FADD v8.4S, v24.4S, v26.4S\n"
          "FSUB v10.4S, v24.4S, v26.4S\n"
          "FADD v9.4S, v25.4S, v27.4S\n"
          "FSUB v11.4S, v25.4S, v27.4S\n"
          "FADD v12.4S, v28.4S, v30.4S\n"
          "FSUB v14.4S, v28.4S, v30.4S\n"
          "FADD v13.4S, v29.4S, v31.4S\n"
          "FSUB v15.4S, v29.4S, v31.4S\n"
          "FADD v16.4S, v0.4S, v4.4S\n"
          "FSUB v20.4S, v0.4S, v4.4S\n"
          "FADD v17.4S, v1.4S, v5.4S\n"
          "FSUB v21.4S, v1.4S, v5.4S\n"
          "FADD v18.4S, v2.4S, v6.4S\n"
          "FSUB v22.4S, v2.4S, v6.4S\n"
          "FADD v19.4S, v3.4S, v7.4S\n"
          "FSUB v23.4S, v3.4S, v7.4S\n"
          "FADD v24.4S, v8.4S, v12.4S\n"
          "FSUB v28.4S, v8.4S, v12.4S\n"
          "FADD v25.4S, v9.4S, v13.4S\n"
          "FSUB v29.4S, v9.4S, v13.4S\n"
          "FADD v26.4S, v10.4S, v14.4S\n"
          "FSUB v30.4S, v10.4S, v14.4S\n"
          "FADD v27.4S, v11.4S, v15.4S\n"
          "FSUB v31.4S, v11.4S, v15.4S\n"
          "FADD v0.4S, v16.4S, v24.4S\n"
          "FSUB v8.4S, v16.4S, v24.4S\n"
          "FADD v1.4S, v17.4S, v25.4S\n"
          "FSUB v9.4S, v17.4S, v25.4S\n"
          "FADD v2.4S, v18.4S, v26.4S\n"
          "FSUB v10.4S, v18.4S, v26.4S\n"
          "FADD v3.4S, v19.4S, v27.4S\n"
          "FSUB v11.4S, v19.4S, v27.4S\n"
          "FADD v4.4S, v20.4S, v28.4S\n"
          "FSUB v12.4S, v20.4S, v28.4S\n"
          "FADD v5.4S, v21.4S, v29.4S\n"
          "FSUB v13.4S, v21.4S, v29.4S\n"
          "FADD v6.4S, v22.4S, v30.4S\n"
          "FSUB v14.4S, v22.4S, v30.4S\n"
          "FADD v7.4S, v23.4S, v31.4S\n"
          "FSUB v15.4S, v23.4S, v31.4S\n"
          "ST1 {v0.4S}, [%0]\n"
          "ST1 {v1.4S}, [%1]\n"
          "ST1 {v2.4S}, [%2]\n"
          "ST1 {v3.4S}, [%3]\n"
          "ST1 {v4.4S}, [%4]\n"
          "ST1 {v5.4S}, [%5]\n"
          "ST1 {v6.4S}, [%6]\n"
          "ST1 {v7.4S}, [%7]\n"
          "ST1 {v8.4S}, [%8]\n"
          "ST1 {v9.4S}, [%9]\n"
          "ST1 {v10.4S}, [%10]\n"
          "ST1 {v11.4S}, [%11]\n"
          "ST1 {v12.4S}, [%12]\n"
          "ST1 {v13.4S}, [%13]\n"
          "ST1 {v14.4S}, [%14]\n"
          "ST1 {v15.4S}, [%15]\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 64),
          "r"(buf + j + k + 128),
          "r"(buf + j + k + 192),
          "r"(buf + j + k + 256),
          "r"(buf + j + k + 320),
          "r"(buf + j + k + 384),
          "r"(buf + j + k + 448),
          "r"(buf + j + k + 512),
          "r"(buf + j + k + 576),
          "r"(buf + j + k + 640),
          "r"(buf + j + k + 704),
          "r"(buf + j + k + 768),
          "r"(buf + j + k + 832),
          "r"(buf + j + k + 896),
          "r"(buf + j + k + 960)
          : "%v0",
            "%v1",
            "%v2",
            "%v3",
            "%v4",
            "%v5",
            "%v6",
            "%v7",
            "%v8",
            "%v9",
            "%v10",
            "%v11",
            "%v12",
            "%v13",
            "%v14",
            "%v15",
            "%v16",
            "%v17",
            "%v18",
            "%v19",
            "%v20",
            "%v21",
            "%v22",
            "%v23",
            "%v24",
            "%v25",
            "%v26",
            "%v27",
            "%v28",
            "%v29",
            "%v30",
            "%v31",
            "memory");
    }
  }
  for (int j = 0; j < 8192; j += 8192) {
    for (int k = 0; k < 1024; k += 4) {
      __asm__ volatile(
          "LD1 {v0.4S}, [%0]\n"
          "LD1 {v1.4S}, [%1]\n"
          "LD1 {v2.4S}, [%2]\n"
          "LD1 {v3.4S}, [%3]\n"
          "LD1 {v4.4S}, [%4]\n"
          "LD1 {v5.4S}, [%5]\n"
          "LD1 {v6.4S}, [%6]\n"
          "LD1 {v7.4S}, [%7]\n"
          "FADD v16.4S, v0.4S, v1.4S\n"
          "FSUB v17.4S, v0.4S, v1.4S\n"
          "FADD v18.4S, v2.4S, v3.4S\n"
          "FSUB v19.4S, v2.4S, v3.4S\n"
          "FADD v20.4S, v4.4S, v5.4S\n"
          "FSUB v21.4S, v4.4S, v5.4S\n"
          "FADD v22.4S, v6.4S, v7.4S\n"
          "FSUB v23.4S, v6.4S, v7.4S\n"
          "FADD v0.4S, v16.4S, v18.4S\n"
          "FSUB v2.4S, v16.4S, v18.4S\n"
          "FADD v1.4S, v17.4S, v19.4S\n"
          "FSUB v3.4S, v17.4S, v19.4S\n"
          "FADD v4.4S, v20.4S, v22.4S\n"
          "FSUB v6.4S, v20.4S, v22.4S\n"
          "FADD v5.4S, v21.4S, v23.4S\n"
          "FSUB v7.4S, v21.4S, v23.4S\n"
          "FADD v16.4S, v0.4S, v4.4S\n"
          "FSUB v20.4S, v0.4S, v4.4S\n"
          "FADD v17.4S, v1.4S, v5.4S\n"
          "FSUB v21.4S, v1.4S, v5.4S\n"
          "FADD v18.4S, v2.4S, v6.4S\n"
          "FSUB v22.4S, v2.4S, v6.4S\n"
          "FADD v19.4S, v3.4S, v7.4S\n"
          "FSUB v23.4S, v3.4S, v7.4S\n"
          "ST1 {v16.4S}, [%0]\n"
          "ST1 {v17.4S}, [%1]\n"
          "ST1 {v18.4S}, [%2]\n"
          "ST1 {v19.4S}, [%3]\n"
          "ST1 {v20.4S}, [%4]\n"
          "ST1 {v21.4S}, [%5]\n"
          "ST1 {v22.4S}, [%6]\n"
          "ST1 {v23.4S}, [%7]\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 1024),
          "r"(buf + j + k + 2048),
          "r"(buf + j + k + 3072),
          "r"(buf + j + k + 4096),
          "r"(buf + j + k + 5120),
          "r"(buf + j + k + 6144),
          "r"(buf + j + k + 7168)
          : "%v0",
            "%v1",
            "%v2",
            "%v3",
            "%v4",
            "%v5",
            "%v6",
            "%v7",
            "%v8",
            "%v9",
            "%v10",
            "%v11",
            "%v12",
            "%v13",
            "%v14",
            "%v15",
            "%v16",
            "%v17",
            "%v18",
            "%v19",
            "%v20",
            "%v21",
            "%v22",
            "%v23",
            "%v24",
            "%v25",
            "%v26",
            "%v27",
            "%v28",
            "%v29",
            "%v30",
            "%v31",
            "memory");
    }
  }
}
void helper_float_14_recursive(float* buf, int depth);
void helper_float_14_recursive(float* buf, int depth) {
  if (depth == 10) {
    helper_float_10(buf);
    return;
  }
  if (depth == 14) {
    helper_float_14_recursive(buf + 0, 10);
    helper_float_14_recursive(buf + 1024, 10);
    helper_float_14_recursive(buf + 2048, 10);
    helper_float_14_recursive(buf + 3072, 10);
    helper_float_14_recursive(buf + 4096, 10);
    helper_float_14_recursive(buf + 5120, 10);
    helper_float_14_recursive(buf + 6144, 10);
    helper_float_14_recursive(buf + 7168, 10);
    helper_float_14_recursive(buf + 8192, 10);
    helper_float_14_recursive(buf + 9216, 10);
    helper_float_14_recursive(buf + 10240, 10);
    helper_float_14_recursive(buf + 11264, 10);
    helper_float_14_recursive(buf + 12288, 10);
    helper_float_14_recursive(buf + 13312, 10);
    helper_float_14_recursive(buf + 14336, 10);
    helper_float_14_recursive(buf + 15360, 10);
    for (int j = 0; j < 16384; j += 16384) {
      for (int k = 0; k < 1024; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "LD1 {v2.4S}, [%2]\n"
            "LD1 {v3.4S}, [%3]\n"
            "LD1 {v4.4S}, [%4]\n"
            "LD1 {v5.4S}, [%5]\n"
            "LD1 {v6.4S}, [%6]\n"
            "LD1 {v7.4S}, [%7]\n"
            "LD1 {v8.4S}, [%8]\n"
            "LD1 {v9.4S}, [%9]\n"
            "LD1 {v10.4S}, [%10]\n"
            "LD1 {v11.4S}, [%11]\n"
            "LD1 {v12.4S}, [%12]\n"
            "LD1 {v13.4S}, [%13]\n"
            "LD1 {v14.4S}, [%14]\n"
            "LD1 {v15.4S}, [%15]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "FADD v18.4S, v2.4S, v3.4S\n"
            "FSUB v19.4S, v2.4S, v3.4S\n"
            "FADD v20.4S, v4.4S, v5.4S\n"
            "FSUB v21.4S, v4.4S, v5.4S\n"
            "FADD v22.4S, v6.4S, v7.4S\n"
            "FSUB v23.4S, v6.4S, v7.4S\n"
            "FADD v24.4S, v8.4S, v9.4S\n"
            "FSUB v25.4S, v8.4S, v9.4S\n"
            "FADD v26.4S, v10.4S, v11.4S\n"
            "FSUB v27.4S, v10.4S, v11.4S\n"
            "FADD v28.4S, v12.4S, v13.4S\n"
            "FSUB v29.4S, v12.4S, v13.4S\n"
            "FADD v30.4S, v14.4S, v15.4S\n"
            "FSUB v31.4S, v14.4S, v15.4S\n"
            "FADD v0.4S, v16.4S, v18.4S\n"
            "FSUB v2.4S, v16.4S, v18.4S\n"
            "FADD v1.4S, v17.4S, v19.4S\n"
            "FSUB v3.4S, v17.4S, v19.4S\n"
            "FADD v4.4S, v20.4S, v22.4S\n"
            "FSUB v6.4S, v20.4S, v22.4S\n"
            "FADD v5.4S, v21.4S, v23.4S\n"
            "FSUB v7.4S, v21.4S, v23.4S\n"
            "FADD v8.4S, v24.4S, v26.4S\n"
            "FSUB v10.4S, v24.4S, v26.4S\n"
            "FADD v9.4S, v25.4S, v27.4S\n"
            "FSUB v11.4S, v25.4S, v27.4S\n"
            "FADD v12.4S, v28.4S, v30.4S\n"
            "FSUB v14.4S, v28.4S, v30.4S\n"
            "FADD v13.4S, v29.4S, v31.4S\n"
            "FSUB v15.4S, v29.4S, v31.4S\n"
            "FADD v16.4S, v0.4S, v4.4S\n"
            "FSUB v20.4S, v0.4S, v4.4S\n"
            "FADD v17.4S, v1.4S, v5.4S\n"
            "FSUB v21.4S, v1.4S, v5.4S\n"
            "FADD v18.4S, v2.4S, v6.4S\n"
            "FSUB v22.4S, v2.4S, v6.4S\n"
            "FADD v19.4S, v3.4S, v7.4S\n"
            "FSUB v23.4S, v3.4S, v7.4S\n"
            "FADD v24.4S, v8.4S, v12.4S\n"
            "FSUB v28.4S, v8.4S, v12.4S\n"
            "FADD v25.4S, v9.4S, v13.4S\n"
            "FSUB v29.4S, v9.4S, v13.4S\n"
            "FADD v26.4S, v10.4S, v14.4S\n"
            "FSUB v30.4S, v10.4S, v14.4S\n"
            "FADD v27.4S, v11.4S, v15.4S\n"
            "FSUB v31.4S, v11.4S, v15.4S\n"
            "FADD v0.4S, v16.4S, v24.4S\n"
            "FSUB v8.4S, v16.4S, v24.4S\n"
            "FADD v1.4S, v17.4S, v25.4S\n"
            "FSUB v9.4S, v17.4S, v25.4S\n"
            "FADD v2.4S, v18.4S, v26.4S\n"
            "FSUB v10.4S, v18.4S, v26.4S\n"
            "FADD v3.4S, v19.4S, v27.4S\n"
            "FSUB v11.4S, v19.4S, v27.4S\n"
            "FADD v4.4S, v20.4S, v28.4S\n"
            "FSUB v12.4S, v20.4S, v28.4S\n"
            "FADD v5.4S, v21.4S, v29.4S\n"
            "FSUB v13.4S, v21.4S, v29.4S\n"
            "FADD v6.4S, v22.4S, v30.4S\n"
            "FSUB v14.4S, v22.4S, v30.4S\n"
            "FADD v7.4S, v23.4S, v31.4S\n"
            "FSUB v15.4S, v23.4S, v31.4S\n"
            "ST1 {v0.4S}, [%0]\n"
            "ST1 {v1.4S}, [%1]\n"
            "ST1 {v2.4S}, [%2]\n"
            "ST1 {v3.4S}, [%3]\n"
            "ST1 {v4.4S}, [%4]\n"
            "ST1 {v5.4S}, [%5]\n"
            "ST1 {v6.4S}, [%6]\n"
            "ST1 {v7.4S}, [%7]\n"
            "ST1 {v8.4S}, [%8]\n"
            "ST1 {v9.4S}, [%9]\n"
            "ST1 {v10.4S}, [%10]\n"
            "ST1 {v11.4S}, [%11]\n"
            "ST1 {v12.4S}, [%12]\n"
            "ST1 {v13.4S}, [%13]\n"
            "ST1 {v14.4S}, [%14]\n"
            "ST1 {v15.4S}, [%15]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 1024),
            "r"(buf + j + k + 2048),
            "r"(buf + j + k + 3072),
            "r"(buf + j + k + 4096),
            "r"(buf + j + k + 5120),
            "r"(buf + j + k + 6144),
            "r"(buf + j + k + 7168),
            "r"(buf + j + k + 8192),
            "r"(buf + j + k + 9216),
            "r"(buf + j + k + 10240),
            "r"(buf + j + k + 11264),
            "r"(buf + j + k + 12288),
            "r"(buf + j + k + 13312),
            "r"(buf + j + k + 14336),
            "r"(buf + j + k + 15360)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_14(float* buf);
void helper_float_14(float* buf) {
  helper_float_14_recursive(buf, 14);
}
void helper_float_15_recursive(float* buf, int depth);
void helper_float_15_recursive(float* buf, int depth) {
  if (depth == 13) {
    helper_float_13(buf);
    return;
  }
  if (depth == 15) {
    helper_float_15_recursive(buf + 0, 13);
    helper_float_15_recursive(buf + 8192, 13);
    helper_float_15_recursive(buf + 16384, 13);
    helper_float_15_recursive(buf + 24576, 13);
    for (int j = 0; j < 32768; j += 32768) {
      for (int k = 0; k < 8192; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "LD1 {v2.4S}, [%2]\n"
            "LD1 {v3.4S}, [%3]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "FADD v18.4S, v2.4S, v3.4S\n"
            "FSUB v19.4S, v2.4S, v3.4S\n"
            "FADD v0.4S, v16.4S, v18.4S\n"
            "FSUB v2.4S, v16.4S, v18.4S\n"
            "FADD v1.4S, v17.4S, v19.4S\n"
            "FSUB v3.4S, v17.4S, v19.4S\n"
            "ST1 {v0.4S}, [%0]\n"
            "ST1 {v1.4S}, [%1]\n"
            "ST1 {v2.4S}, [%2]\n"
            "ST1 {v3.4S}, [%3]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 8192),
            "r"(buf + j + k + 16384),
            "r"(buf + j + k + 24576)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_15(float* buf);
void helper_float_15(float* buf) {
  helper_float_15_recursive(buf, 15);
}
void helper_float_16_recursive(float* buf, int depth);
void helper_float_16_recursive(float* buf, int depth) {
  if (depth == 15) {
    helper_float_15(buf);
    return;
  }
  if (depth == 16) {
    helper_float_16_recursive(buf + 0, 15);
    helper_float_16_recursive(buf + 32768, 15);
    for (int j = 0; j < 65536; j += 65536) {
      for (int k = 0; k < 32768; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "ST1 {v16.4S}, [%0]\n"
            "ST1 {v17.4S}, [%1]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 32768)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_16(float* buf);
void helper_float_16(float* buf) {
  helper_float_16_recursive(buf, 16);
}
void helper_float_17_recursive(float* buf, int depth);
void helper_float_17_recursive(float* buf, int depth) {
  if (depth == 15) {
    helper_float_15(buf);
    return;
  }
  if (depth == 17) {
    helper_float_17_recursive(buf + 0, 15);
    helper_float_17_recursive(buf + 32768, 15);
    helper_float_17_recursive(buf + 65536, 15);
    helper_float_17_recursive(buf + 98304, 15);
    for (int j = 0; j < 131072; j += 131072) {
      for (int k = 0; k < 32768; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "LD1 {v2.4S}, [%2]\n"
            "LD1 {v3.4S}, [%3]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "FADD v18.4S, v2.4S, v3.4S\n"
            "FSUB v19.4S, v2.4S, v3.4S\n"
            "FADD v0.4S, v16.4S, v18.4S\n"
            "FSUB v2.4S, v16.4S, v18.4S\n"
            "FADD v1.4S, v17.4S, v19.4S\n"
            "FSUB v3.4S, v17.4S, v19.4S\n"
            "ST1 {v0.4S}, [%0]\n"
            "ST1 {v1.4S}, [%1]\n"
            "ST1 {v2.4S}, [%2]\n"
            "ST1 {v3.4S}, [%3]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 32768),
            "r"(buf + j + k + 65536),
            "r"(buf + j + k + 98304)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_17(float* buf);
void helper_float_17(float* buf) {
  helper_float_17_recursive(buf, 17);
}
void helper_float_18_recursive(float* buf, int depth);
void helper_float_18_recursive(float* buf, int depth) {
  if (depth == 17) {
    helper_float_17(buf);
    return;
  }
  if (depth == 18) {
    helper_float_18_recursive(buf + 0, 17);
    helper_float_18_recursive(buf + 131072, 17);
    for (int j = 0; j < 262144; j += 262144) {
      for (int k = 0; k < 131072; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "ST1 {v16.4S}, [%0]\n"
            "ST1 {v17.4S}, [%1]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 131072)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_18(float* buf);
void helper_float_18(float* buf) {
  helper_float_18_recursive(buf, 18);
}
void helper_float_19_recursive(float* buf, int depth);
void helper_float_19_recursive(float* buf, int depth) {
  if (depth == 18) {
    helper_float_18(buf);
    return;
  }
  if (depth == 19) {
    helper_float_19_recursive(buf + 0, 18);
    helper_float_19_recursive(buf + 262144, 18);
    for (int j = 0; j < 524288; j += 524288) {
      for (int k = 0; k < 262144; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "ST1 {v16.4S}, [%0]\n"
            "ST1 {v17.4S}, [%1]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 262144)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_19(float* buf);
void helper_float_19(float* buf) {
  helper_float_19_recursive(buf, 19);
}
void helper_float_20_recursive(float* buf, int depth);
void helper_float_20_recursive(float* buf, int depth) {
  if (depth == 18) {
    helper_float_18(buf);
    return;
  }
  if (depth == 20) {
    helper_float_20_recursive(buf + 0, 18);
    helper_float_20_recursive(buf + 262144, 18);
    helper_float_20_recursive(buf + 524288, 18);
    helper_float_20_recursive(buf + 786432, 18);
    for (int j = 0; j < 1048576; j += 1048576) {
      for (int k = 0; k < 262144; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "LD1 {v2.4S}, [%2]\n"
            "LD1 {v3.4S}, [%3]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "FADD v18.4S, v2.4S, v3.4S\n"
            "FSUB v19.4S, v2.4S, v3.4S\n"
            "FADD v0.4S, v16.4S, v18.4S\n"
            "FSUB v2.4S, v16.4S, v18.4S\n"
            "FADD v1.4S, v17.4S, v19.4S\n"
            "FSUB v3.4S, v17.4S, v19.4S\n"
            "ST1 {v0.4S}, [%0]\n"
            "ST1 {v1.4S}, [%1]\n"
            "ST1 {v2.4S}, [%2]\n"
            "ST1 {v3.4S}, [%3]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 262144),
            "r"(buf + j + k + 524288),
            "r"(buf + j + k + 786432)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_20(float* buf);
void helper_float_20(float* buf) {
  helper_float_20_recursive(buf, 20);
}
void helper_float_21_recursive(float* buf, int depth);
void helper_float_21_recursive(float* buf, int depth) {
  if (depth == 20) {
    helper_float_20(buf);
    return;
  }
  if (depth == 21) {
    helper_float_21_recursive(buf + 0, 20);
    helper_float_21_recursive(buf + 1048576, 20);
    for (int j = 0; j < 2097152; j += 2097152) {
      for (int k = 0; k < 1048576; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "ST1 {v16.4S}, [%0]\n"
            "ST1 {v17.4S}, [%1]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 1048576)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_21(float* buf);
void helper_float_21(float* buf) {
  helper_float_21_recursive(buf, 21);
}
void helper_float_22_recursive(float* buf, int depth);
void helper_float_22_recursive(float* buf, int depth) {
  if (depth == 20) {
    helper_float_20(buf);
    return;
  }
  if (depth == 22) {
    helper_float_22_recursive(buf + 0, 20);
    helper_float_22_recursive(buf + 1048576, 20);
    helper_float_22_recursive(buf + 2097152, 20);
    helper_float_22_recursive(buf + 3145728, 20);
    for (int j = 0; j < 4194304; j += 4194304) {
      for (int k = 0; k < 1048576; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "LD1 {v2.4S}, [%2]\n"
            "LD1 {v3.4S}, [%3]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "FADD v18.4S, v2.4S, v3.4S\n"
            "FSUB v19.4S, v2.4S, v3.4S\n"
            "FADD v0.4S, v16.4S, v18.4S\n"
            "FSUB v2.4S, v16.4S, v18.4S\n"
            "FADD v1.4S, v17.4S, v19.4S\n"
            "FSUB v3.4S, v17.4S, v19.4S\n"
            "ST1 {v0.4S}, [%0]\n"
            "ST1 {v1.4S}, [%1]\n"
            "ST1 {v2.4S}, [%2]\n"
            "ST1 {v3.4S}, [%3]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 1048576),
            "r"(buf + j + k + 2097152),
            "r"(buf + j + k + 3145728)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_22(float* buf);
void helper_float_22(float* buf) {
  helper_float_22_recursive(buf, 22);
}
void helper_float_23_recursive(float* buf, int depth);
void helper_float_23_recursive(float* buf, int depth) {
  if (depth == 22) {
    helper_float_22(buf);
    return;
  }
  if (depth == 23) {
    helper_float_23_recursive(buf + 0, 22);
    helper_float_23_recursive(buf + 4194304, 22);
    for (int j = 0; j < 8388608; j += 8388608) {
      for (int k = 0; k < 4194304; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "ST1 {v16.4S}, [%0]\n"
            "ST1 {v17.4S}, [%1]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 4194304)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_23(float* buf);
void helper_float_23(float* buf) {
  helper_float_23_recursive(buf, 23);
}
void helper_float_24_recursive(float* buf, int depth);
void helper_float_24_recursive(float* buf, int depth) {
  if (depth == 23) {
    helper_float_23(buf);
    return;
  }
  if (depth == 24) {
    helper_float_24_recursive(buf + 0, 23);
    helper_float_24_recursive(buf + 8388608, 23);
    for (int j = 0; j < 16777216; j += 16777216) {
      for (int k = 0; k < 8388608; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "ST1 {v16.4S}, [%0]\n"
            "ST1 {v17.4S}, [%1]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 8388608)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_24(float* buf);
void helper_float_24(float* buf) {
  helper_float_24_recursive(buf, 24);
}
void helper_float_25_recursive(float* buf, int depth);
void helper_float_25_recursive(float* buf, int depth) {
  if (depth == 23) {
    helper_float_23(buf);
    return;
  }
  if (depth == 25) {
    helper_float_25_recursive(buf + 0, 23);
    helper_float_25_recursive(buf + 8388608, 23);
    helper_float_25_recursive(buf + 16777216, 23);
    helper_float_25_recursive(buf + 25165824, 23);
    for (int j = 0; j < 33554432; j += 33554432) {
      for (int k = 0; k < 8388608; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "LD1 {v2.4S}, [%2]\n"
            "LD1 {v3.4S}, [%3]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "FADD v18.4S, v2.4S, v3.4S\n"
            "FSUB v19.4S, v2.4S, v3.4S\n"
            "FADD v0.4S, v16.4S, v18.4S\n"
            "FSUB v2.4S, v16.4S, v18.4S\n"
            "FADD v1.4S, v17.4S, v19.4S\n"
            "FSUB v3.4S, v17.4S, v19.4S\n"
            "ST1 {v0.4S}, [%0]\n"
            "ST1 {v1.4S}, [%1]\n"
            "ST1 {v2.4S}, [%2]\n"
            "ST1 {v3.4S}, [%3]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 8388608),
            "r"(buf + j + k + 16777216),
            "r"(buf + j + k + 25165824)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_25(float* buf);
void helper_float_25(float* buf) {
  helper_float_25_recursive(buf, 25);
}
void helper_float_26_recursive(float* buf, int depth);
void helper_float_26_recursive(float* buf, int depth) {
  if (depth == 25) {
    helper_float_25(buf);
    return;
  }
  if (depth == 26) {
    helper_float_26_recursive(buf + 0, 25);
    helper_float_26_recursive(buf + 33554432, 25);
    for (int j = 0; j < 67108864; j += 67108864) {
      for (int k = 0; k < 33554432; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "ST1 {v16.4S}, [%0]\n"
            "ST1 {v17.4S}, [%1]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 33554432)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_26(float* buf);
void helper_float_26(float* buf) {
  helper_float_26_recursive(buf, 26);
}
void helper_float_27_recursive(float* buf, int depth);
void helper_float_27_recursive(float* buf, int depth) {
  if (depth == 26) {
    helper_float_26(buf);
    return;
  }
  if (depth == 27) {
    helper_float_27_recursive(buf + 0, 26);
    helper_float_27_recursive(buf + 67108864, 26);
    for (int j = 0; j < 134217728; j += 134217728) {
      for (int k = 0; k < 67108864; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "ST1 {v16.4S}, [%0]\n"
            "ST1 {v17.4S}, [%1]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 67108864)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_27(float* buf);
void helper_float_27(float* buf) {
  helper_float_27_recursive(buf, 27);
}
void helper_float_28_recursive(float* buf, int depth);
void helper_float_28_recursive(float* buf, int depth) {
  if (depth == 26) {
    helper_float_26(buf);
    return;
  }
  if (depth == 28) {
    helper_float_28_recursive(buf + 0, 26);
    helper_float_28_recursive(buf + 67108864, 26);
    helper_float_28_recursive(buf + 134217728, 26);
    helper_float_28_recursive(buf + 201326592, 26);
    for (int j = 0; j < 268435456; j += 268435456) {
      for (int k = 0; k < 67108864; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "LD1 {v2.4S}, [%2]\n"
            "LD1 {v3.4S}, [%3]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "FADD v18.4S, v2.4S, v3.4S\n"
            "FSUB v19.4S, v2.4S, v3.4S\n"
            "FADD v0.4S, v16.4S, v18.4S\n"
            "FSUB v2.4S, v16.4S, v18.4S\n"
            "FADD v1.4S, v17.4S, v19.4S\n"
            "FSUB v3.4S, v17.4S, v19.4S\n"
            "ST1 {v0.4S}, [%0]\n"
            "ST1 {v1.4S}, [%1]\n"
            "ST1 {v2.4S}, [%2]\n"
            "ST1 {v3.4S}, [%3]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 67108864),
            "r"(buf + j + k + 134217728),
            "r"(buf + j + k + 201326592)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_28(float* buf);
void helper_float_28(float* buf) {
  helper_float_28_recursive(buf, 28);
}
void helper_float_29_recursive(float* buf, int depth);
void helper_float_29_recursive(float* buf, int depth) {
  if (depth == 28) {
    helper_float_28(buf);
    return;
  }
  if (depth == 29) {
    helper_float_29_recursive(buf + 0, 28);
    helper_float_29_recursive(buf + 268435456, 28);
    for (int j = 0; j < 536870912; j += 536870912) {
      for (int k = 0; k < 268435456; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "ST1 {v16.4S}, [%0]\n"
            "ST1 {v17.4S}, [%1]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 268435456)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_29(float* buf);
void helper_float_29(float* buf) {
  helper_float_29_recursive(buf, 29);
}
void helper_float_30_recursive(float* buf, int depth);
void helper_float_30_recursive(float* buf, int depth) {
  if (depth == 29) {
    helper_float_29(buf);
    return;
  }
  if (depth == 30) {
    helper_float_30_recursive(buf + 0, 29);
    helper_float_30_recursive(buf + 536870912, 29);
    for (int j = 0; j < 1073741824; j += 1073741824) {
      for (int k = 0; k < 536870912; k += 4) {
        __asm__ volatile(
            "LD1 {v0.4S}, [%0]\n"
            "LD1 {v1.4S}, [%1]\n"
            "FADD v16.4S, v0.4S, v1.4S\n"
            "FSUB v17.4S, v0.4S, v1.4S\n"
            "ST1 {v16.4S}, [%0]\n"
            "ST1 {v17.4S}, [%1]\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 536870912)
            : "%v0",
              "%v1",
              "%v2",
              "%v3",
              "%v4",
              "%v5",
              "%v6",
              "%v7",
              "%v8",
              "%v9",
              "%v10",
              "%v11",
              "%v12",
              "%v13",
              "%v14",
              "%v15",
              "%v16",
              "%v17",
              "%v18",
              "%v19",
              "%v20",
              "%v21",
              "%v22",
              "%v23",
              "%v24",
              "%v25",
              "%v26",
              "%v27",
              "%v28",
              "%v29",
              "%v30",
              "%v31",
              "memory");
      }
    }
    return;
  }
}
void helper_float_30(float* buf);
void helper_float_30(float* buf) {
  helper_float_30_recursive(buf, 30);
}
int fht_float(float* buf, int log_n) {
  if (log_n == 0) {
    return 0;
  }
  if (log_n == 1) {
    helper_float_1(buf);
    return 0;
  }
  if (log_n == 2) {
    helper_float_2(buf);
    return 0;
  }
  if (log_n == 3) {
    helper_float_3(buf);
    return 0;
  }
  if (log_n == 4) {
    helper_float_4(buf);
    return 0;
  }
  if (log_n == 5) {
    helper_float_5(buf);
    return 0;
  }
  if (log_n == 6) {
    helper_float_6(buf);
    return 0;
  }
  if (log_n == 7) {
    helper_float_7(buf);
    return 0;
  }
  if (log_n == 8) {
    helper_float_8(buf);
    return 0;
  }
  if (log_n == 9) {
    helper_float_9(buf);
    return 0;
  }
  if (log_n == 10) {
    helper_float_10(buf);
    return 0;
  }
  if (log_n == 11) {
    helper_float_11(buf);
    return 0;
  }
  if (log_n == 12) {
    helper_float_12(buf);
    return 0;
  }
  if (log_n == 13) {
    helper_float_13(buf);
    return 0;
  }
  if (log_n == 14) {
    helper_float_14(buf);
    return 0;
  }
  if (log_n == 15) {
    helper_float_15(buf);
    return 0;
  }
  if (log_n == 16) {
    helper_float_16(buf);
    return 0;
  }
  if (log_n == 17) {
    helper_float_17(buf);
    return 0;
  }
  if (log_n == 18) {
    helper_float_18(buf);
    return 0;
  }
  if (log_n == 19) {
    helper_float_19(buf);
    return 0;
  }
  if (log_n == 20) {
    helper_float_20(buf);
    return 0;
  }
  if (log_n == 21) {
    helper_float_21(buf);
    return 0;
  }
  if (log_n == 22) {
    helper_float_22(buf);
    return 0;
  }
  if (log_n == 23) {
    helper_float_23(buf);
    return 0;
  }
  if (log_n == 24) {
    helper_float_24(buf);
    return 0;
  }
  if (log_n == 25) {
    helper_float_25(buf);
    return 0;
  }
  if (log_n == 26) {
    helper_float_26(buf);
    return 0;
  }
  if (log_n == 27) {
    helper_float_27(buf);
    return 0;
  }
  if (log_n == 28) {
    helper_float_28(buf);
    return 0;
  }
  if (log_n == 29) {
    helper_float_29(buf);
    return 0;
  }
  if (log_n == 30) {
    helper_float_30(buf);
    return 0;
  }
  return 1;
}
