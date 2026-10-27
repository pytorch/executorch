/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Standalone QNN repro: FullyConnected + BLOCKWISE_EXPANSION weight + int32
// bias. QNN reads the weight's scaleOffsets / blocksScale8 as in_features (512)
// entries instead of out_features (32); see README.md.
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdint.h>
static int seeded_ = 0;
#include "QnnInterface.h"
#include "QnnOpDef.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpGraph.h"
#define CHECK(x) do { Qnn_ErrorHandle_t e_ = (x); if (e_ != QNN_SUCCESS) { fprintf(stderr, "%s:%d %s failed %lu\n", __FILE__, __LINE__, #x, (unsigned long)e_); exit(1); } } while (0)
static void* readFile(const char* p, size_t* n) { FILE* f = fopen(p, "rb"); fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET); void* b = malloc(*n); fread(b, 1, *n, f); fclose(f); return b; }
static QNN_INTERFACE_VER_TYPE qnn;
static int envInt(const char* k, int d) { const char* v = getenv(k); return v ? atoi(v) : d; }
static const char* recPath(const char* f) { static char b[512]; const char* d = getenv("REC_DIR"); snprintf(b, sizeof b, "%s/%s", d ? d : ".", f); return b; }
static Qnn_BackendHandle_t backend;
static Qnn_DeviceHandle_t device;
static Qnn_ScaleOffset_t so4[32] = {{0.008895148523151875f, 0}, {0.011323409155011177f, 0}, {0.005254262592643499f, 0}, {0.005643129348754883f, 0}, {0.0072091324254870415f, 0}, {0.010125703178346157f, 0}, {0.00884011946618557f, 0}, {0.012468256987631321f, 0}, {0.008532392792403698f, 0}, {0.01010987814515829f, 0}, {0.007579405792057514f, 0}, {0.00805104710161686f, 0}, {0.004663622938096523f, 0}, {0.005971954669803381f, 0}, {0.007088289596140385f, 0}, {0.009093944914638996f, 0}, {0.010693460702896118f, 0}, {0.011607244610786438f, 0}, {0.0059020486660301685f, 0}, {0.006984540726989508f, 0}, {0.010550076141953468f, 0}, {0.012635661289095879f, 0}, {0.008009820245206356f, 0}, {0.012269249185919762f, 0}, {0.008209003135561943f, 0}, {0.009400955401360989f, 0}, {0.012970875017344952f, 0}, {0.004787185695022345f, 0}, {0.006118134129792452f, 0}, {0.007798369508236647f, 0}, {0.007188393268734217f, 0}, {0.012785717844963074f, 0}};
static uint8_t bs4[512] = {16, 4, 2, 1, 12, 6, 3, 9, 4, 2, 1, 12, 6, 3, 9, 4, 4, 16, 1, 12, 6, 3, 9, 4, 2, 1, 12, 6, 3, 9, 4, 2, 2, 1, 16, 6, 3, 9, 4, 2, 1, 12, 6, 3, 9, 4, 2, 1, 1, 12, 6, 16, 9, 4, 2, 1, 12, 6, 3, 9, 4, 2, 1, 12, 12, 6, 3, 9, 16, 2, 1, 12, 6, 3, 9, 4, 2, 1, 12, 6, 6, 3, 9, 4, 2, 16, 12, 6, 3, 9, 4, 2, 1, 12, 6, 3, 3, 9, 4, 2, 1, 12, 16, 3, 9, 4, 2, 1, 12, 6, 3, 9, 9, 4, 2, 1, 12, 6, 3, 16, 4, 2, 1, 12, 6, 3, 9, 4, 4, 2, 1, 12, 6, 3, 9, 4, 16, 1, 12, 6, 3, 9, 4, 2, 2, 1, 12, 6, 3, 9, 4, 2, 1, 16, 6, 3, 9, 4, 2, 1, 1, 12, 6, 3, 9, 4, 2, 1, 12, 6, 16, 9, 4, 2, 1, 12, 12, 6, 3, 9, 4, 2, 1, 12, 6, 3, 9, 16, 2, 1, 12, 6, 6, 3, 9, 4, 2, 1, 12, 6, 3, 9, 4, 2, 16, 12, 6, 3, 3, 9, 4, 2, 1, 12, 6, 3, 9, 4, 2, 1, 12, 16, 3, 9, 9, 4, 2, 1, 12, 6, 3, 9, 4, 2, 1, 12, 6, 3, 16, 4, 4, 2, 1, 12, 6, 3, 9, 4, 2, 1, 12, 6, 3, 9, 4, 16, 16, 1, 12, 6, 3, 9, 4, 2, 1, 12, 6, 3, 9, 4, 2, 1, 1, 16, 6, 3, 9, 4, 2, 1, 12, 6, 3, 9, 4, 2, 1, 12, 12, 6, 16, 9, 4, 2, 1, 12, 6, 3, 9, 4, 2, 1, 12, 6, 6, 3, 9, 16, 2, 1, 12, 6, 3, 9, 4, 2, 1, 12, 6, 3, 3, 9, 4, 2, 16, 12, 6, 3, 9, 4, 2, 1, 12, 6, 3, 9, 9, 4, 2, 1, 12, 16, 3, 9, 4, 2, 1, 12, 6, 3, 9, 4, 4, 2, 1, 12, 6, 3, 16, 4, 2, 1, 12, 6, 3, 9, 4, 2, 2, 1, 12, 6, 3, 9, 4, 16, 1, 12, 6, 3, 9, 4, 2, 1, 1, 12, 6, 3, 9, 4, 2, 1, 16, 6, 3, 9, 4, 2, 1, 12, 12, 6, 3, 9, 4, 2, 1, 12, 6, 16, 9, 4, 2, 1, 12, 6, 6, 3, 9, 4, 2, 1, 12, 6, 3, 9, 16, 2, 1, 12, 6, 3, 3, 9, 4, 2, 1, 12, 6, 3, 9, 4, 2, 16, 12, 6, 3, 9, 9, 4, 2, 1, 12, 6, 3, 9, 4, 2, 1, 12, 16, 3, 9, 4, 4, 2, 1, 12, 6, 3, 9, 4, 2, 1, 12, 6, 3, 16, 4, 2, 2, 1, 12, 6, 3, 9, 4, 2, 1, 12, 6, 3, 9, 4, 16, 1, 1, 12, 6, 3, 9, 4, 2, 1, 12, 6, 3, 9, 4, 2, 1, 16};
static uint32_t dims5[2] = {32, 512};
static uint32_t dims7[1] = {32};
static uint32_t dims9[2] = {8, 512};
static uint32_t dims11[2] = {8, 32};

static void* guardPlace(const void* src, size_t elem, int n, int nvalid, int padMode, size_t* outBytes) {
  size_t bytes = elem * n, pg = 4096, tot = ((bytes + pg - 1) / pg) * pg;
  char* m = (char*)mmap(NULL, tot + pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  mprotect(m + tot, pg, PROT_NONE);
  char* p = m + tot - bytes;
  for (int i = 0; i < n; i++) {
    char* d = p + i * elem;
    if (i < nvalid) memcpy(d, (const char*)src + i * elem, elem);
    else if (padMode == 0) memset(d, 0, elem);
    else if (padMode == 1) memcpy(d, (const char*)src + (i % nvalid) * elem, elem);
    else if (padMode == 2) memset(d, 0xff, elem);
    else if (padMode == 3) { for (size_t k = 0; k < elem; k++) d[k] = (char)rand(); }
    else if (padMode == 4) { if (!seeded_) { srand(getpid() ^ (unsigned)(size_t)&seeded_); seeded_ = 1; } for (size_t k = 0; k < elem; k++) d[k] = (char)rand(); }
    else if (padMode == 6) { memcpy(d, (const char*)src + (i % nvalid) * elem, elem); if (i == envInt("HUGE_IDX", 100)) { float f = (float)atof(getenv("HUGE_VAL") ? getenv("HUGE_VAL") : "1e10"); memcpy(d, &f, 4); } }
    else if (padMode == 7) { memcpy(d, (const char*)src + (i % nvalid) * elem, elem); if (i == envInt("HUGE_IDX", 100)) { int32_t o = envInt("OFF", 5); memcpy(d + 4, &o, 4); } }
    else if (padMode == 8) { memcpy(d, (const char*)src + (i % nvalid) * elem, elem); if (!seeded_) { srand(getpid() ^ (unsigned)(size_t)&seeded_); seeded_ = 1; } int32_t r = rand(); memcpy(d, &r, 4); }
    else if (padMode == 9) { memcpy(d, (const char*)src + (i % nvalid) * elem, elem); if (!seeded_) { srand(getpid() ^ (unsigned)(size_t)&seeded_); seeded_ = 1; } int32_t r = rand(); memcpy(d + 4, &r, 4); }
    else if (padMode == 5) { uint64_t v = (uint64_t)(size_t)p ^ ((uint64_t)rand() << 4); memcpy(d, &v, elem < 8 ? elem : 8); }
  }
  return p;
}

static int buildOnce(int iter, const char* outPrefix) {
  Qnn_ContextHandle_t ctx;
  CHECK(qnn.contextCreate(backend, device, NULL, &ctx));
  QnnHtpGraph_CustomConfig_t vtcm = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  vtcm.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE; vtcm.vtcmSizeInMB = envInt("VTCM_MB", 8);
  QnnGraph_Config_t gVtcm = QNN_GRAPH_CONFIG_INIT; gVtcm.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM; gVtcm.customConfig = &vtcm;
  QnnHtpGraph_CustomConfig_t opt = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  opt.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
  opt.optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG; opt.optimizationOption.floatValue = 3.0f;
  QnnGraph_Config_t gOpt = QNN_GRAPH_CONFIG_INIT; gOpt.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM; gOpt.customConfig = &opt;
  const QnnGraph_Config_t* gcfg[] = {&gVtcm, &gOpt, NULL};
  Qnn_GraphHandle_t graph;
  CHECK(qnn.graphCreate(ctx, "LinearPerBlock", gcfg, &graph));
  Qnn_QuantizeParams_t r0; memset(&r0, 0, sizeof(r0)); r0.encodingDefinition = QNN_DEFINITION_DEFINED; r0.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET; r0.scaleOffsetEncoding.scale = 1.0f; r0.scaleOffsetEncoding.offset = 0;
  Qnn_QuantizeParams_t r2; memset(&r2, 0, sizeof(r2)); r2.encodingDefinition = QNN_DEFINITION_DEFINED; r2.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET; r2.scaleOffsetEncoding.scale = 1.0f; r2.scaleOffsetEncoding.offset = 0;
  static Qnn_BlockwiseExpansion_t bw4; memset(&bw4, 0, sizeof(bw4)); bw4.axis = envInt("AXIS", 0); bw4.scaleOffsets = so4; if (getenv("GUARD_SO")) bw4.scaleOffsets = (Qnn_ScaleOffset_t*)guardPlace(so4, sizeof(Qnn_ScaleOffset_t), envInt("GUARD_SO", 32), envInt("NBLK", 16) == 16 ? 32 : 32, envInt("PAD", 0), NULL); bw4.numBlocksPerAxis = envInt("NBLK", 16); bw4.blockScaleBitwidth = 4; bw4.blockScaleStorageType = (Qnn_BlockwiseExpansionBlockScaleStorageType_t)0; bw4.blocksScale8 = bs4; if (getenv("HEAP")) { Qnn_ScaleOffset_t* hs = (Qnn_ScaleOffset_t*)malloc(sizeof(so4)); memcpy(hs, so4, sizeof(so4)); bw4.scaleOffsets = hs; uint8_t* hb = (uint8_t*)malloc(32 * bw4.numBlocksPerAxis); memcpy(hb, bs4, 32 * bw4.numBlocksPerAxis); bw4.blocksScale8 = hb; } if (getenv("GUARD_BS")) bw4.blocksScale8 = (uint8_t*)guardPlace(bs4, 1, envInt("GUARD_BS", 512), 32 * envInt("NBLK", 16), envInt("PADBS", 1), NULL);
  Qnn_QuantizeParams_t r4; memset(&r4, 0, sizeof(r4)); r4.encodingDefinition = QNN_DEFINITION_DEFINED; r4.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION; r4.blockwiseExpansion = &bw4;
  Qnn_Tensor_t r5 = QNN_TENSOR_INIT; r5.version = QNN_TENSOR_VERSION_1; r5.v1.name = "b__frozen_param0"; r5.v1.type = (Qnn_TensorType_t)4; r5.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER; r5.v1.dataType = (Qnn_DataType_t)776; r5.v1.memType = QNN_TENSORMEMTYPE_RAW; r5.v1.quantizeParams = r4; r5.v1.rank = 2; r5.v1.dimensions = dims5;
  size_t n5; void* d5 = readFile(recPath("blob0.bin"), &n5); r5.v1.clientBuf.data = d5; r5.v1.clientBuf.dataSize = (uint32_t)n5;
  CHECK(qnn.tensorCreateGraphTensor(graph, &r5));
  Qnn_QuantizeParams_t r6; memset(&r6, 0, sizeof(r6)); r6.encodingDefinition = QNN_DEFINITION_DEFINED; r6.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET; r6.scaleOffsetEncoding.scale = 0.0f; r6.scaleOffsetEncoding.offset = 0;
  Qnn_Tensor_t r7 = QNN_TENSOR_INIT; r7.version = QNN_TENSOR_VERSION_1; r7.v1.name = "b__frozen_param1"; r7.v1.type = (Qnn_TensorType_t)4; r7.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER; r7.v1.dataType = (Qnn_DataType_t)818; r7.v1.memType = QNN_TENSORMEMTYPE_RAW; r7.v1.quantizeParams = r6; r7.v1.rank = 1; r7.v1.dimensions = dims7;
  size_t n7; void* d7 = readFile(recPath("blob1.bin"), &n7); r7.v1.clientBuf.data = d7; r7.v1.clientBuf.dataSize = (uint32_t)n7;
  CHECK(qnn.tensorCreateGraphTensor(graph, &r7));
  Qnn_QuantizeParams_t r8; memset(&r8, 0, sizeof(r8)); r8.encodingDefinition = QNN_DEFINITION_DEFINED; r8.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET; r8.scaleOffsetEncoding.scale = 1.3696509995497763e-05f; r8.scaleOffsetEncoding.offset = 0;
  Qnn_Tensor_t r9 = QNN_TENSOR_INIT; r9.version = QNN_TENSOR_VERSION_1; r9.v1.name = "x_u16"; r9.v1.type = (Qnn_TensorType_t)0; r9.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER; r9.v1.dataType = (Qnn_DataType_t)1046; r9.v1.memType = QNN_TENSORMEMTYPE_RAW; r9.v1.quantizeParams = r8; r9.v1.rank = 2; r9.v1.dimensions = dims9;
  CHECK(qnn.tensorCreateGraphTensor(graph, &r9));
  Qnn_QuantizeParams_t r10; memset(&r10, 0, sizeof(r10)); r10.encodingDefinition = QNN_DEFINITION_DEFINED; r10.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET; r10.scaleOffsetEncoding.scale = 0.00027984799817204475f; r10.scaleOffsetEncoding.offset = -28339;
  Qnn_Tensor_t r11 = QNN_TENSOR_INIT; r11.version = QNN_TENSOR_VERSION_1; r11.v1.name = "out_u16"; r11.v1.type = (Qnn_TensorType_t)1; r11.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER; r11.v1.dataType = (Qnn_DataType_t)1046; r11.v1.memType = QNN_TENSORMEMTYPE_RAW; r11.v1.quantizeParams = r10; r11.v1.rank = 2; r11.v1.dimensions = dims11;
  CHECK(qnn.tensorCreateGraphTensor(graph, &r11));
  { Qnn_Tensor_t in_[] = {r9, r5, r7}; Qnn_Tensor_t out_[] = {r11}; Qnn_OpConfig_t op = QNN_OPCONFIG_INIT; op.version = QNN_OPCONFIG_VERSION_1; op.v1.name = "boltnn_quantized_linear_tensor_op10"; op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW; op.v1.typeName = QNN_OP_FULLY_CONNECTED; op.v1.numOfInputs = 3; op.v1.inputTensors = in_; op.v1.numOfOutputs = 1; op.v1.outputTensors = out_; if (!getenv("NOVAL")) CHECK(qnn.backendValidateOpConfig(backend, op)); CHECK(qnn.graphAddNode(graph, op)); }
  CHECK(qnn.graphFinalize(graph, NULL, NULL));
  Qnn_ContextBinarySize_t sz; CHECK(qnn.contextGetBinarySize(ctx, &sz));
  unsigned char* bin = (unsigned char*)malloc(sz); Qnn_ContextBinarySize_t wr;
  CHECK(qnn.contextGetBinary(ctx, bin, sz, &wr));
  static const unsigned char pat[16] = {0x75,0x39,0,0,0,0,0,0,0x4a,0x74,0,0,0,0,0,0};
  int found = 0;
  for (Qnn_ContextBinarySize_t i = 0; i + 16 <= wr; i++) if (!memcmp(bin + i, pat, 16)) { found = 1; break; }
  if (outPrefix) { char fn[512]; snprintf(fn, sizeof fn, "%s_%03d.bin", outPrefix, iter); FILE* f = fopen(fn, "wb"); fwrite(bin, 1, wr, f); fclose(f); }
  free(bin);
  CHECK(qnn.contextFree(ctx, NULL));
  return found;
}
int main(int argc, char** argv) {
  const char* lib = argv[1]; int n = argc > 2 ? atoi(argv[2]) : 20; const char* outPrefix = argc > 3 ? argv[3] : NULL;
  void* h = dlopen(lib, RTLD_NOW | RTLD_GLOBAL); if (!h) { fprintf(stderr, "%s\n", dlerror()); return 1; }
  typedef Qnn_ErrorHandle_t (*GetProviders)(const QnnInterface_t***, uint32_t*);
  GetProviders gp = (GetProviders)dlsym(h, "QnnInterface_getProviders");
  const QnnInterface_t** prov; uint32_t np; CHECK(gp(&prov, &np));
  qnn = prov[0]->QNN_INTERFACE_VER_NAME;
  CHECK(qnn.backendCreate(NULL, NULL, &backend));
  QnnHtpDevice_CustomConfig_t soc; memset(&soc, 0, sizeof soc); soc.option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC; soc.socModel = envInt("SOC", 75);
  QnnHtpDevice_CustomConfig_t arch; memset(&arch, 0, sizeof arch); arch.option = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH; arch.arch.deviceId = 0; arch.arch.arch = (QnnHtpDevice_Arch_t)envInt("ARCH", 79);
  QnnDevice_Config_t dSoc = QNN_DEVICE_CONFIG_INIT; dSoc.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM; dSoc.customConfig = &soc;
  QnnDevice_Config_t dArch = QNN_DEVICE_CONFIG_INIT; dArch.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM; dArch.customConfig = &arch;
  const QnnDevice_Config_t* dcfg[] = {&dSoc, &dArch, NULL};
  CHECK(qnn.deviceCreate(NULL, dcfg, &device));
  int ok = 0;
  for (int i = 0; i < n; i++) { int f = buildOnce(i, outPrefix); ok += f; printf("%c", f ? 'P' : 'F'); fflush(stdout); }
  printf("\nbias present in %d/%d context binaries\n", ok, n);
  return 0;
}
