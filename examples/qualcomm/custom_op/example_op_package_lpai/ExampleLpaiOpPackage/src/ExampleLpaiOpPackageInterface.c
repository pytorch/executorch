//==============================================================================
// Auto Generated Code for ExampleLpaiOpPackage
//==============================================================================

#include <string.h>
#include <stdio.h>
#include <stdbool.h>

#include "QnnOpPackage.h"
#include "QnnLpaiOpPackage.h"
#include "QnnLpaiOpPackageInfrastructure.h"

extern QnnOpPackage_OperationInfo_t get_ExampleCustomOp_opInfo_Inference();
extern QnnOpPackage_OperationInfo_t get_ExampleCustomOp_opInfo_Compiler();

// ---------------------------------------------------------------------------
// Op Package Info (derived from Opdef XML)
// ---------------------------------------------------------------------------

#define CUSTOM_OP_COUNT 1

static char sg_packageName[] = "ExampleLpaiOpPackage";
static const char* sg_opNames[CUSTOM_OP_COUNT] = {"ExampleCustomOp"};
static QnnOpPackage_OperationInfo_t sg_opInfos[CUSTOM_OP_COUNT];
static QnnOpPackage_Info_t sg_packageInfo = QNN_OP_PACKAGE_INFO_INIT;

// Global data
// NOTE: sg_globalInfra is intentionally non-static so that per-op inference
QnnLpaiOpPackage_GlobalInfrastructure_t sg_globalInfra = QNN_LPAI_OP_PACKAGE_GLOBAL_INFRASTRUCTURE_INIT;
bool sg_packageInitialized = false;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static QnnOpPackage_OperationInfo_t getOpInfo(const char* opName) {
    for (size_t i = 0; i < CUSTOM_OP_COUNT; i++) {
        if (strcmp(opName, sg_opInfos[i]->opType) == 0) {
            return sg_opInfos[i];
        }
    }
    return NULL;
}

// ---------------------------------------------------------------------------
// QNN OpPackage APIs
// ---------------------------------------------------------------------------

/**
 * @note Things to do here for LPAI backend:
 *   1. Store the global infrastructure so inference ops can use it
 *   2. Register each op's info (inference or compiler mode)
 */
Qnn_ErrorHandle_t ExampleLpaiOpPackageInit(QnnOpPackage_GlobalInfrastructure_t infrastructure) {
    if (sg_packageInitialized) {
        return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
    }

    // QnnOpPackage_GlobalInfrastructure_t is a pointer type and the backend is
    // allowed to hand over something the package must reject, hence the
    // QNN_OP_PACKAGE_ERROR_INVALID_INFRASTRUCTURE error code. The compiler side
    // of an LPAI op package does not use the infrastructure accessors, so keep
    // the zero initialized default instead of dereferencing a null pointer.
    if (infrastructure != NULL) {
        sg_globalInfra = *infrastructure;
    }

#if defined(LPAI_INFERENCE_ONLY)
    sg_opInfos[0] = get_ExampleCustomOp_opInfo_Inference();
#else
    sg_opInfos[0] = get_ExampleCustomOp_opInfo_Compiler();
#endif

    sg_packageInitialized = true;
    return QNN_OP_PACKAGE_NO_ERROR;
}

/**
 * @note Things to do here for LPAI backend:
 *   1. Release all allocated memory of opPackage
 *   2. Destroy globalInfrastructure
 */
Qnn_ErrorHandle_t ExampleLpaiOpPackageTerminate(void) {
    if (!sg_packageInitialized) {
        return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    }

    // NOTE: QnnOpPackage_OperationInfo_t is a *pointer* typedef, and each entry
    // points at a statically allocated struct owned by the per-op source file.
    // Only the pointers are cleared here; memset'ing sizeof(the struct) into
    // &sg_opInfos[i] would write past the end of this array.
    for (size_t i = 0; i < CUSTOM_OP_COUNT; i++) {
        sg_opInfos[i] = NULL;
    }

    sg_packageInitialized = false;
    return QNN_OP_PACKAGE_NO_ERROR;
}

/**
 * @note Things to do here for LPAI backend:
 *   1. Return opPackage_Info (packageName, op type names, count, opInfos).
 *      This information is passed to lower-level modules.
 */
Qnn_ErrorHandle_t ExampleLpaiOpPackageGetInfo(const QnnOpPackage_Info_t** info) {
    if (!sg_packageInitialized) {
        return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    }
    if (!info) {
        return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    }

    sg_packageInfo.packageName    = sg_packageName;
    sg_packageInfo.operationInfo  = sg_opInfos;
    sg_packageInfo.numOperations  = CUSTOM_OP_COUNT;
    sg_packageInfo.operationNames = sg_opNames;

    *info = &sg_packageInfo;
    return QNN_OP_PACKAGE_NO_ERROR;
}

#ifndef LPAI_INFERENCE_ONLY
/**
 * @note Things to do here for LPAI backend:
 *   1. Validate opConfig against the package opdef (counts match)
 *   2. Delegate to op-specific validateOp callback for deeper checks
 */
Qnn_ErrorHandle_t ExampleLpaiOpPackageValidateOpConfig(Qnn_OpConfig_t opConfig) {
    if (!sg_packageInitialized) {
        return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    }

    // A validation failure otherwise surfaces to the user only as a generic
    // "not supported" from the partitioner, so the reason is reported here.

    // Validate package name
    if (!opConfig.v1.packageName || strcmp(opConfig.v1.packageName, sg_packageName) != 0) {
        fprintf(stderr, "ExampleCustomOp: unexpected package name %s\n",
                opConfig.v1.packageName ? opConfig.v1.packageName : "(null)");
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }

    // Validate operation type exists
    const char* operationType = opConfig.v1.typeName;
    if (!operationType) {
        fprintf(stderr, "ExampleCustomOp: null operation type name\n");
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }

    QnnOpPackage_OperationInfo_t opInfo = getOpInfo(operationType);
    if (!opInfo) {
        fprintf(stderr, "ExampleCustomOp: unknown operation type %s\n", operationType);
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }

    // Validate tensor and parameter counts
    if (opConfig.v1.numOfInputs != opInfo->numOfInputs) {
        fprintf(stderr, "%s: expected %u inputs, got %u\n",
                operationType, opInfo->numOfInputs, opConfig.v1.numOfInputs);
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }
    if (opConfig.v1.numOfOutputs != opInfo->numOfOutputs) {
        fprintf(stderr, "%s: expected %u outputs, got %u\n",
                operationType, opInfo->numOfOutputs, opConfig.v1.numOfOutputs);
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }
    if (opConfig.v1.numOfParams != opInfo->numOfParams) {
        fprintf(stderr, "%s: expected %u params, got %u\n",
                operationType, opInfo->numOfParams, opConfig.v1.numOfParams);
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }

    // Validate input datatypes: each input is checked against its own supported type list
    if (opInfo->inputDatatypes) {
        for (uint32_t i = 0; i < opConfig.v1.numOfInputs; i++) {
            Qnn_DataType_t dtype = opConfig.v1.inputTensors[i].v1.dataType;
            bool found = false;
            for (uint32_t j = 0; j < opInfo->inputDatatypes[i].numOfTypes; j++) {
                if (dtype == opInfo->inputDatatypes[i].DataTypes[j]) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                fprintf(stderr, "%s: unsupported input[%u] data type 0x%x\n",
                        operationType, i, (unsigned)dtype);
                return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
            }
        }
    }

    // Validate output datatypes: each output is checked against its own supported type list
    if (opInfo->outputDatatypes) {
        for (uint32_t i = 0; i < opConfig.v1.numOfOutputs; i++) {
            Qnn_DataType_t dtype = opConfig.v1.outputTensors[i].v1.dataType;
            bool found = false;
            for (uint32_t j = 0; j < opInfo->outputDatatypes[i].numOfTypes; j++) {
                if (dtype == opInfo->outputDatatypes[i].DataTypes[j]) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                fprintf(stderr, "%s: unsupported output[%u] data type 0x%x\n",
                        operationType, i, (unsigned)dtype);
                return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
            }
        }
    }

    // Delegate to op-specific validation if provided
    if (opInfo->validateOp) {
        Qnn_ErrorHandle_t ret = opInfo->validateOp(opConfig);
        if (ret != QNN_OP_PACKAGE_NO_ERROR) {
            return ret;
        }
    }

    return QNN_OP_PACKAGE_NO_ERROR;
}
#endif  // LPAI_INFERENCE_ONLY

// ---------------------------------------------------------------------------
// Logging stubs (extend as needed)
// ---------------------------------------------------------------------------

// These accept the backend's logging callback and deliberately discard it: the
// diagnostics in this package go to stderr through fprintf instead, which keeps
// them visible on the host without depending on the caller's log level. A
// production op package would keep the callback and route its messages through
// it so that they land in the QNN log alongside everything else.

/** @brief Set callback for logging and configure log level. */
Qnn_ErrorHandle_t ExampleLpaiOpPackageLogInitialize(QnnLog_Callback_t callback, QnnLog_Level_t maxLogLevel) {
    return QNN_OP_PACKAGE_NO_ERROR;
}

/** @brief Change the active log level. */
Qnn_ErrorHandle_t ExampleLpaiOpPackageLogSetLevel(QnnLog_Level_t maxLogLevel) {
    return QNN_OP_PACKAGE_NO_ERROR;
}

/** @brief Clear the logging callback and reset log level. */
Qnn_ErrorHandle_t ExampleLpaiOpPackageLogTerminate(void) {
    return QNN_OP_PACKAGE_NO_ERROR;
}

// ---------------------------------------------------------------------------
// Interface Provider (QNN OpPackage v1.4)
// ---------------------------------------------------------------------------


Qnn_ErrorHandle_t ExampleLpaiOpPackageInterfaceProvider(QnnOpPackage_Interface_t* interface) {
    if (!interface) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    interface->interfaceVersion      = (Qnn_Version_t)QNN_OP_PACKAGE_API_VERSION_1_4_0;
    interface->v1_4.init             = ExampleLpaiOpPackageInit;
    interface->v1_4.terminate        = ExampleLpaiOpPackageTerminate;
    interface->v1_4.getInfo          = ExampleLpaiOpPackageGetInfo;
#ifndef LPAI_INFERENCE_ONLY
    interface->v1_4.validateOpConfig = ExampleLpaiOpPackageValidateOpConfig;
#else
    interface->v1_4.validateOpConfig = NULL;
#endif
    interface->v1_4.createOpImpl     = NULL;
    interface->v1_4.freeOpImpl       = NULL;
    interface->v1_4.logInitialize    = ExampleLpaiOpPackageLogInitialize;
    interface->v1_4.logSetLevel      = ExampleLpaiOpPackageLogSetLevel;
    interface->v1_4.logTerminate     = ExampleLpaiOpPackageLogTerminate;
    return QNN_OP_PACKAGE_NO_ERROR;
}
