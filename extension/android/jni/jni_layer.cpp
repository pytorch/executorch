/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/portable_type/tensor_impl.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/util/util.h>

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

#ifdef __ANDROID__
#include <android/log.h>

// For Android, write to logcat
void et_pal_emit_log_message(
    et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    const char* function,
    size_t line,
    const char* message,
    size_t length) {
    int android_log_level = ANDROID_LOG_UNKNOWN;
    if (level == 'D') {
        android_log_level = ANDROID_LOG_DEBUG;
    } else if (level == 'I') {
        android_log_level = ANDROID_LOG_INFO;
    } else if (level == 'E') {
        android_log_level = ANDROID_LOG_ERROR;
    } else if (level == 'F') {
        android_log_level = ANDROID_LOG_FATAL;
    }
  __android_log_print(android_log_level, "ExecuTorch", "%s", message);
}
#endif

using namespace torch::executor;

namespace executorch_jni {

    constexpr static int kTensorDTypeUInt8 = 0;
    constexpr static int kTensorDTypeInt8 = 1;
    constexpr static int kTensorDTypeInt16 = 2;
    constexpr static int kTensorDTypeInt32 = 3;
    constexpr static int kTensorDTypeInt64 = 4;
    constexpr static int kTensorDTypeHalf = 5;
    constexpr static int kTensorDTypeFloat = 6;
    constexpr static int kTensorDTypeDouble = 7;
    constexpr static int kTensorDTypeComplexHalf = 8;
    constexpr static int kTensorDTypeComplexFloat = 9;
    constexpr static int kTensorDTypeComplexDouble = 10;
    constexpr static int kTensorDTypeBool = 11;
    constexpr static int kTensorDTypeQint8 = 12;
    constexpr static int kTensorDTypeQuint8 = 13;
    constexpr static int kTensorDTypeQint32 = 14;
    constexpr static int kTensorDTypeBFloat16 = 15;
    constexpr static int kTensorDTypeQuint4x2 = 16;
    constexpr static int kTensorDTypeQuint2x4 = 17;
    constexpr static int kTensorDTypeBits1x8 = 18;
    constexpr static int kTensorDTypeBits2x4 = 19;
    constexpr static int kTensorDTypeBits4x2 = 20;
    constexpr static int kTensorDTypeBits8 = 21;
    constexpr static int kTensorDTypeBits16 = 22;

    class TensorHybrid : public facebook::jni::HybridClass<TensorHybrid> {
    public:
        constexpr static const char* kJavaDescriptor =
                "Lorg/pytorch/executorch/Tensor;";

        explicit TensorHybrid(exec_aten::Tensor tensor) {}

        static facebook::jni::local_ref<TensorHybrid::javaobject>
        newJTensorFromTensor(const exec_aten::Tensor& tensor) {
            // Java wrapper currently only supports contiguous tensors.

            const auto scalarType = tensor.scalar_type();
            int jdtype = 0;

            if (ScalarType::Byte == scalarType) {
                jdtype = kTensorDTypeUInt8;
            } else if (ScalarType::Char == scalarType) {
                jdtype = kTensorDTypeInt8;
            } else if (ScalarType::Short == scalarType) {
                jdtype = kTensorDTypeInt16;
            } else if (ScalarType::Int == scalarType) {
                jdtype = kTensorDTypeInt32;
            } else if (ScalarType::Long == scalarType) {
                jdtype = kTensorDTypeInt64;
            } else if (ScalarType::Half == scalarType) {
                jdtype = kTensorDTypeHalf;
            } else if (ScalarType::Float == scalarType) {
                jdtype = kTensorDTypeFloat;
            } else if (ScalarType::Double == scalarType) {
                jdtype = kTensorDTypeDouble;
            } else if (ScalarType::ComplexHalf == scalarType) {
                jdtype = kTensorDTypeComplexHalf;
            } else if (ScalarType::ComplexFloat == scalarType) {
                jdtype = kTensorDTypeComplexFloat;
            } else if (ScalarType::ComplexDouble == scalarType) {
                jdtype = kTensorDTypeComplexDouble;
            } else if (ScalarType::Bool == scalarType) {
                jdtype = kTensorDTypeBool;
            } else if (ScalarType::QInt8 == scalarType) {
                jdtype = kTensorDTypeQint8;
            } else if (ScalarType::QUInt8 == scalarType) {
                jdtype = kTensorDTypeQuint8;
            } else if (ScalarType::QInt32 == scalarType) {
                jdtype = kTensorDTypeQint32;
            } else if (ScalarType::BFloat16 == scalarType) {
                jdtype = kTensorDTypeBFloat16;
            } else if (ScalarType::QUInt4x2 == scalarType) {
                jdtype = kTensorDTypeQuint4x2;
            } else if (ScalarType::QUInt2x4 == scalarType) {
                jdtype = kTensorDTypeQuint2x4;
            } else if (ScalarType::Bits1x8 == scalarType) {
                jdtype = kTensorDTypeBits1x8;
            } else if (ScalarType::Bits2x4 == scalarType) {
                jdtype = kTensorDTypeBits2x4;
            } else if (ScalarType::Bits4x2 == scalarType) {
                jdtype = kTensorDTypeBits4x2;
            } else if (ScalarType::Bits8 == scalarType) {
                jdtype = kTensorDTypeBits8;
            } else if (ScalarType::Bits16 == scalarType) {
                jdtype = kTensorDTypeBits16;
            } else {
                facebook::jni::throwNewJavaException(
                        facebook::jni::gJavaLangIllegalArgumentException,
                        "exec_aten::Tensor scalar type is not supported on java side");
            }

            const auto& tensorShape = tensor.sizes();
            std::vector<jlong> tensorShapeVec;
            for (const auto& s : tensorShape) {
                tensorShapeVec.push_back(s);
            }
            facebook::jni::local_ref<jlongArray> jTensorShape =
                    facebook::jni::make_long_array(tensorShapeVec.size());
            jTensorShape->setRegion(0, tensorShapeVec.size(), tensorShapeVec.data());

            static auto cls = TensorHybrid::javaClassStatic();
            // Note: this is safe as long as the data stored in tensor is valid; the
            // data won't go out of scope as long as the Method for the inference is
            // valid and there is no other inference call. Java layer picks up this
            // value immediately so the data is valid.
            facebook::jni::local_ref<facebook::jni::JByteBuffer> jTensorBuffer =
                    facebook::jni::JByteBuffer::wrapBytes(
                            (uint8_t*)tensor.data_ptr(), tensor.nbytes());
            jTensorBuffer->order(facebook::jni::JByteOrder::nativeOrder());

            static const auto jMethodNewTensor =
                    cls->getStaticMethod<facebook::jni::local_ref<TensorHybrid::javaobject>(
                            facebook::jni::alias_ref<facebook::jni::JByteBuffer>,
                            facebook::jni::alias_ref<jlongArray>,
                            jint,
                            facebook::jni::alias_ref<jhybriddata>)>("nativeNewTensor");
            return jMethodNewTensor(
                    cls, jTensorBuffer, jTensorShape, jdtype, makeCxxInstance(tensor));
        }

    private:
        friend HybridBase;
    };

    class JEValue : public facebook::jni::JavaClass<JEValue> {
    public:
        constexpr static const char* kJavaDescriptor =
                "Lorg/pytorch/executorch/EValue;";

        constexpr static int kTypeCodeTensor = 1;
        constexpr static int kTypeCodeString = 2;
        constexpr static int kTypeCodeDouble = 3;
        constexpr static int kTypeCodeInt = 4;
        constexpr static int kTypeCodeBool = 5;

        static facebook::jni::local_ref<JEValue> newJEValueFromEValue(EValue evalue) {
            if (evalue.isTensor()) {
                static auto jMethodTensor =
                        JEValue::javaClassStatic()
                                ->getStaticMethod<facebook::jni::local_ref<JEValue>(
                                        facebook::jni::local_ref<TensorHybrid::javaobject>)>("from");
                return jMethodTensor(
                        JEValue::javaClassStatic(),
                        TensorHybrid::newJTensorFromTensor(evalue.toTensor()));
            } else if (evalue.isInt()) {
                static auto jMethodTensor =
                        JEValue::javaClassStatic()
                                ->getStaticMethod<facebook::jni::local_ref<JEValue>(
                                        jlong)>("from");
                return jMethodTensor(JEValue::javaClassStatic(), evalue.toInt());
            }  else if (evalue.isDouble()) {
                static auto jMethodTensor =
                        JEValue::javaClassStatic()
                                ->getStaticMethod<facebook::jni::local_ref<JEValue>(
                                        jdouble)>("from");
                return jMethodTensor(JEValue::javaClassStatic(), evalue.toDouble());
            } else if (evalue.isBool()) {
                static auto jMethodTensor =
                        JEValue::javaClassStatic()
                                ->getStaticMethod<facebook::jni::local_ref<JEValue>(
                                        jboolean)>("from");
                return jMethodTensor(JEValue::javaClassStatic(), evalue.toBool());
            } else if (evalue.isString()) {
                static auto jMethodTensor =
                        JEValue::javaClassStatic()
                                ->getStaticMethod<facebook::jni::local_ref<JEValue>(
                                        facebook::jni::local_ref<jstring>)>("from");
                std::string str = std::string(evalue.toString().begin(), evalue.toString().end());
                return jMethodTensor(JEValue::javaClassStatic(), facebook::jni::make_jstring(str));
            }
            facebook::jni::throwNewJavaException(
                    facebook::jni::gJavaLangIllegalArgumentException,
                    "Unsupported EValue type: %d",
                    evalue.tag);
        }

        static TensorImpl JEValueToTensorImpl(
                facebook::jni::alias_ref<JEValue> JEValue,
                std::vector<exec_aten::SizesType>& shapeVec,
                std::vector<uint8_t>& temp_dim_order_storage,
                std::vector<int32_t>& temp_strides_storage) {
            static const auto typeCodeField =
                    JEValue::javaClassStatic()->getField<jint>("mTypeCode");
            const auto typeCode = JEValue->getFieldValue(typeCodeField);
            if (JEValue::kTypeCodeTensor == typeCode) {
                static const auto jMethodGetTensor =
                        JEValue::javaClassStatic()
                                ->getMethod<facebook::jni::alias_ref<TensorHybrid::javaobject>()>(
                                        "toTensor");
                auto jtensor = jMethodGetTensor(JEValue);

                static auto cls = TensorHybrid::javaClassStatic();
                static const auto dtypeMethod = cls->getMethod<jint()>("dtypeJniCode");
                jint jdtype = dtypeMethod(jtensor);

                static const auto shapeField = cls->getField<jlongArray>("shape");
                auto jshape = jtensor->getFieldValue(shapeField);

                static auto dataBufferMethod = cls->getMethod<
                        facebook::jni::local_ref<facebook::jni::JBuffer::javaobject>()>(
                        "getRawDataBuffer");
                facebook::jni::local_ref<facebook::jni::JBuffer> jbuffer =
                        dataBufferMethod(jtensor);

                const auto rank = jshape->size();

                const auto shapeArr = jshape->getRegion(0, rank);
                shapeVec.reserve(rank);
                temp_dim_order_storage.reserve(rank);
                temp_strides_storage.reserve(rank);

                auto numel = 1;
                for (int i = 0; i < rank; i++) {
                    shapeVec.push_back(shapeArr[i]);
                }
                for (int i = 0; i < rank; i++) {
                    temp_dim_order_storage.push_back(i);
                }
                for (int i = rank - 1; i >= 0; --i) {
                    temp_strides_storage[i] = numel;
                    numel *= shapeArr[i];
                }
                JNIEnv* jni = facebook::jni::Environment::current();
                ScalarType scalar_type;
                if (kTensorDTypeFloat == jdtype) {
                    scalar_type = ScalarType::Float;
                } else if (kTensorDTypeInt32 == jdtype) {
                    scalar_type = ScalarType::Int;
                } else if (kTensorDTypeInt8 == jdtype) {
                    scalar_type = ScalarType::Char;
                } else if (kTensorDTypeUInt8 == jdtype) {
                    scalar_type = ScalarType::Byte;
                } else if (kTensorDTypeDouble == jdtype) {
                    scalar_type = ScalarType::Double;
                } else if (kTensorDTypeInt64 == jdtype) {
                    scalar_type = ScalarType::Long;
                } else if (kTensorDTypeHalf == jdtype) {
                    scalar_type = ScalarType::Half;
                } else if (kTensorDTypeComplexHalf == jdtype) {
                    scalar_type = ScalarType::ComplexHalf;
                } else if (kTensorDTypeComplexFloat == jdtype) {
                    scalar_type = ScalarType::ComplexFloat;
                } else if (kTensorDTypeComplexDouble == jdtype) {
                    scalar_type = ScalarType::ComplexDouble;
                } else if (kTensorDTypeBool == jdtype) {
                    scalar_type = ScalarType::Bool;
                } else if (kTensorDTypeQint8 == jdtype) {
                    scalar_type = ScalarType::QInt8;
                } else if (kTensorDTypeQuint8 == jdtype) {
                    scalar_type = ScalarType::QUInt8;
                } else if (kTensorDTypeQint32 == jdtype) {
                    scalar_type = ScalarType::QInt32;
                } else if (kTensorDTypeBFloat16 == jdtype) {
                    scalar_type = ScalarType::BFloat16;
                } else if (kTensorDTypeQuint4x2 == jdtype) {
                    scalar_type = ScalarType::QUInt4x2;
                } else if (kTensorDTypeQuint2x4 == jdtype) {
                    scalar_type = ScalarType::QUInt2x4;
                } else if (kTensorDTypeBits1x8 == jdtype) {
                    scalar_type = ScalarType::Bits1x8;
                } else if (kTensorDTypeBits2x4 == jdtype) {
                    scalar_type = ScalarType::Bits2x4;
                } else if (kTensorDTypeBits4x2 == jdtype) {
                    scalar_type = ScalarType::Bits4x2;
                } else if (kTensorDTypeBits8 == jdtype) {
                    scalar_type = ScalarType::Bits8;
                } else if (kTensorDTypeBits16 == jdtype) {
                    scalar_type = ScalarType::Bits16;
                } else {
                    facebook::jni::throwNewJavaException(
                            facebook::jni::gJavaLangIllegalArgumentException,
                            "Unknown Tensor jdtype %d",
                            jdtype);
                }
                const auto dataCapacity = jni->GetDirectBufferCapacity(jbuffer.get());
                if (dataCapacity != numel) {
                    facebook::jni::throwNewJavaException(
                            facebook::jni::gJavaLangIllegalArgumentException,
                            "Tensor dimensions(elements number:%d inconsistent with buffer capacity(%d)",
                            numel,
                            dataCapacity);
                }
                return TensorImpl(
                        scalar_type,
                        shapeVec.size(),
                        shapeVec.data(),
                        jni->GetDirectBufferAddress(jbuffer.get()),
                        temp_dim_order_storage.data(),
                        temp_strides_storage.data(),
                        TensorShapeDynamism::DYNAMIC_UNBOUND);
            }
            facebook::jni::throwNewJavaException(
                    facebook::jni::gJavaLangIllegalArgumentException,
                    "Unknown EValue typeCode %d",
                    typeCode);
        }
    };

    class ExecuTorchJni : public facebook::jni::HybridClass<ExecuTorchJni> {
    private:
        friend HybridBase;
        std::unique_ptr<torch::executor::Module> module_;

    public:
        constexpr static auto kJavaDescriptor =
                "Lorg/pytorch/executorch/NativePeer;";

        static facebook::jni::local_ref<jhybriddata> initHybrid(
                facebook::jni::alias_ref<jclass>,
                facebook::jni::alias_ref<jstring> modelPath,
                facebook::jni::alias_ref<
                facebook::jni::JMap<facebook::jni::JString, facebook::jni::JString>>
                extraFiles) {
            return makeCxxInstance(modelPath, extraFiles);
        }

        ExecuTorchJni(
                facebook::jni::alias_ref<jstring> modelPath,
                facebook::jni::alias_ref<
                facebook::jni::JMap<facebook::jni::JString, facebook::jni::JString>>
                extraFiles) {
            module_ = std::make_unique<torch::executor::Module>(modelPath->toStdString(), torch::executor::Module::MlockConfig::NoMlock);
        }

        facebook::jni::local_ref<JEValue> forward(
                facebook::jni::alias_ref<
                facebook::jni::JArrayClass<JEValue::javaobject>::javaobject>
                jinputs) {
            return execute("forward", jinputs);
        }

        facebook::jni::local_ref<JEValue> runMethod(
                facebook::jni::alias_ref<jstring> methodName,
                facebook::jni::alias_ref<
                facebook::jni::JArrayClass<JEValue::javaobject>::javaobject>
                jinputs) {
            return execute(methodName->toStdString(), jinputs);
        }

        facebook::jni::local_ref<JEValue> execute(
                std::string method,
                facebook::jni::alias_ref<
                facebook::jni::JArrayClass<JEValue::javaobject>::javaobject>
                jinputs) {

            std::vector<EValue> evalues = {};

            std::vector<TensorImpl*> tensor_impl_buffer;
            std::vector<std::vector<exec_aten::SizesType>*> shapeVec_buffer;
            std::vector<std::vector<uint8_t>*> temp_dim_order_storage_buffer;
            std::vector<std::vector<int32_t>*> temp_strides_storage_buffer;

            static const auto typeCodeField =
                    JEValue::javaClassStatic()->getField<jint>("mTypeCode");

            for (int i = 0; i < jinputs->size(); i++) {
                auto jevalue = jinputs->getElement(i);
                const auto typeCode = jevalue->getFieldValue(typeCodeField);
                if (typeCode == JEValue::kTypeCodeTensor) {
                    std::vector<exec_aten::SizesType>* shapeVec = new std::vector<exec_aten::SizesType>;
                    shapeVec_buffer.push_back(shapeVec);
                    std::vector<uint8_t>* temp_dim_order_storage = new std::vector<uint8_t>;
                    temp_dim_order_storage_buffer.push_back(temp_dim_order_storage);
                    std::vector<int32_t>* temp_strides_storage = new std::vector<int32_t>;
                    temp_strides_storage_buffer.push_back(temp_strides_storage);
                    auto* tensor_impl = new TensorImpl(JEValue::JEValueToTensorImpl(jevalue, *shapeVec, *temp_dim_order_storage, *temp_strides_storage));
                    evalues.emplace_back(EValue(exec_aten::Tensor(tensor_impl)));
                    tensor_impl_buffer.push_back(tensor_impl);
                } else if (typeCode == JEValue::kTypeCodeInt) {
                    int64_t value = jevalue->getFieldValue(typeCodeField);
                    evalues.emplace_back(EValue(value));
                } else if (typeCode == JEValue::kTypeCodeDouble) {
                    double value = jevalue->getFieldValue(typeCodeField);
                    evalues.emplace_back(EValue(value));
                } else if (typeCode == JEValue::kTypeCodeBool) {
                    bool value = jevalue->getFieldValue(typeCodeField);
                    evalues.emplace_back(EValue(value));
                }
            }

#ifdef EXECUTORCH_ANDROID_PROFILING
            auto start = std::chrono::high_resolution_clock::now();
            auto result = module_->execute(method, evalues);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            ET_LOG(Debug, "Execution time: %lld ms.", duration);
#else
            auto result = module_->execute(method, evalues);
#endif

            ET_CHECK_MSG(
                    result.ok(),
                    "Execution of method %s failed with status 0x%" PRIx32,
                    method.c_str(),
                    result.error());
            ET_LOG(Info, "Model executed successfully.");

            for (auto* buffer: tensor_impl_buffer) {
                delete buffer;
            }
            for (auto* buffer: shapeVec_buffer) {
                delete buffer;
            }
            for (auto* buffer: temp_dim_order_storage_buffer) {
                delete buffer;
            }
            for (auto* buffer: temp_strides_storage_buffer) {
                delete buffer;
            }

            return JEValue::newJEValueFromEValue(result.get()[0]);
        }

        static void registerNatives() {
            registerHybrid({
                                   makeNativeMethod("initHybrid", ExecuTorchJni::initHybrid),
                                   makeNativeMethod("forward", ExecuTorchJni::forward),
                                   makeNativeMethod("runMethod", ExecuTorchJni::runMethod),
                           });
        }
    };

} // namespace executorch_jni

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    return facebook::jni::initialize(
            vm, [] { executorch_jni::ExecuTorchJni::registerNatives(); });
}
