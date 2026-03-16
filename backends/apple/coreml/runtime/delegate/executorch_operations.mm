#pragma once

#include "executorch_operations.h"
#import <coreml_backend/delegate.h>
#import "ETCoreMLStrings.h"
#import "backend_delegate.h"

#import <executorch/runtime/core/evalue.h>
#import <executorch/runtime/platform/log.h>
#import <executorch/runtime/backend/interface.h>

#include <array>
#import <memory>

namespace executorch::core_ml_backend_delegate {
  using executorch::runtime::get_backend_class;

static std::unique_ptr<executorch::backends::coreml::CoreMLBackendDelegate> backendInterfaceLazy_;

void register_backend_coreml() {
    auto backendInterface = executorch::runtime::get_backend_class(ETCoreMLStrings.delegateIdentifier.UTF8String);
    if (backendInterface == nullptr) {
      backendInterfaceLazy_ = std::make_unique<executorch::backends::coreml::CoreMLBackendDelegate>();
      executorch::runtime::Backend backend{ETCoreMLStrings.delegateIdentifier.UTF8String, backendInterfaceLazy_.get()};
      std::ignore = register_backend(backend);
    }
  }

} // namespace executorch::core_ml_backend_delegate
