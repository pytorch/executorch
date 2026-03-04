// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260304"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug_suffix = "_debug"
let dependencies_suffix = "_with_dependencies"

func deliverables(_ dict: [String: [String: Any]]) -> [String: [String: Any]] {
  dict
    .reduce(into: [String: [String: Any]]()) { result, pair in
      let (key, value) = pair
      result[key] = value
      result[key + debug_suffix] = value
    }
    .reduce(into: [String: [String: Any]]()) { result, pair in
      let (key, value) = pair
      var newValue = value
      if key.hasSuffix(debug_suffix) {
        for (k, v) in value where k.hasSuffix(debug_suffix) {
          let trimmed = String(k.dropLast(debug_suffix.count))
          newValue[trimmed] = v
        }
      }
      result[key] = newValue.filter { !$0.key.hasSuffix(debug_suffix) }
    }
}

let products = deliverables([
  "backend_coreml": [
    "sha256": "9ec594d2b97a2ba8202cf5dc57cf2c05751464146e6e42e49eaf6d188edc5267",
    "sha256" + debug_suffix: "67bc24f3f9e26b9ae5cee461a9e77016c4463ec1bd6d3684531e0a8e8278879e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "f0b6b871e596d55502fbce489e8569ef5d3aa4e6b24f509f214a574682264835",
    "sha256" + debug_suffix: "dbe95b7aa24baa6fb9e88979e3df4b6e817ff9d4c47df37080b5e0fc06481fab",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "458db17391d4eff09bc0fb2b652e3ebfc6219bcf81ba1165e8bdf90b43c147f0",
    "sha256" + debug_suffix: "524c291054e8d8876500e56a92f98622943d0be5d1e6de255cf164b27ea6e9f9",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "b17c9a15db89caa3046d0a88a919d69a0963d8ce984ec1d37f6c3735766a5b0d",
    "sha256" + debug_suffix: "60f4378d9b46254312a58ee7057e222576c5f828da471e7d9433d8a1a33b9623",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "aff010af6f6c62565b1d49c134bc82e933b8359c0a2ade5bb8615cf6ef26e021",
    "sha256" + debug_suffix: "dcb5fd68fb2088d55d8858272478a6d95e7e3a9a7022f09a9cca5ae87d5307b1",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "21563b06ff7aa2be9daf1c5b8fcb67e258382469591f0a6d9180a4177a3b4ad6",
    "sha256" + debug_suffix: "b236d074ca48d7dc9484307735069dbeb06ca281ac653c7842d84a470e8bc615",
  ],
  "kernels_optimized": [
    "sha256": "c0c3434a4dd624fcf5b07978e80493cac0cc165ebe40dc5e39f0623210cbab05",
    "sha256" + debug_suffix: "9c8eae1b0af7b1ee21d7985b42f5745e91b59f9dceacac5c27577a414b8aa943",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "786f942a3d4590d6e4fae6c1e0db4e2009639783c1817d7f41c4c3d6c22e4914",
    "sha256" + debug_suffix: "151d8fe4bb640867664d3f6c182cffcbcb50558e60da24cb7c24a44ac7f6658e",
  ],
  "kernels_torchao": [
    "sha256": "30579bfe0c50d56bee9b36a1c8a48323a88da50c9e5b3a34e2ea81dea2745494",
    "sha256" + debug_suffix: "dbbe0d619600da51a12c7d07e5c7ef2045b30a3d43605e61ec6291e807c64659",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "47560c6825af7ee33ce59977083df53e170800b71d43f3ce2eaac701d501dc09",
    "sha256" + debug_suffix: "54f9844372df866002c428a0cf4b10351711d31506f1f0b6a0caa6b9553c00f1",
  ],
])

let packageProducts: [Product] = products.keys.map { key -> Product in
  .library(name: key, targets: ["\(key)\(dependencies_suffix)"])
}.sorted { $0.name < $1.name }

var packageTargets: [Target] = []

for (key, value) in targets {
  packageTargets.append(.binaryTarget(
    name: key,
    url: "\(url)\(key)-\(version).zip",
    checksum: value["sha256"] as? String ?? ""
  ))
}

for (key, value) in products {
  packageTargets.append(.binaryTarget(
    name: key,
    url: "\(url)\(key)-\(version).zip",
    checksum: value["sha256"] as? String ?? ""
  ))
  let target: Target = .target(
    name: "\(key)\(dependencies_suffix)",
    dependencies: ([key] + (value["targets"] as? [String] ?? []).map {
      key.hasSuffix(debug_suffix) ? $0 + debug_suffix : $0
    }).map { .target(name: $0) },
    path: ".Package.swift/\(key)",
    linkerSettings:
      (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
      (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
  )
  packageTargets.append(target)
}

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v12),
  ],
  products: packageProducts,
  targets: packageTargets
)
