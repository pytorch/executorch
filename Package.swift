// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260313"
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
    "sha256": "6c7b7e6b4f9eea50eb6ceda427e185215002f19d287563b1df0204141cbbb803",
    "sha256" + debug_suffix: "8a2b447840df1610622a478d0be864cb265c4384a3d35ca9b9581c951e5fd37c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "24712bd3c1bd62f9c1a25efb00444cfba41c97a4a1928e84e7da10e2c3024eba",
    "sha256" + debug_suffix: "48aa55769b5b3677fd298c25093d686f5787daf410d722d7e94785f8a9f4d074",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "0c73845e51414215608e2c83da27e70883354d91362cc9df5669b2cd0884d22e",
    "sha256" + debug_suffix: "8e2f8f7248055ae2ff98ad4d73f929c86cdd66aa5ec630ecf0a5867d996dc02c",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "7bf64a6633fe2bb1f3d9fa65914c53a19ea21b08c9d1a70070c8d2170cff8095",
    "sha256" + debug_suffix: "205767c633ca34d3136c70e3aadc2576d821771c8a4211354db5705b675a675c",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "0a4ec402316c167e4b13c4c00cde51bbd9f9c5cd1a5bf26a2961f4cb38e2fa69",
    "sha256" + debug_suffix: "3d598c27347e253225d6910f153a784081b0332a9e7d22c6b2b370220bc3665c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "96ec4d41d5af408731acd953e36c40bc65a5085d4c46d2757dd702b287da67b7",
    "sha256" + debug_suffix: "9bcf5004035a264416e9b338e8e1bba1eae26d937913edfa60220e85e5073665",
  ],
  "kernels_optimized": [
    "sha256": "539f90f0d1a3c933e4ee83b30b87765fe1faef013ad3cde3111fc74ff1dd072f",
    "sha256" + debug_suffix: "bcbe23ec0fee19ed1847d806b4211432f99d147b144a37fcc7065f7fc6cb3bdb",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "34ac4a4c3408be93a490c80098fe9bf864de374a920cc6ef48d6e585b8e9dd58",
    "sha256" + debug_suffix: "dee21a7e8fc5e2987885acbad8b88adcde29f4993d951d4dce1892752e766d9d",
  ],
  "kernels_torchao": [
    "sha256": "8c5775197e1c95823318a793c0f740048ddcfb021dd6ac2dc6eb248ad4185f3c",
    "sha256" + debug_suffix: "2a7be833c9322d6dad1b213a8d473b2a53ddbe1993d271d50447f9b2ecd99327",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "c33e3bfa733c1ed079b8f83782049ce33fc8768f4820431513a7771a38e95f49",
    "sha256" + debug_suffix: "72b48735c6565fe073b2c68932d89297d0edc7a621f092b9b566b50ab2fdb049",
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
