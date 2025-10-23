// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251023"
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
    "sha256": "b4d090f6d465dadee3da1acd8794104d1fafbf345609fda7a5efcaa21e448bd7",
    "sha256" + debug_suffix: "fc9b45b3e17a54761ed4757db286670485db33fb0fedf06ca5b7718ff4fca71f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7762b139889175a327d5b721fc93d4bf61f2171680bf628e985a57e954201ec6",
    "sha256" + debug_suffix: "953c97b118f2f0a9073e433aa95da01881504541371047300bae342e7232ae52",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f48ead35092e36b6464bae6ad8c72574c45771d173994825e5a97619cba48a36",
    "sha256" + debug_suffix: "414fe1b73233e31e2c608ac03f98b09a8295b345284aaa7393c92cdc43c63eae",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "9a876231759d1fc52520bb97d1d5fe992eed1dcb4c03a791f103c523b510dbaf",
    "sha256" + debug_suffix: "546236f0d8466b4cc5c1cda9e00fda14b5667efdac6a38db77c9647db2a1fc8b",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "20afced5812ef7c23d7ff8020a62dfd6a3eecc90232db521d88ff777e7bd7fa5",
    "sha256" + debug_suffix: "884f2dc55284019f0ed6fd6faa8095c2319d09babfec032ac399d70061705568",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "692b970bdbeb386a8ba39e79f73781f03b4d0c902745d6ed4041739382c50541",
    "sha256" + debug_suffix: "107a174aa426e8541c536cfa57a543222385d576f9c5ffb86cd2de10a4f07ab9",
  ],
  "kernels_optimized": [
    "sha256": "4a9e5fb2c4dfdbdb54efee7395d88e1e14981af92932f688c9bc646aaac92f99",
    "sha256" + debug_suffix: "29e07d4b6147f47dd274dd7aa45c4a635f9585e51036d52fcbc3629a6d724753",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "dde483d812a5df301d919f6c94bff57b1899917f6917b4865dd33cd33e4691c9",
    "sha256" + debug_suffix: "4869cfbd532806cd7c7de95cee127db0f56dbc92ebb48bce8d5b736a596d701e",
  ],
  "kernels_torchao": [
    "sha256": "29c9aaa412c9c5ce9752ded20057dffa05424313e58de082539072d7c8b983ee",
    "sha256" + debug_suffix: "1cf0a69f04825997755cfb05292ef4ab6e7faa0cb3138d9249bfc31604be86eb",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "ca8a0f254e296e18db6f4474ee194f454f6274afa287bc2e2f7d88dcc8c9f589",
    "sha256" + debug_suffix: "c695c68dbb5b2fb6c5ff926b99a932ce8a82663a30629abee42e3761f3507c08",
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
