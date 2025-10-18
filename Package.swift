// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251018"
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
    "sha256": "ebadfd9173567ef80eb21376e4fdc99141890f12c1aebfa24d363e77332eee5e",
    "sha256" + debug_suffix: "c54d246953da209624fe41aed73a8ddbf16d5114cdce36909d5bb5b5ccb26138",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "bea2ca5ae84ed121de20d766536a8b85963bd6b0e3e55559520bcfa20cbc0eba",
    "sha256" + debug_suffix: "bfee76e4905056c51d4b41615adcc398c410936f26ab375dda329a90365a8001",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b7829f7a6f1364f0d5ad05b1c5d2d23a1604f28d56430aca2d449e24a49cedfd",
    "sha256" + debug_suffix: "025afacc6f893177aef58be091c5d39d4ac3d7560dfac5b704d7dae462a0eb3c",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "065306b16fa2e1bccb2fa0eac6961c3b463768281aac9f7ec80ca9332278861a",
    "sha256" + debug_suffix: "5f6af800825288e3c4ce9575979bed5c6cb38e7fac3c877b30e025928e68e549",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "2b2d142893452e4afbba6af6f6adafbf6fb367adad7b1f94e3a08cf059396014",
    "sha256" + debug_suffix: "cb70d3ff2b97d5fd5078900eb13c65f5d57c2e0406d1bb9021685b727fc65f43",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "306bd954df95e17d64f15d32a1146343e99ad9cf379686e822b8831e303f7d9c",
    "sha256" + debug_suffix: "8707ac1a3e63fc4d4ae4bbbe9f7c20c70224bcd12cb4ddb423ea54b3e1b8ddb9",
  ],
  "kernels_optimized": [
    "sha256": "8d4a2404ca4880f1ed67a3706cfc3b6db1f39ebfd8457ec1334e847e52cf748f",
    "sha256" + debug_suffix: "13ad42b22d8ef05c2eb7e06fa777a54e673398985b3cdd524f9fea987962d9be",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "2e2641ad7e8d4c22036529c69f0aaf6ec532136572c70d12787de2c0353d5e7e",
    "sha256" + debug_suffix: "aff2d28dda62ea03cce1eb50d3f38c1d66e6516195b65504ec12629d12307850",
  ],
  "kernels_torchao": [
    "sha256": "387656e3a6910a7091445d0f839d91559486dcc5e7a9aba37a1840db899f7b4c",
    "sha256" + debug_suffix: "9633809ca89ff9b29fc096f731775403d90f848082fb26a6d336afde93a64add",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "53d5c69c9d51138c519a14f705e15f24d84a1bea20173da2a69ffce8ec0d74c6",
    "sha256" + debug_suffix: "1924b2dbc0c510016945bfbbc259a0f7d0578279fd2068188d869f531e0dc20f",
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
