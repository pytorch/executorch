// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250928"
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
    "sha256": "efbaa0d64edbeea629b9af33bfac6527faa2e491a747e0e4f1cddf642ea488f3",
    "sha256" + debug_suffix: "fc1c269dbe411070315349e357c67067fd26ad41bc1809b9c6a74a87bb959348",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "20af8c37d8ebd84820d6629975aefe2ebb14c9346c772280785907b9aa8437b2",
    "sha256" + debug_suffix: "8921f3478fa8ef3d4fb88edb40239876708fa64360954375bd7927f625dee9f6",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e7578d3a2770a0668c9569ef4a133995e83e67833a1ecdb5b3b2ff047779333e",
    "sha256" + debug_suffix: "029154337d309d939cdb080a5b2d27f676561b961675d44a2a42f4c7d6a0a02d",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "1d720577b503570fc3a3a71210861af135e04c619c8a2142cb4b329d2ba2297a",
    "sha256" + debug_suffix: "c2cc15b0802d6839e88c91fa306acb9bd1116a2d1d79ec6796f83e6273899c2b",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "0fe26f215b5337cf1ea8d1f489719a6df5b4bb2fb05615b5f77ef26d0bf98bb5",
    "sha256" + debug_suffix: "74cd4f7f6714d5a728d35d21282a39a523d304f62e8c05dd581c3dbe95100b6e",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "829b0b98e2932a1fa2d1e956e906f22c2fbf3108a4f85e1b0cf32fd01b614d91",
    "sha256" + debug_suffix: "f793682e6482fa9b369d63c2025018029fecdfe4efadd70681e8ea329723cd47",
  ],
  "kernels_optimized": [
    "sha256": "caeb632f51b4e522e5ff514b7be4b5166fff5bcd7658002945cefc0517d9c159",
    "sha256" + debug_suffix: "96b7dfb21c8cc513eec83d62cbe4d591e048d3a5260319ad97cf46468466a7f5",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "37d6f1907ccc3e42dfce59bc3fd81407d92e212469b6b400c9100dd1529d8991",
    "sha256" + debug_suffix: "9a58617efcf92da6478a5aeacac155bd12435aed82162f0dd3542bf8fb9402d7",
  ],
  "kernels_torchao": [
    "sha256": "5f7290d578dc4a7f7222146b20a5268196f85a818a1d30810b71f8d6e715d67b",
    "sha256" + debug_suffix: "d21ee2cd2d9caea2ad2f6e087c696be659598d8f17e97f55a64f81c4ad2a9fd3",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "3b6755ccba7019f789d59839953d71206ae28c003aedc1e7490137d5153d4cd7",
    "sha256" + debug_suffix: "c92f15cb8a3fb2d214eec95164c4e0e361ef4f4de0451ce91cad462983e13aee",
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
