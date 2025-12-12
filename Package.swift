// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251212"
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
    "sha256": "3e1e79004a061d44168c7e70bff1ea7c408fe7ea9aa02cd1bffa3d4ff5989d19",
    "sha256" + debug_suffix: "d705502aec18c0aedf9c7576c418e5d3173bb4e9b3970e8a01176530389cf305",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7caf9c2745d691d53d704400d13e31cbc17a7d5dbc291d4417267e24c1cb68f5",
    "sha256" + debug_suffix: "dc902f28383f847a106921fe7368d302fd2e19d10f3c74abc7078133d63bc329",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f9fa62e36864838a24ac4145230958feb24097a8f39f583e6c1cc016074e5a12",
    "sha256" + debug_suffix: "52f57855de19ad9b0d22fba07f0b6fec84b95180079536444eeaba5e652716c6",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "072c35f12b98eb7dbc15cfce58d1a55a3830f30268cccbe2a7021b424a75f3f5",
    "sha256" + debug_suffix: "3b7d2543c0cca950b21e8f41ba06e7f31be1e49d533bed93d80ee47cec952624",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "5cb35ba8b343b74db0cf3f35e88b9d6b4aa1c62391566ce06ba18ffec257e504",
    "sha256" + debug_suffix: "1c6a3897e9974e6743608231f27ab3271b141308123ea346d6273cfbcdf3da73",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "24d08e03b3f288382e924eb599432130c5ea673370813a08289ae1eccac8885b",
    "sha256" + debug_suffix: "b9e1744ac3144f2b64d5c3fa7bbab3628502015a9d12d6a20a8744c08c42b6b2",
  ],
  "kernels_optimized": [
    "sha256": "f540fb9fb13c668cca553bbe0d0f63692c06f82e57c5fa5c3b11393b99297af6",
    "sha256" + debug_suffix: "5e6e437d20ed625a12468278cc35a0822b8e73ef891f88f176b9c07020cf888c",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "c71ca92a05dfa3b8374e6e5b72ca02aa73c575968df11b1fe1a76c0aee732bbe",
    "sha256" + debug_suffix: "ccb0398b86f34f6405a4ef53299e1a34926e8d8de16656a5e08ad37db449393f",
  ],
  "kernels_torchao": [
    "sha256": "f6809ef8a4fab3123ccc6620ce14a727411897af172e4e605eb76a5401deb77d",
    "sha256" + debug_suffix: "6260aa4bc4759ca45482e7d9af4bdf211064542a2f7d250ef569acab8401664e",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "971f93130464653c939513eff64d176961f7d8b357205e63470e9a09df80dbc5",
    "sha256" + debug_suffix: "b6cc91c6136f3f2a117cb0abd8ddc75a1bc4c963106674096d54de8d9a11095c",
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
