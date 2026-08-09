// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260809"
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
    "sha256": "997e17a63fbcd60d71f6ae45d9abca665a48523b26c68380a063a09facb530f4",
    "sha256" + debug_suffix: "2e0c8ad6296fed4559546c8d3cb33df5f8ce9087d10a47c842af28ae60722d34",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "9bc05e401eeb1c9f7bc2097609b3cfc48d8b3552982f23b00f3c5c0f65c65041",
    "sha256" + debug_suffix: "c30bca6727255507f5b98514bf879ceb4266ad756738548aa43777730ad25cbe",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "62de95a0ad079ec672bf2cc55aabf6b11f29c21ba799ae26639e13b7396f4bc2",
    "sha256" + debug_suffix: "7ce8a8aa2518b870f0089ccbe171f38b4c492d9a19313047d9a7aee61e19098b",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "687628d0986ebc890d6a80e76f5061eb176a8df2edd8977df2ccde039fbf7b40",
    "sha256" + debug_suffix: "0b656faa73c5a1676940cd8b85956bf6d484588386bd9efdd9c5ad70badf182f",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "da9c25af8cca5563f8e62d1854b2e1dbfce1adb817ba2015b0f6e3137822291b",
    "sha256" + debug_suffix: "76375ab5dd1958c6e9aa20165d08af2e9017ef45607fd4f62982c38f057447aa",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "36fad9d47d1f7fb26310ad2c137f1e18997d461661e0b587b538d9c89478b5ce",
    "sha256" + debug_suffix: "f4ce4acd2515e8d7aa788ddeed70380303374bfda670d7fc2b129870640b01f4",
  ],
  "kernels_optimized": [
    "sha256": "15444ae96054ed8fc1b607e8ec0c3a57aa2d1b1b36a5f50d29f0ff1add2eee61",
    "sha256" + debug_suffix: "19a843c0f8d12a2eab77a0a1531dc2149f9666c5c7aa3ebde6b7e62e25b47723",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9c670bb025ad0879ce420e50a4cda6cbe53abdda96eef80e22cd974cc0481c56",
    "sha256" + debug_suffix: "59d69321066e4b8355d8841e2f5336b663d88bc3fa7d41ce1b8dc089eda97641",
  ],
  "kernels_torchao": [
    "sha256": "5ed70b07a16afa49b1c15fa32bda6ffe76eb74a41c7295b163de71201b7f67df",
    "sha256" + debug_suffix: "b0241e874ea6a4ae97878e01445a03ef9c19d7219b5fa37e3565a078ba8c8296",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "4a10d10400ebbd3022719692c8fb474fb87cfd11936c14ea03ce0299118b91c1",
    "sha256" + debug_suffix: "e63f0b1218c7f1f44ed6e0a5a3447abf8d083ce23087b429bdb789a049e3ed25",
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
