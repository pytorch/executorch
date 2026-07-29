// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260729"
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
    "sha256": "eaff275e81ecbc68ed4ef216d6cb6c9592979e41ee7530c9848c75b6aa044960",
    "sha256" + debug_suffix: "b8511f947b917d3e9ea408228880ff55338290ae8eb0b35a9c31196c3b42a3f0",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "6d49a22d95cb26aa7ef49693087762a5b140e9df6788b8e18d7980ee7ca2e065",
    "sha256" + debug_suffix: "b9ff72a205d9073a1e725c991f39356508155cb89a267e62c6e46b97f8174963",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e5350a5b74f87461ebe20628df7f152297e31f3c2e8f3982b8b3766866bc1818",
    "sha256" + debug_suffix: "2dfc80e58e1013369ab16a36d017549b29741e2babef9da23048c5225f5c7ed4",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "a7ad2f90e540def334ecb81dc754df5435a6dc5d37b4788a5e7d777838e4abf9",
    "sha256" + debug_suffix: "b2f09aeec059da0a620597d7ddd5495d238fbc5c017eb6b53b311e717b4555a3",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "d6d59b8002139b6e7d41be3d1202f37906deff3f44dee58731bdcbf7e9944215",
    "sha256" + debug_suffix: "d4a99641f60cec45effa12b1cf2597d6615c2e5a808df954317b93c588395825",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "ed95161a5dfa872c2b98433ea43cae1c9d3dc63a57abc3ba180547aedc3f9f00",
    "sha256" + debug_suffix: "c8f8a377ad87aa37b1e2b2e90440f26cf8670a229ff22593a38543430085b50c",
  ],
  "kernels_optimized": [
    "sha256": "6d7aa6cef4c4b0c4124fa1663ff3a652b1ff77d07d0f714ca4ab190dc05fea9e",
    "sha256" + debug_suffix: "303203f146f7ab7f6d34116406c2854bd2aaed91973d04919e62271ba4f5ca78",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "7641099f3c315b9c15241c5c8232ab1482bacd2e56c21d4fa87a5a82c2b1bddb",
    "sha256" + debug_suffix: "74ad41e2424a3b73ab811b073f4c040a1d10b16457a7bda9a96389a32a085f1e",
  ],
  "kernels_torchao": [
    "sha256": "6560a7bfc9ccde01b10a32fd6c355b602c2e128d81942a9b597f9223d2e77f8a",
    "sha256" + debug_suffix: "dd11e3a52dd8ccdcdafc483a5908041980b5d37567f2ff73e2e4d4b502aa058d",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "03711f259451af78c270250e3bfbfc3e7c3fa01ef8686aab950a76e64e52d162",
    "sha256" + debug_suffix: "0aa0a770b90fa85d4439fa28da9924612f6737a7a1df33c6e9650a4edc5c2ad8",
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
