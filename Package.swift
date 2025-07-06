// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250706"
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
    "sha256": "3fe82f58e64468d68918de4ddf68f053503a542396d7ecc5e91bd8cfd7a49364",
    "sha256" + debug_suffix: "fa4a21efbf8debb68236b013f8ee6a600198c288394eb5e6ecb70dd9c03779ce",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "ecde9491388cf5d91106619a5e14ae06a48d73f8378ed204c84d3f8035ff2758",
    "sha256" + debug_suffix: "4b49ac067f82cd18ae379634740b8fca5be3ca0dde8d2c208149aba2d2e15e2c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "77c9c4fa8de1cfe1f67a441e60b3d0e30e258690e2e8cc5892590a97245deaee",
    "sha256" + debug_suffix: "869a190fbf2274753e2a4534144cd4dc3e3b50e46c388a5cedc5d66b65707443",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "0b6c6c1a4e069be00f8943fcb78e6f4e9f1f9871021880851ce4bf903f49bb0b",
    "sha256" + debug_suffix: "21f125a30fd53576375f78c5d9e87bb150e05b4a8002abf42a5bbcf19eb1c8df",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "955023e6385f8baaa2650cf7b355d2adff85de3c9a0721ef8df7f5b2ece46e6d",
    "sha256" + debug_suffix: "c0e0409bce67180e4019b023fb124a40b9d3409c2940f47beebad5c580b2b686",
  ],
  "kernels_optimized": [
    "sha256": "6cd03bb7e1f0178e3d4215ae18b0adc8f8aaea6257cb02a0f5dc0d976f1d55f7",
    "sha256" + debug_suffix: "1ec6b606b9c7bfaf66889f1c94edd375457bc61f1f8386e32d6bd92ff54135dc",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "d0d2f1e8200b9063d06e9651570a661c3d38e7b5091a0feda233318e2e8998c7",
    "sha256" + debug_suffix: "f33df21a04daceb04b1ffa720919180cc4a39afd9030f81daa8dee3b8b7e62a8",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "5c7fa84e47fffe40cd2e8da9e95733669396a0b295e39b4ebd506237f1e293a4",
    "sha256" + debug_suffix: "51acd20a4a46e82e7ee87b9be942e8cfeebf6fc4acb74f72d56cf42aea73313b",
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
