// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251004"
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
    "sha256": "5606b0bab3e335574fab2610bec1cfdb2e3d519713f92e75f113407f7ebb1806",
    "sha256" + debug_suffix: "94f4e903d316c9845d44313f7fc78b51c898aae55b881ff463377cb2193843c3",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b0125956a26050cfe2fac2427221ab3653fe3b5e63ea6d30003d7c4d342b2f09",
    "sha256" + debug_suffix: "9d4a3fee993acae4946835a494a63fe4349da9dca84a905953e37eb144d92509",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d25abf9b72048e2d13e33db7fd1aa39132ff3f33625b7a51b0bb433ca67add5d",
    "sha256" + debug_suffix: "ae4d21b84f0fe7f6b210c8b953dce88fff06208f845910296e42dab1c32bb485",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "4bb7bffb7ffcaa61baf76bb8092405da9ce1cefc61c10b56d099ca0cb9074bfb",
    "sha256" + debug_suffix: "6dbdce78b8b9df321b2056379047f5ddfc3dd49b285bce346257f6ace476655b",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "f764aff6f0f2e8e622857292c7de13624a8f5dba3eed0f707bb2c2cf4a4e2d20",
    "sha256" + debug_suffix: "b62e9b13ab6e09ec9af50985c085e9aac0674b679485202772c82a4ee92394dc",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "0a200165f0ef07541db2f8350257be177394558632fd9e77adb67e3d2fbdd998",
    "sha256" + debug_suffix: "8977c351a91ac65cc235731a5067f7faee29738fd58dcd052706936f6a74f310",
  ],
  "kernels_optimized": [
    "sha256": "9b5e8be815bacb12af07d9aa4a0260030bc2f1a52613023565dd6971f0734d0e",
    "sha256" + debug_suffix: "a06f383670356b239efaab77ccef1b9c042a7dbb13ce2ac49e4516883a77b3a3",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "40dafabfd94f7fe3a56b9b19ef69e8d38ffa87b0eb5cc312f78c74e278e962c1",
    "sha256" + debug_suffix: "62bcc1b9952d5f2ef9c48fa7f46c08daa1941fd0c94facfceef74403c9011f05",
  ],
  "kernels_torchao": [
    "sha256": "61d2f0305f14b4d570d50e0900b9ce8bca3eab48fdbee2aaeb6e4480a4ef22d6",
    "sha256" + debug_suffix: "a138cd0b97e7b3395f09ded9f95dedaa04ab652e56cf57e96444c065271b22c8",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "f9566b98b444b7510e1ff78d93eddc95a54e172ca5b69a35bb2a85d479df6e63",
    "sha256" + debug_suffix: "84e48bc2a71615b0f95b9604393c81636e07936bec35c56cfd81df780d639859",
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
