// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260531"
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
    "sha256": "2df69e2638eac2980b3c4ab9e2218e4a4d45a463143ecc3963799fac2a79c1be",
    "sha256" + debug_suffix: "fb2834a11098499aaa76cb296ca6a97f02c8f342481d2d45699e1f90718ca8f9",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "1b09a62b47fa1e56527dc1d7337d0193fff9f7a016a839555af1043ed16ac1e4",
    "sha256" + debug_suffix: "73dcc01f07679e39ca311549acebd5fddbdc198c2c380a2222f99c98416a20f2",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "64c8f102d68e30684bd0899cd65a1e4c38558fee7070ef8a306f007e1db5e3d6",
    "sha256" + debug_suffix: "47dcdd86ecef322041ed032d36fdff2ce488aaf7a6a6caa55471632acaa2c8e9",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "b7e441d527b42c470e590de38f73ebd7802dcc5aa5be43a01fee49e934488492",
    "sha256" + debug_suffix: "c2ab1411d0b567ded1b398bf2fb55d8143f17abdd891d41e5c9ec3a5113f2191",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "8a3705a32f359e53f2f6f87f22a7cbd178e7b77f24149a89422244c1b28ffc73",
    "sha256" + debug_suffix: "ab74fa5b87a6dcf5f4a0bdcc8a7210f19ddf90a284980b6be27a5be5a5be62d3",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "ecb0652b56c2006d78825231a1d767ac48ef2cc0e070c2f489d0e2ac8cbeac56",
    "sha256" + debug_suffix: "b3d46622d557157b4c3426e2bf0651d447e220745184b1a1d6e6005cd3e3cf73",
  ],
  "kernels_optimized": [
    "sha256": "dafd9a648168eb5a68527fd5fdc122d225d1a2ccd70754e892c3d034b0e88919",
    "sha256" + debug_suffix: "31a56e991e451bab834e7df5e5af8775a110ccc4483a1e70012bb0084769042f",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "e1db2a5badc5635a7a3bcc6cb48f5d5e7c080addf799349378a0d40c46343d60",
    "sha256" + debug_suffix: "3a5f2b16483c28602970b74dab82bc0a2717c4334ce9fe229d43719a2a00bac8",
  ],
  "kernels_torchao": [
    "sha256": "a73b739926f3a8a39ada116c0aa12cc22581ae70554438bc4a2c48e1907aaf8a",
    "sha256" + debug_suffix: "3f678beb1a3d229f74a7d437c3f852bc177aeed74f559cbf93ba58c0da07cdb8",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "d580cbdb5ba3ec918e711a5458655565d4c082d65e1c25bf2c1fe7653e445286",
    "sha256" + debug_suffix: "c56f4a9e584a381012a0445868fc48daaa73b217a69aca51b6cf60ed0d7f27c5",
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
