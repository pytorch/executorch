// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260307"
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
    "sha256": "17fe07d4b12aba1b016e28985a03f8990710cdc32d7fcbbc7464363f3885845c",
    "sha256" + debug_suffix: "a3fafb6ae232dea80a41386693c0ffafdb791062768304b9c70c7f9972c1a9a8",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "1c50d4572f051396ca736ef50b8030656e2d13106d062e43b4e1c1f3c32244ff",
    "sha256" + debug_suffix: "da9bda41678a5e30b1fd860f3545607feb76733d1434f64c296646df43d5ae41",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e24a1a4f2999eaa655d2deb08e6c9f8f2c10ff8ff46f3e666d45e623eeef5a56",
    "sha256" + debug_suffix: "033583c12f8ec5d184473a8f1bc70ff3927de025b56b5d49e2583b17de6f9c82",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "9594fd73bcf8a1695adaf3ab6e07426a3f9d28b6fd404cf7bc1d6eb50740b9d0",
    "sha256" + debug_suffix: "459f5017cd4b70532093be477ff6e9e9f011e320f993e1bb33e0ea399404a074",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "5bce44654deef1bc9cb1057ddf1f1b21465cdc99e5d203c087e37e31247a9503",
    "sha256" + debug_suffix: "f8636c30c6def57efb8c2be1527e6a2104b0652937417d7c4825a5dda2f36bce",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "51cc8d1c45d913f021a81a625093462beda5c1019c35d41d5f9087aa4fd98e0a",
    "sha256" + debug_suffix: "3864259ceb711ceb1d183174f489ccb9bd4d6d2da9f901fb530952f161657dc6",
  ],
  "kernels_optimized": [
    "sha256": "ed33138c62175c8fd21941b7681e6670d7b4aea8ac79433b94ca36ba12d9cc27",
    "sha256" + debug_suffix: "e9d8e05d4defa1f68eb56964d3fa2ffd629f34ff1379b9c6eec0ce55a8fa6b4f",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "50cb77696d195df5ce24cbb3a93a28238103ee039b1fc888f5c4bcc966c0b038",
    "sha256" + debug_suffix: "f77301b8e9a14745bb59e91917613ddf6b207ef46f12494d21f93071cb2efa59",
  ],
  "kernels_torchao": [
    "sha256": "610721217fcb152593ccf84f6394a36575c13a931fe98626917ffa6439505f11",
    "sha256" + debug_suffix: "f87ae3081972fb2ec7d8b7ea6fa9e4d5bcb3e1ccb80f93b5308f39f1a02c3f19",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a2815f2015bdc23a6f3f392b07aa74d793e98f02ea63d2d9ea87945023b67103",
    "sha256" + debug_suffix: "9abbf256b756ed65cd50c5390a540ad911fc9c411b5f0bc6c2a5be23f37a1a6c",
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
