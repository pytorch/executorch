// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250604"
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
    "sha256": "e156a82aa2dca3a09f23a81056546f06d75cee0b9b88596eeee2a6a81553f3c5",
    "sha256" + debug_suffix: "2f29bec49763f3d077ac7188cdc1d38124237206e9618a27bac48a455a9879cf",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "d4162b45a5085b83b9d82c0da42f1efbd7c99a7be33af6e5524f57015576d3ac",
    "sha256" + debug_suffix: "e181c358ebf8de6458ef5f35827276317e13bdd677ef1166110aeceb3cd0f2fa",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "eaf7437ab0cb659d3e7fbd54e4fb0dcde560afb2c08463b2173173233ea57b5c",
    "sha256" + debug_suffix: "cbcb9e80d5a1f2dfa111843b8d48fd02773409ed4b7a4d13ea230c99fed78033",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "3e763c3449c7a927697680d682be5be51e5ac88d225feb67e711280a4031f875",
    "sha256" + debug_suffix: "cfcc46a9ad4004efd16a227e08e46999e3dc6ad90b202ebfd8e9c963bf541f0d",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "85de448ab2a0e7132400437005e847019d8fd844480ed8a7bdf7c4cbfefe61e5",
    "sha256" + debug_suffix: "2979ebd0ce58534ce2232ac89a19697b73039bf664958716e52ff1d82a8139ab",
  ],
  "kernels_optimized": [
    "sha256": "9c06b5536aedc4c2fdf7ea2ba607a08c0142e2330661dedbf604c4ca078b4c01",
    "sha256" + debug_suffix: "a8925b2361a5d2172042059479257c2c5b339895b8e660165d02a8a5f5d92f4b",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "fea71b11334f74c705eb3d64d6eb4b966d11e6fbe9c233762c59aad4fe12acf5",
    "sha256" + debug_suffix: "a954b995fb0d60434f851accf14d17f926f34fd2b3b37ec8de6304d3e11e1dc3",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "bbf06e73dbe9f9cf35602739155bed15ecd0dcba0569d245f7ca8cf66a65a0b1",
    "sha256" + debug_suffix: "5c5c4c1680d40f3e30ebbacb23fcdfbdc99867c5c6a9f94c9030aa9054a04b1a",
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
