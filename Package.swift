// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251211"
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
    "sha256": "0126e790c7b8f971d53879df1521ea90088d65a93444732b7ca5ed4844183df4",
    "sha256" + debug_suffix: "8efa976f18d428f468549d39af2ab464b97ee97b78f1304ee75d22deb2ed91a2",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "22801eb5811b734637e351ff3bebc437c3c03bf42acc9bfc77e09730f9e817d1",
    "sha256" + debug_suffix: "ace7d5e01b361a467b056b3df29b8388166c9ea10d6e4e6440082e2555976a42",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "fa48f95cc1012c3b8708bcc10e7d8dd591814b7e2446a7922d37aea223068a0a",
    "sha256" + debug_suffix: "ea9f547cd95548211802e2b43824da1a46ab83eda3062da82a02491d5bdd4203",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "74975dcafa2847c203ef40c96fc706ac7f5bb92afd02664a07acb5a58eb6f503",
    "sha256" + debug_suffix: "2147cf730fef9a6e0d1dcfb64f9f975d9dcb69368f00255eb5ba629c9a08dc92",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "8a9cbce6116512bc38da39bc60915ac87f0d01e5de3829a37ede67542a2129a4",
    "sha256" + debug_suffix: "326a31169fe9550ccebcdf37d8d9fad741d049ff7577cd41b7a1ed102375a0fa",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c610d4f6aee9c4e1e42fa1753120bad2f8fa62f3acf55fa17b29c40b0c562ec0",
    "sha256" + debug_suffix: "8e44aa2a807ffa0604766bb6c380ce912821ca173177ffe332f9e29a7805bc2c",
  ],
  "kernels_optimized": [
    "sha256": "6e1be0beeb9f671854879bd985aa5892c851926d4a346ad97af2c529cef10b39",
    "sha256" + debug_suffix: "fd927861abdaef1326266674cd15b9fb82c9ce23313964313e4b8c3efc2f572f",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "4a1cf67ab130c7d9163e830345a8432715b7b6694a060cf0333ae35e726da89c",
    "sha256" + debug_suffix: "94c12d4a038ff3c0bb13a900051a8367c75bdad39635a9f5853b4d4395d7644f",
  ],
  "kernels_torchao": [
    "sha256": "50548ee0dad183daf7753f5185d76424a711953ac1bc445c200ea960d207833f",
    "sha256" + debug_suffix: "503595b13cb825912d07f066414761d4326196097201f425085c7da9116686fe",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "be01d1370e7d6a18d3f06c6ad003658514c909b07e606ef42de33180314c595a",
    "sha256" + debug_suffix: "f1ebdbb9995193fb2fff2d73138925d13217474c41f302022ea294fc5e299b5d",
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
