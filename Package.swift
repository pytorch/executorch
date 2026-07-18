// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260718"
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
    "sha256": "631c9ea9e9bb6a584ae723cb4dfaac916ff203f1c2bf2743d0bd3f75b0cc6881",
    "sha256" + debug_suffix: "8f825e566206fbc1a76c83d643b30e433dec540cb995a97a8c7c25ac27bccc2b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "3928069b21a1c9bbc85822bdb1ea6e2a1953edda6eb79ff0c9af4d45c16a666e",
    "sha256" + debug_suffix: "3eb9bc335cbfdd755adba0bc7c22f8ee37516925da4f0e3742477fa6ecdc41c5",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "682f6b0a205eb7da445e589858741a1712e8fa73c37899c1aba05076b96a8a4a",
    "sha256" + debug_suffix: "1952198739eb1b185d242560bf7489c45466da103aeb1b6c81a24a5899473fbb",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "a9b065707d12645fb8226cf317246e5e1df5a319f525f7b12f5e51ef506d5192",
    "sha256" + debug_suffix: "35253d78f9ae488d9b74349e9309176e4ce8abee17eef592d4105bc0091d9bd4",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "ede8365880fae6beafd1ad44386de407e1105d84fbe08ca0a484f186488deaf2",
    "sha256" + debug_suffix: "e4fbff53e5ab78ddc3f498604784cb710efde3822a76f15f68e3ed4c943b5c72",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "dcccbc92c5fe54eea2bb745fe12870b65eb56d6cbc13d38000a9078d43dbe155",
    "sha256" + debug_suffix: "1f2f6c316632946a9f3e102a713cadcf90dbdb5faaf022e26d6792732d967299",
  ],
  "kernels_optimized": [
    "sha256": "297de306db31b2f8a365e1b0a6c89c121d4898ae62cf4f98ae043c117bbb5f8a",
    "sha256" + debug_suffix: "1963c75484357e5d5e950ee1f10deae7ec066878873e23af0692c59489024449",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "163117810a80a73c968c443c6f2cd26510f6e721e71bf391707d663801e05e99",
    "sha256" + debug_suffix: "92955f5b8385e0244bfb5719b8e39cf8037cf5edde243c2d223afa1dc01fefa6",
  ],
  "kernels_torchao": [
    "sha256": "da412d4b18ada137f36fa12f75965d206869f37e51efa6f98a731df1a238ddcf",
    "sha256" + debug_suffix: "e586ebb3861bbdf77b26d302414d7100d53db877f94f073b580d52695e75fcc0",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "26a263d4029be87a42b6a2e8a2972eb32e7b15d7c378e4ec040dfe25eb0f1c9c",
    "sha256" + debug_suffix: "747996186ec79cd781cb04bd08dc41d5be3219eddb02f41461cd9d5f2a646c22",
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
