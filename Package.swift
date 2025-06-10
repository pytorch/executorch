// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250610"
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
    "sha256": "529a84b9f34cdba6a7d0d78a6890986deacc7882346ec44af2cfd71858d8683a",
    "sha256" + debug_suffix: "6b16f9b4114332e610763e27a2068a2d884401a9c493432e01cd696a59dca568",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "e4a4f070c1e7430b992344e04a218a6086047663a81f290ec464c112f746ff97",
    "sha256" + debug_suffix: "82b1568ec3b0b30256bf34091a83317c7c4ce03fa2bc416acf511f15dfe326e8",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d699e8e596feff02c19404f949596b0052157923ab454747a83b960d102424e1",
    "sha256" + debug_suffix: "b43b94c61777cc82918c855840162eff0adf0b73dfa52d03dc99bd28f214fd12",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f0e2837b106cca6ba8573faad623325ef6c13874c20bc1063aa43e2103b90723",
    "sha256" + debug_suffix: "cbff9da06cb73fb973f90a8c6f43aeaa6aa232723f07e85487bb119c585e141f",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "20ab3f307bc24f1b6038ded89d3f0a190187d3cb2ec37c768197ceaa4555ffdd",
    "sha256" + debug_suffix: "8363555cba065a81daa5ebfb5946580ec040219fc31079f229c727790071268f",
  ],
  "kernels_optimized": [
    "sha256": "322e32ea70263620fde03a1945a7df6b792111d775625fb456ed841fa28e672b",
    "sha256" + debug_suffix: "0e92fc81065007fbeb6a6f3761c77de83dd9c21997d1d1679324fccec9808daa",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "de4b60024bea60ece307f270c8d944a9cee567144e87fd62a85287b6ac059d4a",
    "sha256" + debug_suffix: "8a5126f49fcf72b4678e6eda8304a25a03f2c7b92d12c6a805595323d65e1506",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "ee9c425a08791d7a967bde6ed8f8d7df7069b04f27517f728ab4f36dcb03dc8d",
    "sha256" + debug_suffix: "9e5e6ee69a248677832c0c734920f94e58a7a5c724769965edab730839cde293",
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
