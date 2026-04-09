// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260409"
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
    "sha256": "89a05011352da269db7b6028301ee0857e26bf15108695f767275f772e6338ee",
    "sha256" + debug_suffix: "80237ef1d723082fc877e7cc06a4cbddaf14367dfecb5a3371127e7fdde9a4f4",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "0478da392cea4deca4834871c1acb23c6416f90857ed2fe33383e5694596301c",
    "sha256" + debug_suffix: "a8646973496539dd0ae066c55623f134fa5fe1a7dced6684271caeefc3b4d75d",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ceb640d199fb3f148fc702a4fbe3a58084c0d45f762e3502529d611aba304839",
    "sha256" + debug_suffix: "5924ff36f4b30e4723ec3d4afc9cc2ec0e726e23904c3e52d136b208e56bc0bc",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "1c1683e9e51f130f85e003f032f2a82ff90edd4db68ae2bcbf79827303584692",
    "sha256" + debug_suffix: "bfbc3cf413d51ede49b2c2b6eaee7014fd3d2e406ed982783bc1f2eaa72a9363",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "da070ca28d5dbd3536821704ec804665927f05801c1701d369fc971f33a12670",
    "sha256" + debug_suffix: "d17f83aea6ed0129034c0c316adf2793ba5e0b87b3216abea38d2c7bfaa4a735",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "cd8e50777c2333e383ec0c507d42a8b8cf27a73f74c5a8b6cc8caa248c777318",
    "sha256" + debug_suffix: "1749a37923324807e1471bb29b0006724bc02144b909ab8390a8822e4e1515f3",
  ],
  "kernels_optimized": [
    "sha256": "cff31d4b7e624f0dace71cc7ad4a81c0afcb0b2352de77a8cf6ff8cf3167f2a0",
    "sha256" + debug_suffix: "97802d48eab9dcdbf8b89a136699453e6b76c50648a623d70e30ce7968ddba5c",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "dbc9a33619067addd40935fb85953a1ea379ef4616004fa5ad9eba0a51b3e550",
    "sha256" + debug_suffix: "fe18367336e24b385a887bf8b1054228420ff3221693bf5243ba79e3eb2c617e",
  ],
  "kernels_torchao": [
    "sha256": "6a2c260d4ab08d6c00c828ed66894fe33723cf6ecfa743734774744619c00b30",
    "sha256" + debug_suffix: "aa52622179dca8237de20014e8f33adc29c77a5c857a15f5ea814dad7d6b835d",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "25fe0eb7c3d21f031535297f7b6da187e69d635816d17487f0a386da59bf2684",
    "sha256" + debug_suffix: "0835b17f2855e7786da5b88ba5d23775ee775bf70562402aab16a6047b172695",
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
