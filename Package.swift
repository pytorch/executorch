// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260123"
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
    "sha256": "26f5cd5f489536e2dec23a8beedd289be424927bba9224cb127a76a0b2e689ad",
    "sha256" + debug_suffix: "212e1b21463dff89693e1249f3f79dfc343e4b27de55a0eb88664328fdc90248",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "ce5470d77932e6ace167baa1c79bb8242153bebaac97ccbcd36866499abe3140",
    "sha256" + debug_suffix: "901b29c961e18b2b09e7f015d7939e73cf453b3885a4b8f1a79779b5fadf6cc1",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "8e83ea3cc14a4c9f0a975f763eeb5b1739aa48d0aa59e08c55cfe40bb2c3022e",
    "sha256" + debug_suffix: "d9da2c7af4f485594040c1c46849575c5e088d4a201ee0630e304d04dd8167a3",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "6201d344c9ec2828b6a469b3fe47bb4a24ba8b46d875c4fe32fc97d902061dda",
    "sha256" + debug_suffix: "9b7541d1d857029f9255f0c544af1755c091dfacb76cdc5e904b4abf8c53609f",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e8e5609bf701fa4ee5c3bb90867d6104b7679aff7fbd41bcc40ad16ff7287807",
    "sha256" + debug_suffix: "5b519ba507fce9052d539ae4bd757a064cef18987e6cb3b88442c7a10515f6e6",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "f3015cab803ee7eae852c54e643e9b906608333fb69ef3613bc66e1e8964814a",
    "sha256" + debug_suffix: "299d88d2fa8c44a99cb4ae75b7742479594c6cb36a4c70bafbece4edb671b75e",
  ],
  "kernels_optimized": [
    "sha256": "53f877ac9b95191267ce1a4ea0aee4245bc68e043c123547e7ca0308314bb189",
    "sha256" + debug_suffix: "70ce2c97ae4cea0dff0673d0b81c3d03cb72cccbf330673e61ee6c1365897d51",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "5912038c39c9eadbb76ded6b3c6d47ea1eff2390d6ed094af9470bca9da60a52",
    "sha256" + debug_suffix: "df4b12b14efc2220a660d914f065d528897ec363b8793938df1130470369d9ce",
  ],
  "kernels_torchao": [
    "sha256": "e9676dc66e3f37d79586beb6b243d1e36a950143aaf0070ca6a2919381401eef",
    "sha256" + debug_suffix: "c762184fb2cecfda169a681aaa1cbf297b6380090159090338b3ee9305649e39",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "99f94291fc569e4653945d7a5fca9880708826ac678a76f247aced11dcf94e49",
    "sha256" + debug_suffix: "236c09bee57d35c9307601144a0dbbe7d9b309fb833a3e69ad4e35cb6f9d960e",
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
