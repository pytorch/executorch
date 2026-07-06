// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260706"
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
    "sha256": "3aa17bd1a92bc0445cf601a5b18cca92d2d04cf325817b0e8b0a2a16a12cf735",
    "sha256" + debug_suffix: "dbfaf0bffb99ad1ce3c9310bcf38929d7d8a1eaf17696855c6c48004f735b8a3",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "0bae29eba26ad930182d0fa698b63028dc00940086b47a155214302e4a0238ab",
    "sha256" + debug_suffix: "cef821386cad5e84072254fc5b1c2d8c24e7f4902bb7e0c3a3e39dce250f1379",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "39c86c6f483093e1f419cd5231a9343063396310c1229658713157434a0059c1",
    "sha256" + debug_suffix: "84bc7f064f49a4c9d7da8bc3615594f2a50eaf1bc96bb2f5a29ce4e65b02f132",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "783fcef7069efb904614e65057331bc872c25dbac9153ef2bab8fb42fc97f0e2",
    "sha256" + debug_suffix: "2ed0043d155402a2785c02cf8ad4efb85e2e4d3fa35972478bd3211e0971db7d",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "8dc86c764a489f6501866b36d52e8b0cd555bf464f4bf0b3d8e3e3f7c1b4bdc6",
    "sha256" + debug_suffix: "475e8a8f50211a447e33d4e22ac383e3d0eee461ee85cc22ae4e56b0b8fd3a8d",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "2a1d493e67027eb08fc47e3ce2d8b0a5ce2cb1fda572d63fa7febe2893452aa2",
    "sha256" + debug_suffix: "bf8a79dcbbdf9462c4cefdf4baa079579f74e97ef1fba69c4825a3034edc80b1",
  ],
  "kernels_optimized": [
    "sha256": "671aebc6be5eeded8abeaa116a008b8897d641f316aeb6b947cc4c4755db01be",
    "sha256" + debug_suffix: "73cc663cc9dbb8b5e3f027818bf516888e5fceee1f1d8fe75f8ec3debae5a297",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ff6f9a874f421683684ae08f64a96286400c0df734cc128de5ef7f32c10f229b",
    "sha256" + debug_suffix: "cc77b4b29c4364e718cd582db82ba72334300b1095757ac943e64f955dba9815",
  ],
  "kernels_torchao": [
    "sha256": "34b4d712290a26e534ced391aa73117b02a4adfe60fe99f71e203f84c3369749",
    "sha256" + debug_suffix: "624556c84fe047424b9b9e9826ab4bd506f4856d9c3b0ba62252ee2fd5db038b",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "7ad42b9e19f77334bc20babcf611b3a35bfff40988e643e87ac7ed83dd7786cb",
    "sha256" + debug_suffix: "8295a212eff13ac5bee9e260a1350bd26cd45a993a0961b416f23ccc680ec224",
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
