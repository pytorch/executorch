// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260816"
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
    "sha256": "cce0a84ebb035815e1dc533e09c2c678ec06412fe112b9136dc6e3ba666b4fbc",
    "sha256" + debug_suffix: "c9a3f31b23aa339a69edcec3473960b939a514ebf800eb1039e9a4a5c80539f3",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b845101135c47446f3171264c25ad98fc56265dcc42a09efb0ff65f2cc0c7a66",
    "sha256" + debug_suffix: "3f8fa77efc8fba26febcd9376b0d48f65c5d81743feecff390e3b9000f7c1024",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "14c0330878f073f1707106a05dafb67d09b1b4ef0a0586d54e33d0714949bcaa",
    "sha256" + debug_suffix: "384808a66ee4aba183278396ebcd3a2d503ca0e14982aaae0a3578d2c5d58ad5",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "8aa7ee56c7c69a1ebb4cb359ef3c20283a8d29e7c34c155e13740285bdb183f8",
    "sha256" + debug_suffix: "81ed8371fb5827c178eabf7fb851c6925a78b7145a25547226a1423952f9f55c",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "fae572d02c470f9a6562787c8d0c966ab091fa856ed2f60f0037da6ac4e2d05c",
    "sha256" + debug_suffix: "9362cf63fc3d05dce8a35d69dd0383ebed4fcf61b3d5d4c581b49f12ab80665a",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "e1574557ef899b3dadec562701c4bf738885c590d17c508de92a294cbc5cc467",
    "sha256" + debug_suffix: "41278714548e1dc1cc91da9fc24bf97db3dfc2fcecfc55258a83ecf63c0690a3",
  ],
  "kernels_optimized": [
    "sha256": "efecf27679b82a27304531a3e97958f04224d5eee15aced081c548711ffebcae",
    "sha256" + debug_suffix: "d5f49f8aaa642bfb38df8db212ce4b71242c04687c63614a7fe33b4cbeebd683",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "79eb45980d6cfe21b14b85f1273b3d8eea9380997630db3501b91907a6d8b7c3",
    "sha256" + debug_suffix: "31d6d4966549ebcdad917a09b819a2d4c3f335666e3b9aaba0e8f45b348e70ca",
  ],
  "kernels_torchao": [
    "sha256": "e16faec9f4c540dbbd42be7d0d33647515144fef1cf71a61006257f25d71c939",
    "sha256" + debug_suffix: "0c019e4afb0dfcafa4b477d8966231a6401922c0964f34952b0c4141d78f4c0d",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "5164cc3880f743ec421fc2f5c0a1000d17cfebda5280e4072508b87b65cf7e6e",
    "sha256" + debug_suffix: "77f128651bbaea6e2865292b8b62f7d85ab794a7be7951d4481965ebf0878d82",
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
