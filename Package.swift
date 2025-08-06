// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250806"
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
    "sha256": "c341544818d08c982ae4f7e5ef2e097e83edc52bd7ff75dc80527abebdfd0f01",
    "sha256" + debug_suffix: "449bbba2e5e1e4264c97983eaa1dc634642867b4166e7e21845af1bc4923d729",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "fb1ba7d6b80ed07e6c5750d5afbd921d9bac31edeb42ff3e4bfb6eaec47152db",
    "sha256" + debug_suffix: "8a40527c52bac1bf83df8eef1f05e78c3a4a2d1f01b4147a18f49fd310dbffb1",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "baf6554953cd378019f3d2c7b1bc38e7675a2f05378140e3a609223110aea2ca",
    "sha256" + debug_suffix: "4fd97e4aa4b6c5f88d3a308e6cf2a2bd42a29a39c7895f99ac857a644492bce5",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "6ff4f5f70e09cbcabcd845aeec3c2b617ebb2a6877c16f45cff4036bb384e72c",
    "sha256" + debug_suffix: "da91abd72ae04f9fe8cc7df90b4f153dd747316fd7c1ebbc4d2ceb515753e467",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "fb94628de64c5b73b2947562bb92077ac95494d55ba835eb2a7e8b8066d994e2",
    "sha256" + debug_suffix: "5020fd082508bbd2a16e1d9416e5e8c3c4468dc2afeb5d97a964046156dd9d71",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "ca9bb6d31c500d0904c94621e453e1c38b53f69af220a0b3168c473f6f1f1d0a",
    "sha256" + debug_suffix: "63b8ddc274d1656c3ed32a28798e8d3a0c53ac670342e168aeeae73a638a4942",
  ],
  "kernels_optimized": [
    "sha256": "b13b3581d1a1257bddeb47e5edff0fd3b85766e9d2ee7ff13d790e0bd1a081e8",
    "sha256" + debug_suffix: "a288a3805e27082faf98b9d071dbe2f15296b917b88c452d96d01143955cb067",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "8e9a5f0ff8205e0dce76dd0992acf9cc98eb421232376a883e6e1fab28773e75",
    "sha256" + debug_suffix: "a8473b61d6dfc6d9acdd1f4ca8a1ca1319d58443a2115430110dd86b330525a1",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "4d6fe8065328af57151ad89d0bd5eb3962edd573404c439e6e59a5c07602f123",
    "sha256" + debug_suffix: "fa68399632fb790d4600a9fae6eb89f98e7ee58d5dd23a6126920ab2f28c2c14",
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
