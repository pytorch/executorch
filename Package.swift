// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260305"
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
    "sha256": "9afb7010890d8f88707c082f9f15c2784b41bf9d9def20f5ddda330b91b2c348",
    "sha256" + debug_suffix: "e5283133fa841299c0a670d74c0cb421d073aa71bd6f296de6e90dd06a452738",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8731ed020b5bbdbd06b1bef532e683fe3a09615b2687809cd50eba994ac790c7",
    "sha256" + debug_suffix: "dec37fbe81ba9c6a00df4f3f91f05708a045b0c387e4334555e5f529091e6db2",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "3d2d2e1dbe8f5701844c6cbc335fe1bdcca2f81b0d3efaaf14d70bbe9ab70dfc",
    "sha256" + debug_suffix: "46c9eb7e56d1eea07faf1ab1c5e6c5b41a75c52e557ad8b057766625e0f2ded4",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "299b5447bdd55c9563eaa4235f513a81b0a86385fbb6284b8b14b8bc15f1a5fd",
    "sha256" + debug_suffix: "754c539bfce7b6fdb76244cb516d01f5f8760e0c714db098c43d0091e4ccb69c",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "348e329ef8aa6cb2b4c9eb73ac1d4d5e1cdcfade2f35a3ef27eab9b391682615",
    "sha256" + debug_suffix: "96a076f16eac353d7fed3cfd7df2ce2146a681061aad36aeeac21a1375296741",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "0501559979cf363e57a137ef9484c768774fd64f871715435887ef5b40e60e10",
    "sha256" + debug_suffix: "a6a5778cee4687648ab92362e12df1286c11f62bbbce4f7324d5d17743f205c5",
  ],
  "kernels_optimized": [
    "sha256": "f6f9e63a0cf087e93e5f3ef02ac244cecbdcbd7af4603c19eb87228de7917b2b",
    "sha256" + debug_suffix: "ca1b743ff823fd7258b8a1985266fe1463ca11369d539c38ae8b2fa5e9fff846",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9c5ef397094d732248c6cef90711a53372be64423b59e0f2efb11ad13073c23c",
    "sha256" + debug_suffix: "fa2009479dea76cc5be3df4bc9b5b742dbd24fe7aa272c687c97b7ca78a82411",
  ],
  "kernels_torchao": [
    "sha256": "6062995d11b0f7fbddfe53ced66de2295e021a0c80a5ff1a539e3ea260a43881",
    "sha256" + debug_suffix: "e9cd172d6d70080eaf72f07fe54cdcde5c2cc231a0d8b43981e64de833ac5906",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "cb9656a48c68a897a4cb6514df5f1e4b713a74063e51583b43b5b2a18fe3c340",
    "sha256" + debug_suffix: "79491486806f176f9b40c3b1a53a6d64157d7f17666bb6a8a399a197912ac66e",
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
