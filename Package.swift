// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260326"
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
    "sha256": "6a2480c3d857e553088b66dc52146a03be1848881a591edb07376d9dc329957e",
    "sha256" + debug_suffix: "15029508dafed69795e2471ca4987e6f946bc376351bc9968cdd77457ec99137",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8e9c137819795a6f47574bfc5e75bd5036eebaf9ed90b1f517309d0fc1eafa98",
    "sha256" + debug_suffix: "9257eb06333023960fab0743e220bd5ee3693b040458ab0e98cc90faf6c5439a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c26a49b63b30273dd87aeb5257985bd77a67b597035df9c4501dbb4f05428b1d",
    "sha256" + debug_suffix: "4e5e94043b62af9f02b44a1f4a8a79729db6fa2c62eb40caf5ef9479fed97c0a",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "4d97bdc956525eed050ee921cc12cd7669296b9b07b9484bed2bc06e8cb04c03",
    "sha256" + debug_suffix: "c0fbffead4187a65901c6d16293beb93b2bd96cede2f8b50b0573c0075e5e4e0",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "094b12366aadc7112b5a1ed8f45447bccc2608e94a3c77279662f600b599b751",
    "sha256" + debug_suffix: "8d43a902414f13bc21b7e96f231890a01690f9fe77e3aaec7d8f0c5867e64fab",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "4847cdd956402a6bdf86c673dc9032fd6c3c4b8bb9aec475dc8791191707d3c7",
    "sha256" + debug_suffix: "738b24a8a56c4d2e43addcf1216671632c57e3d5445b3fe31c80b731a589555a",
  ],
  "kernels_optimized": [
    "sha256": "c052f833f00eca5f473dd1aac006c919b6d2db75b24ae30bfc09cf87486be83a",
    "sha256" + debug_suffix: "5baab96ff9a55b8b27776596e3fc36b2cd2e325336d5edaedb4a3a0ab4654021",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "70a5a88a5ed47c550db16643ab667bbaef219b931f1dba5e4565605bd172d498",
    "sha256" + debug_suffix: "56768f5803865f539fdbc46201ea7dd2d5177f4991ff8f7f14828e9deed7ea47",
  ],
  "kernels_torchao": [
    "sha256": "8c57119b7ec856e36ad839be702410204e5e3364770a4c59f36ca04f8a11059f",
    "sha256" + debug_suffix: "e31a0cb9ae44130415973889b0be2e5f2ab889c66048d3f26a237d83ff0c6045",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a62da8a1ec21bb1aa6425f57ebf096c221e9cc2f73f0ae24ae47c169d9a9a5a5",
    "sha256" + debug_suffix: "638ccd6baa7c842e8bbb790c4e96c1d2f2f2af96330c7169786f7c3058131b2a",
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
