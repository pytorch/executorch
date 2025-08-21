// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250821"
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
    "sha256": "5f20c0461e91b358f543bd7a2de609cd9ba44c6d8e3fe655114bbfe4661e2523",
    "sha256" + debug_suffix: "ef3ac70c72e73478829f124c5b9233af1f6111ee2f827f65fd09575582d8ae77",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "d47d3fd853db756d5fe6e4c0860f8c7da6886ec7ba262713c1af4532e86eee0d",
    "sha256" + debug_suffix: "ec48f6d43b95957290939608d2569edb027f9254608833219fe9a1e56701004b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "287dff9f271f97a832564747e5cddc7c4b4c4a461f3c6832b23a58e20b02d47e",
    "sha256" + debug_suffix: "01bb1c2cd5c875a3293602696f80d70934332ff98ce32bd86cb359ec0a729560",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "66dd47fdbe83bb57578426312d9dc3e1388767caea1ad92edeecdec8f543a54a",
    "sha256" + debug_suffix: "325ed3472925fb1f116334ef1a118a185bb2d6a341b8aed18dcd3246d81363cb",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "4e819d29898e9b918b07efa9b57dbc0991f1e8fd5df5485c44fb8e45ba2359b1",
    "sha256" + debug_suffix: "4a101b634249d736b575fbc9bf7c1b7fac2bb0b0d8b71997e784819746a21b88",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "959892c1b6faf6419572ca190b00c553ab0569859283fda462fceef51c6926cf",
    "sha256" + debug_suffix: "c91255eb8113d28f867e2a40ec62a1ef508256f6058edde78d7a8e08098eebe4",
  ],
  "kernels_optimized": [
    "sha256": "938bfa14c6ec23300bba45b0f72488f766dda4a2809649bebc093863c4493680",
    "sha256" + debug_suffix: "e35ebc712ce84053667f5d4567ade282408526a322e189d43cf45c8069ce2536",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9007ce54793966c0f23cf5722bdcaac96d00b0ee37c8ddc4fc3657e1f63010dd",
    "sha256" + debug_suffix: "6ae8b49aae16a4313c820ea8023fb7b15f44e02c6ee910cb887adf0d7f3d1ea2",
  ],
  "kernels_torchao": [
    "sha256": "__SHA256_kernels_torchao__",
    "sha256" + debug_suffix: "__SHA256_kernels_torchao_debug__",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a659b25ad92e0face0e5579cba7b8618e3d01a1a58595d3f1ce29151e9acc4bc",
    "sha256" + debug_suffix: "3c09404f7559d17fa98cbaf50f03ecd6cfa1b20c580db6b092bab5b5d5f2f945",
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
