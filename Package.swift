// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260415"
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
    "sha256": "67b0abeb07f0df36523ea3e5f68c6908090d81775b345dcdd5cbd719c559ca2a",
    "sha256" + debug_suffix: "d22f62e7eca6f168a2eeb1de1e0f96efcae0ce07a449014b51932f6d9479b562",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "e803fb09b47e5c2eb449b131ff101d73699b53b9bd0d97e52778232510a3b126",
    "sha256" + debug_suffix: "b37bd13b441e67a9787ef78250d7658645f808920ebd38cad92e3c41a0b8bd33",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1f63b55f84882d318f00cfcce85cef785b0a735f623cf983fa6d086f1a8045e3",
    "sha256" + debug_suffix: "315b4c0a94310122dcbf37583b7c0546be00138f561ce72c34abc748f40fd2e9",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e46802032c5d1e982b704aa8034bbb4a774549563d73b477c7dae24e93320c69",
    "sha256" + debug_suffix: "1691bc765f0f1db78ecb5d5609885b3baf5fc9dbaf7cfebf8ec20d50328f7caf",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "a37f56e03697f74b7f2fc3300e7eece771848b4bf87233f8ab517e602192f9db",
    "sha256" + debug_suffix: "db995714479736d4ef271cab137bb399f20c384fb4bdda6bdeb10f8c19cb9c6f",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "808efb77e228025b94a9f69471a9cd5e19e8de405bd589f5b167e5a90b01273e",
    "sha256" + debug_suffix: "1ea64d9903d82a1c2c88929f60682fb469ca707f5918f67e577fd06319f7a30d",
  ],
  "kernels_optimized": [
    "sha256": "f738f96b7b7574ca348135ad9e29ced4a39797ba3101006520feb75e0cde513e",
    "sha256" + debug_suffix: "7043f0d380ffb5b604edf2dabc47a22094c291415652514048716dd67adfd930",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "88a1e1d14c9dc35f1c7dbfeaf9dd9077f1fdd80d209ceb24e7073d02698b827b",
    "sha256" + debug_suffix: "9cddd38a30989c2be82f8f272b51aedd753a4536e1d30e9ab9248e2caa2ac5f4",
  ],
  "kernels_torchao": [
    "sha256": "f1e24edff697d1d62169926e49024d49649dcfd44cf1899f41f5f1f583b3df45",
    "sha256" + debug_suffix: "7edd8bcac44260a11ccbcb9331e39624454117df54ffaac6b507dc1315f12227",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "ad53b9561a8d835b4d6f1f643c669563fdb3d23d4b732bf9896bd51416085aa8",
    "sha256" + debug_suffix: "c6fc2d1c8c016e4cb25eb4bb0593040363dbe801e998ee0920878c060aa7beb7",
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
