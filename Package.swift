// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250911"
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
    "sha256": "48e3e43395ae5028316eeccdada9faea1ed777ef94e833edc22585e5453b59d9",
    "sha256" + debug_suffix: "9e1602988b80504be32d34e7ffd0f2c7b324c9424f510dd58879e98018c877d8",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8099fb29fbf0970b0577932f46ed6ebfdefd2f59800b5eec58260831f897596a",
    "sha256" + debug_suffix: "c987f3b5df2a20bb487b4e4d60de696ab210b554eb7870d79eba2fd1c73c814a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "5b877e6126d551a71ba8a4109c7a6a25c0ba8838e8732bff1dd5be190afaf383",
    "sha256" + debug_suffix: "9d5c2bcdd65dbfb74b98030e6868c5a89b52c3a3ab9ae12a32bb2f52e39fc48e",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "c52767c111e2b4a54f79bff093a7a2036066b8b9a215a7f3e58a20ddb87b645d",
    "sha256" + debug_suffix: "e3a58ed81bd8b3b0c2fd96bdc283b2c0880571366061bee7f53e521e9d86f45f",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "f63f65386671f50ce61312ce02e939bb50506c32fcd710cc7f6fb208d9f9f2e3",
    "sha256" + debug_suffix: "8dc7a63fce54f6ea28f7d773a4bb3481f3691b15ec86c41299997c22a81fe928",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "e9f1e67dc40aca3167b23957338d233587acb4328dc79ee5e33cc0dc8593f994",
    "sha256" + debug_suffix: "403359db1c4e46c4d74261d6bd6797708c2e155f8bc29a73cfbd3bb1a232b050",
  ],
  "kernels_optimized": [
    "sha256": "4d408979cbf08286d4183ede6f93aa9fdf4ac9ace789d2a0775c1084be72cb35",
    "sha256" + debug_suffix: "f81d893fba97e3fd1f285485ff1991c36af4f559267ea9fc9fb45799fcdcaa25",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "45dba177ab69ae57a897219359264e8a4176246f01d4451d8c6dc16b27be10a1",
    "sha256" + debug_suffix: "7607fc8945e2f5470338cdb27d2f23d87693c3dd941a7b40b7a22b2f0ebabd80",
  ],
  "kernels_torchao": [
    "sha256": "a25276d6f9646a22a0ad0806e3dd20d49f1cfd6fb988cf66cc1c8a80a0d34855",
    "sha256" + debug_suffix: "6eef7e994976858519734114d7909d26aa012f23119dd83d9ae5ff87fc345221",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "7b894362cc0cec31079969e2f2f5343bf67a4ddf9f664214087550db8e2b417e",
    "sha256" + debug_suffix: "82be025a5528b6f169153c698d1071084ec5449e27c26eeaba86b52c0ec2e3d1",
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
