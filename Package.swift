// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260105"
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
    "sha256": "b315f2bb36219d924dd10862b34170c789a41fdb108f3f72e60c9478427b0978",
    "sha256" + debug_suffix: "2d91a2b4cdb4751d25b5852715c60dea248b77e309c68dd0f135af5f8ed1cf82",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "81976e3e188e13981a24e17083235fd53d76255ce83753c6d09faa836cf24a27",
    "sha256" + debug_suffix: "31665119a11a131d945cb8832b19a570eb0eccb4993f27f233a749de848e56f7",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f228fb1ed68fe525162b78efdb05789c277934037b73e0e926447591c4d2551f",
    "sha256" + debug_suffix: "bff3ecf13edf7aa71877c9bcb23986363a83f09e29f38af2f6c5b9e8a0a6fdb5",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "b05a55807c1c5d8f93d6e9ee1ee4ea9a3f52cd8c365888d1edcdb14186c04ec9",
    "sha256" + debug_suffix: "4b933c85f2e2ccad08a896cdfba96d083e3e17e11e85d8b8b50b82fa19f83837",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "59d3085d3efaae870583c6e1985868e8e69b0c394801f44199ec9beda24ec6a7",
    "sha256" + debug_suffix: "8157240a46208d726da1373af853b559683f48b327ad9132d47a782e1a8d9cff",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "a0f12a67f358028bdedba921fe7bc8666af34265a75293972ffc8129f94a9d22",
    "sha256" + debug_suffix: "cba9480a5ebc58bfa3aaa007363fe528e88f49d9d098c7241be9a6676f7d425d",
  ],
  "kernels_optimized": [
    "sha256": "45e9e4ba2d69fc433ba7ca75bee1d773d264e97639e6108037d2a94a9f4d2df2",
    "sha256" + debug_suffix: "5a20b7611522d345b6bc5353a8d85c26daf11faebeee4fed3955f60ccd3a70f4",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "cba5992dacb7b602887556fe71e972f138928b39e336c8113da440f8f285c682",
    "sha256" + debug_suffix: "b49b71f45a3f77cf5a90c68dd8dd5fb53fdca866b26d86c36a7ea43730a497ad",
  ],
  "kernels_torchao": [
    "sha256": "ef21a134f0763782d310c7adc4f0cb5291da97ddd9a17458a5d41eb38d61b944",
    "sha256" + debug_suffix: "75fe5d720bb7a9daa2e1bc563002bdd547a849e8ea92ebb46749132565c88661",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "ce12d63e7024d824001a0954cf39112115f5e9c950c5e3862d9e7f557b904c96",
    "sha256" + debug_suffix: "faef1cebe5561f13806f1004f1a44fa571a82a99bdd4c2bc784ccd7eb2dbb45b",
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
