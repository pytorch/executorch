// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260416"
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
    "sha256": "a51b4ebd6015272b7039da8938b6b6f3a5cdaecdc72b39d59f05e6cca9854ea0",
    "sha256" + debug_suffix: "2b032a88c9e4692b8bf948bdf9f08e682d1f9aa298b195f1d838b9e40668c194",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "1683a4ee2e3b0703eb4b246fbe7f62d343a1dc1de8ddef59cb601cfd10b1b0cc",
    "sha256" + debug_suffix: "29bcd82267370c761fdede5e5695d4aa3b0ea6789c756795d983400261b4565f",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "991447d48fabc73efd8b13ab245645f0c86e7ccceb8c6856a5b5b911b883a374",
    "sha256" + debug_suffix: "a16f2acd2538a951633926a24c0665769ae769d97a1ec4ca36ea990839a2568b",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "c97f4d0200881804b5b8c6a22f399fea723d93a3fb2f0c9471d90a8a1fe89d7a",
    "sha256" + debug_suffix: "d553c59f24540609dd476a5f3d5aaa99b023e87694cb330916f428082b2694c3",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "a9aeb7907bda2ff7305313e5fe9008c79d51ee5b42e235402e322ec41502d804",
    "sha256" + debug_suffix: "9dc6930400584a3fb0d56c48004b64b99ccd67654b99bf752c36516088cadacf",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "3a5105df5e53efbf708be889ecc4c4de0c25337d410f04a003145f6c0ab8b3a2",
    "sha256" + debug_suffix: "2a939fcf53cf7c2d4c5c87781dda37cacf1153557aa069d26df92e910c1738ce",
  ],
  "kernels_optimized": [
    "sha256": "fe599bc57f49ce74331e0ad741e41fb67e264f1e5b088008601907e8515639de",
    "sha256" + debug_suffix: "5ffdaba83ae25a9d6837378a78dcec0378e3e13461535de98bda51fb2a5f9390",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "5d5d85784ce44d9599f0dfade0e8654cb6e392853ceb732c071e7bc1be08b297",
    "sha256" + debug_suffix: "539523907c4e34e7fba01c468c32decb9ab2cff1c692eb5aca2064dc20d5e5b5",
  ],
  "kernels_torchao": [
    "sha256": "d24722388033a75d7b86af36e59bcdff9822ee034217bc01892fab2cb98f1881",
    "sha256" + debug_suffix: "3c4ebd57abda194003a1b49568d70aa8affda062aacf93eb03c729a1ceab3bf7",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "b7389e43fd73184dc79e5d34843b3809375883c9d116fba955ee1a919a011dec",
    "sha256" + debug_suffix: "7b5bca4f49b15a2ed435015086b4df9430e35dabcb8e2b53ba679683644ad065",
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
