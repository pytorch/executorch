// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260116"
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
    "sha256": "e965e20be5368c87768f75a91c2e36a38c804ec7c8b70e465fad25f281eb5d25",
    "sha256" + debug_suffix: "62ac4ce73dd65a2f9e1dcc73d03b5fc4933d0be827227acfe9459146bca752c1",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "a30e8a0758f7655c7c603b3e38281eb9c1744d5265c86ea36cf17290d0a2ddec",
    "sha256" + debug_suffix: "f6b84b83f625cd38364b90527c542fcf89637a9d1341ae35e09e7b9f0bb07976",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "77cd5f416c6574fdabcb689d36fcf0b21be71b667d2722e6e0de393ac5942826",
    "sha256" + debug_suffix: "5238fe20d379da197d71b58a197482b6bb4b445710a6e107b5cd9d6b2e94fa95",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "ebfb753f2e8fe47533fe2349b3dc361e83a4eef998b219e54447dbd04494889e",
    "sha256" + debug_suffix: "15b7190194bf59e347b92999819924dd7d01d16ecff5adf145e9bcc8940a0bf8",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "fba1938fdcbec11724ecc258d9345b004b48723d86bf282943a45538647085c2",
    "sha256" + debug_suffix: "4fd0b243d61f0fd250a22dd805c731e71fc1551da67539aeb270c73bebc3185d",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "a1337df765d779eda122924c2c8086b830d7165600239b4104eb40798f1fabe6",
    "sha256" + debug_suffix: "92f9f6f018e83d793ef06477b797bfa9847b8fb06668227fc0211352b7d7c86e",
  ],
  "kernels_optimized": [
    "sha256": "4d4bd76dffda5d23508ae723fc9ecb25e0ed5d68090bfb18bda566287097acbf",
    "sha256" + debug_suffix: "cc3acc18511c1aa91dc9614dac0f500f878b7554b5f1371514719da6a28fa91f",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "b04a1500a65de002d0538ef7538359c4f0943693dd9209ab38cf97fc23dbc840",
    "sha256" + debug_suffix: "76aff79be929a30aff0c7ca3fa4603100abb32234bf76fa656566bfeb00a9bb2",
  ],
  "kernels_torchao": [
    "sha256": "af2b93fab577d52bdad8557359e88282a65d0d8939c5f2722bc48d3c72e3538b",
    "sha256" + debug_suffix: "be6de698cc92c4a63da24b8f27685f8b87e2005b9168fe3002ae91f60034b45d",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "390f45479139b945120b370dc901f489999b059ebfda5ec62478702202eb88d9",
    "sha256" + debug_suffix: "1f729434633b906551b7c9b822a7508343aaff19c8c9e4c62818257618921643",
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
