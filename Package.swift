// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260617"
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
    "sha256": "5284a8c61768dd30ca6f027c8a466c3113149f1825edfb10fff972db6274a6ca",
    "sha256" + debug_suffix: "4296fe5bee82cbafd9e0367977f70f9e8588005bfd89659c0466b73ab61898d3",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "9639383dbb0d1833a501f10ff50997d11a8ecb2323919e9f8705974538c4ab22",
    "sha256" + debug_suffix: "4e402eaa0e45aa71105d416ab459d1be81dd672ceb142fd636bce40a084bc5de",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "9400c85c1f6cbce15df97a371c516fbbe2e589a2eaad8850ac363b7ed59463f5",
    "sha256" + debug_suffix: "682576f10798467d93bcb5b6b553eb3c3925e2541301e183e9c0a47f025fa052",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d9d88a79ca18d18e47bc489b722c4416b203c83a01a4f941bb002b63c326315d",
    "sha256" + debug_suffix: "adbb382fb3f7dcd16c12b4a3c8364d1e2b92e6861218e6669e8ea69ebcd868ff",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "037918326bfe0059cb5630d56006b2a51882b500eab12df353b4928469967bd4",
    "sha256" + debug_suffix: "1ab46623f2ee0c60ffca90ec01b9523f66269ef174ba4c1a636da5521a93556b",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "5cdbd24ca693e4ccbc14920813443237b8a01ab2b6ce2a95ee06bf96e704be51",
    "sha256" + debug_suffix: "30a92d514ee14d6d19ee0da29b0de9a23c300c98f201b7d26f5ac68f8074ef1d",
  ],
  "kernels_optimized": [
    "sha256": "1742c6e6c45ebedbf394bcd52bb57084efd999c7f603644bb34669ceacb365c5",
    "sha256" + debug_suffix: "977ae83a47a2af3eadcd20f1fe0f943a109af1cda2517d3a0f2831aa116a81dd",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "c6006a13f1c92d57cacb7b0075223dcd4eb8449c7c15049263f2acf77e079879",
    "sha256" + debug_suffix: "b797f5fe9ec0de62b40629e06ee2ed2ea9d11f7d90b589cdf1092e8cb6305b56",
  ],
  "kernels_torchao": [
    "sha256": "db85ff1d5b06ff6929da55e7c208b171020e78b31fcd23f6f6965429b5857235",
    "sha256" + debug_suffix: "2e10ba214e6052b55d5f62ac0075db19b234ce06835e338e52eecc6bf197bf57",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "070ef97f562508e55ad5fb504cf5947e65fb6967d6d0c483455ed2b01f70ba3e",
    "sha256" + debug_suffix: "bd5f3cf491f81badd66441ae71e6918868d4d5a4b83e0bacb9cd13396e15dd60",
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
