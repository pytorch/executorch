// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250825"
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
    "sha256": "f11f6cac9ef9221ed2d29d53f7052adeb78dc8b139eefe2cfe81aef503490d8d",
    "sha256" + debug_suffix: "c413b5ce1736418e4b10b4736533478b7e5875ade001f1816e8b9358c7c8a5e3",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "d6297767ec3be17d37a081ee86d034e1b6ccf49d10c075b95d58aacac5d05c4c",
    "sha256" + debug_suffix: "be51914a21a9f9f5842434094a36c090459165e5dcda37bdaeccf4df2678279e",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "efd51a5c365499f02bf19b83b64847840f3efca75d8e4aba974ea08ab44f5671",
    "sha256" + debug_suffix: "dd044ed8a4b52a098c98746c759ab9c26151c4f2cc2cb1e561272981edcc9cfc",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "97191cd120692d0d1def74b7668643d3b96788fd07868b16f06ee8d59c9dd682",
    "sha256" + debug_suffix: "181b934c130ac01a4e154b1709f6e92228ac4c146c04be0227da940ba45a612c",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "113bd6490a0806c7aa3466f067c07b0b3473f4104d01deeb58f5ca0519e527fa",
    "sha256" + debug_suffix: "14fbc607111762b6aacefc32d0812f5bb4a1a200da3c3ca1f33e0d4c97cedc0f",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "6c7ec6eda5f6adb1deb83ad96c999f97a3e6bcc1e2239441e33cf0bc1b809068",
    "sha256" + debug_suffix: "0b28422e942a2c62ed7472c3e59ec46f866b938fbf44c649dff8b8a42ec3e4d6",
  ],
  "kernels_optimized": [
    "sha256": "5c7a0fa6fb4e7a0000bd7f5622bcde45e458d30301af4795e20250612bfcba3e",
    "sha256" + debug_suffix: "74b6d2b2e18265e84a4639ca58d6cf2d97d20762259dcb83c87a6fd4ada296ef",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "b3cfe92df969789aa76f89217f147341119329105dbe8066729a93aee51b7ff0",
    "sha256" + debug_suffix: "875ebae9bba42ca278b4cdddf83468eb263097ca903d666d4e9899023d02ae81",
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
    "sha256": "8b2578fefa00420d14027ddd74325375d9dba35f51e196cf5ee6dcf07d82bb06",
    "sha256" + debug_suffix: "acabb64be6b999525749367375769a14f858f46c916b30ec648a4b58f4beb649",
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
