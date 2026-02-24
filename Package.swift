// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260224"
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
    "sha256": "49b6e968e2e344f626fab18642b3f1ce2fadac43229294079447e487605408a6",
    "sha256" + debug_suffix: "d8f7efacee23221ebc7a3327a057b3d853fbeb65567f7016060431d3c739a96c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "0f70bb54a49adee85c3b88ede6ba8223e70387592474291a861accc3cca27df8",
    "sha256" + debug_suffix: "9f843ccbdaab76558fd3044abdd20a0eed971463368205b21107fad7aec6662f",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a42f62fed14816561f223736e2435d129dd0dea6500c9fe1bfaf18c979314d2d",
    "sha256" + debug_suffix: "e6df337d2191550d08ee6a817d75b4c9ebcb29a0f3bac280ccc95ef114b624ff",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "2b77aa7c1734b51d41ffdd9d7db67119f55be164b166e13f67594ffe54122d21",
    "sha256" + debug_suffix: "8de0227e4068ac05f7ac5ff5c0bf86100d3bbba07a7429251162070ac341bddf",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "6a588b06d1c6f4ff418fa321f9e315fc0330d3a34278fb79b3d4d788fb2adad6",
    "sha256" + debug_suffix: "9c46fec620430343548ac2bd10a66e99be19ad569f348fe0f7b9ef4e72c9ff53",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "2e9c369adca914abbe9ce36802d88b1b293ba7ca26d3110f3e46fb20860f7396",
    "sha256" + debug_suffix: "4ec74b34a715965074c555822f0a08edca9c2c52fa21c0b37dec3d22298f4e69",
  ],
  "kernels_optimized": [
    "sha256": "77cf9425e1fce7311db406f2bff01836dc8153803aedbcfffea097a2a751b879",
    "sha256" + debug_suffix: "0d0faa5b65e3dcc9516657c541970e97441c18e4639925344e8eef2133627974",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "dc163b5696a5f9782755741558f49cd5bc773177d57f2997c8300b8fc7865a37",
    "sha256" + debug_suffix: "a0c1feb445647c83593df09a840fd1d8fc86a8ef0d4048a788ac5be21b914931",
  ],
  "kernels_torchao": [
    "sha256": "9042fdc42edd8418f85dce093c3ff54c5085dfbc698d3456ab82528393525d3e",
    "sha256" + debug_suffix: "574e9b8f6f265419a7517d0e338a7662b893ee499079836c2bae753a57dfda78",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "674d9d26d426b7675c0c5a0a6efc8996d1abe17502cab874133dd885fc38dd50",
    "sha256" + debug_suffix: "e2b0512a74a5a5b09afe872ed591925c8ec3f65532510062be6c3b0312f8e1da",
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
