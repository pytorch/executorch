// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251015"
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
    "sha256": "565fa65f5f0a135d110d8a9a6a61043a34131c91e5dbbe99214c38243c20f55b",
    "sha256" + debug_suffix: "03189e82f529d91d6b07642993f32a81b3e348aeb2d734205e2b27ab97d5c366",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "3525d1b53b2aee4178ec99301a478c2b4d5fb78fea8271d4d45cfef01b5d2bba",
    "sha256" + debug_suffix: "6b6d722d55b5644596859d9c3dfdc89def78d2675291f45d4bce12e4fb680c85",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a51d47b91d6dd4bb4fe72f50f3a236f8089cd5873e34f5f8ed1560d487b8d87f",
    "sha256" + debug_suffix: "ad2b8124e1149cdd8174b6856bb9764d3cf47ef69b36161706745a412c2207cd",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "870bccf3911c796bb619f3614723919ff8551c506c8f322c3f01bf4cbfc28ff6",
    "sha256" + debug_suffix: "26cd7db509b18547f7fa307099d323dfd48ea3e687c199f8d59b56cb46c5d643",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "dfc1e456b6394f0f11405773849f15bb5534d2efb718d0b94e69bfb06037521d",
    "sha256" + debug_suffix: "7771a97607d12d3dda5091b967aba3fa0613063e91ad9c598a8c15d4f801ef0a",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "7de657a437ae2f755f4d5a017d96a5922265558d7358713671c5bc303fab846b",
    "sha256" + debug_suffix: "9ad935d44f2b3eda75591ec34eb248a6172819f21a929f4d30510cc33ae123f7",
  ],
  "kernels_optimized": [
    "sha256": "999835e77f8d5d2903c2ca305def0f509bd80a8c23765b6795f1aff6f75e0de2",
    "sha256" + debug_suffix: "038a490f3fa29d9a4af8dab0e734ab6e3a23f605483f87934b549ce310bc33a9",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "b3a733ec9ea9358c5d65d8201825d5a23dc7c7a795edf3c59a74f63450b6afa6",
    "sha256" + debug_suffix: "53f93cbf53deb545c4eaefbb2f76e25ab71225133fa7bda3a4ac57292a81eb65",
  ],
  "kernels_torchao": [
    "sha256": "077db459397a4f68ec297ad84dff2bed1f8f8c99ac50d8bffccc256f800905ef",
    "sha256" + debug_suffix: "a0e8ca4665b5a7a7ca2e4d29e8aa6a74f69bc13342bf2496851d3d7c8b6595e5",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "e9db5a57c7c86288d1bbfd05a6bb22fec81f30933115fba30fb8b5f03c3eb71f",
    "sha256" + debug_suffix: "daa752f2f126d26eca6ec8a90dbcf38bf7d13887754bf89a38be3ff9b8295203",
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
