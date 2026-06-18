// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260618"
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
    "sha256": "69bbdda79aca67776a61676cf64c206e2535736c409f7a803c0b255fa6347926",
    "sha256" + debug_suffix: "6cea1bd538897f02178882c817f7cf1cc7ca710a9f48b568aa409df4446f9e85",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "16a0c1e005e842342f6be7d8154b06ca690bbb7cf80b7771f9e1bba98db350f4",
    "sha256" + debug_suffix: "55cebf1241808bbe8565fdb5e2da93cd532f96c4a57cf7930f96c0f547497379",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "4b02ba36671fe2a0852138b3d8e9c3cca4095ddbd1f08d11008c5f4009af047c",
    "sha256" + debug_suffix: "7107a1ff2fb51e71f221bf18e7b43c00a1595442eccb15ca300c27043fc5c698",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "398155ddf8b3c9ac9f56120fd11393497865e84e350b340cbf2b5797ec4f9aa0",
    "sha256" + debug_suffix: "f3498fd00426c9c8e444e35005f808622c74158f4575b6b9aaddcf1c16d1273d",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "6123c7a6ff640565b16170a948f1f384e97b2bd40273a7940593b91a94dd18e8",
    "sha256" + debug_suffix: "f18a48e527ce28e7a1af32230ebcd9301f0f73235e943e1e8fbff26918b6fdf2",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "888b1c382a82027818b68a4b0ef551a07c317abd0264bf28e8ca5821b210245b",
    "sha256" + debug_suffix: "ba56432dd6b423c7d1e66a046d89d5ff7183b475c4f215d1aa40aff2ba7d0c7b",
  ],
  "kernels_optimized": [
    "sha256": "4d62e538e72d76c4576280921d02f31b4f00fc69d3dd3da4d6c7a566f704a776",
    "sha256" + debug_suffix: "c02c907f16e008df5ea265c66ec20a3393072ab2cc4da9b329ffd73509e76f95",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "70f715e1ca42ec6d96f87d64a784bfc5914c06319df7b220d2404cb6258bd952",
    "sha256" + debug_suffix: "57b73b3cd2a3f58b42393417df39e9b424f6d08a42abd48b623aae7771e60cbd",
  ],
  "kernels_torchao": [
    "sha256": "2d6789f25e4ddc7504e97960eab9710e9925986ddc203a8793052a2a1042369c",
    "sha256" + debug_suffix: "c32f6c22bc2525f0c39d662ca1359c695d3b3631e6714ba119f5640f3b0c4274",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "7088b4bbd79b9b46af601904ea94d199ddb319547578a5c4615be9cbafd30027",
    "sha256" + debug_suffix: "86ce8d6ea30865fa37ca19efd5e1b553bb81ac68a7004ddd3bd711ac03318e85",
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
