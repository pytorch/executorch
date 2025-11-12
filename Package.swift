// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251112"
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
    "sha256": "f21ae7aff4a5970135317b19b805fc1929cb755f23d6ae50784a76d1f546ef97",
    "sha256" + debug_suffix: "15bf699ee25c166d86e809d8e0081a71f22b80ef2599d546359e4f3e16f216f3",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "feabbf2159f4ad856c0663871de4c6f5298b2bb0ba0475542fdb29b91ce1e473",
    "sha256" + debug_suffix: "01905404b07e7fe6d16fe785a48cf1f14b44e93644d6c304118d51e64e5c9aef",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "cf4dfb2841a8a127b63eb99269d8d3582c5bcc5cc714b2b3a4fb91531ca01692",
    "sha256" + debug_suffix: "f7a8618cbb81c5fda6815a0edd92de34d7949878f8fcb0d11b03322fa09bec48",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "0045aabf0c58391980f569f9db140529a8a59861bdc2986bdde7c3468dbe0183",
    "sha256" + debug_suffix: "aae4ff775aad6732fa062503399af0444f1a3253c6e8daead6df31c48d3779da",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "0a2184f8fce18dcab8745e443fb80959a030cc1eb6d656a94dda5ef6ec9d39e0",
    "sha256" + debug_suffix: "0c5fe854179cbcbf509b011cd60d257f5da5dd1cb6e52d22b977077a151afec3",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "3e1950982d15ab7e814d242926d912d5fe0b90dfcfb7b735cc7660c977de9054",
    "sha256" + debug_suffix: "7b62e3775ce189e8f8effa0c7006630da6aac6eca5bd51f044ac9e01c490960c",
  ],
  "kernels_optimized": [
    "sha256": "ed2adb1fe3e20ce42acb58d2a56bfa14d7198ac1e76bd77a0df687e5df191df3",
    "sha256" + debug_suffix: "689d6677bde134306a04e3e931c97198e3094c8e4d8f9b2ec1ebc601b2ade10b",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "2e3084d3fac698d70d46f3e77d8162eb8cd51d24bc037d8f9c8f429acb0b6a38",
    "sha256" + debug_suffix: "e1dc0406994461253e6308ded8345de048f096b3e2a440805832d0904f7ac3f6",
  ],
  "kernels_torchao": [
    "sha256": "958ba7cca1d2ab2612d87840ef257aac3fa3dc511483f5626771750c161674ba",
    "sha256" + debug_suffix: "f27ba87898ab3250487aab9016b5b19285181c34903b15f1e6fcff31d620af87",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "ae756a62078cea4a5a6fbb12be519b71e4d87804c4bfbe0d1c44f84a9e7756da",
    "sha256" + debug_suffix: "b3b8122b601cbc785dfd1aac9caf32558f524f0d6f35a6ce85ec069fb92f70d0",
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
