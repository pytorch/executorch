// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251006"
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
    "sha256": "b519be348777a1875e528e194b5265c0f0baa8e16b83b48c8e5b65a2c84cd791",
    "sha256" + debug_suffix: "f90013d9599e64c94029f735dc950881c8991433f076c75ca169464d77554a91",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8a790e1d4d192ab64d2e2ec4fb8e120af450789b10ffe023a039aca1afcd4bbb",
    "sha256" + debug_suffix: "78102416d126f6a7da708bd092e356533eba7878038c131102d73b5c1e1fc18e",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "23024a8292665cbec5bf3e45d5caed8cdf8af438777695b67bfdfd6a7b8b19db",
    "sha256" + debug_suffix: "e893ca4205835da850b35819d6b77ed48c4f3466051f1ea244232d0bd0303bf0",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "3852adbf581189540583c279d200143aa03057476392d9633460ddac1c484957",
    "sha256" + debug_suffix: "ca798420f5a530265b87ab2e96b3842866dd976bab5ff5c497fef91120e248f4",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "8deb6c090430bff837b44337799ce8a583fac71366f0af95cb0a8a2a3af74f4e",
    "sha256" + debug_suffix: "7fa58b418c47728579936af91d63bd40620e1ccf0681ca72c5d51c5b802694aa",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "1aa177616d8a65cf279909e1c021dc79d0b60eb1a255f45dc65bbc2b535760ee",
    "sha256" + debug_suffix: "ad183e124d4c00f24205b979cc540e30e146067ee00e37a58cc2aa9efcab5072",
  ],
  "kernels_optimized": [
    "sha256": "012875de4c4392ab50183d4ad83f1ca67c4a6fe923b880cecc67684ead6d73bd",
    "sha256" + debug_suffix: "37497376bb8fba4a1901ba341b1dec31ee964fd9381b7907d8396b48bc2aace2",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "d4333209791ca1d83dfdf5d5ec60b09170b044c4ce9778ec146c7bc6045649b8",
    "sha256" + debug_suffix: "bd9ba0371d8c8ed9767470480f8510df3d525b212d1b031b9b7d496224062710",
  ],
  "kernels_torchao": [
    "sha256": "216fc0c910e276dac7ce79f0d638a56b8f1333ba9abe8e3f6fc5fce0619f736d",
    "sha256" + debug_suffix: "f4bf4e4e39783f0c7347aedba3271a975e4acfde2da05299d68377d1f6731e76",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "e062550ac3117040cb39c5262db35fb2d980e00967f3a6ea5d8ad5c18d8e865e",
    "sha256" + debug_suffix: "055b6f4c0a5119c50c17fc43ca2b6cc9d52902250004ff7cf77f0033dbe327f2",
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
