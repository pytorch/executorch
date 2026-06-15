// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260615"
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
    "sha256": "993d5c02b69d3f178a5ef8eed5488fde546983af875bcd3ebceefd921af10d91",
    "sha256" + debug_suffix: "f8f61fb98099391795b4ceceaae1148caf555c979c130dfd2e7470d210beb2ed",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "c40fe7bff42eb31ea41ee223a8b739a8abbb2ab79d83666be7537450cb5984b8",
    "sha256" + debug_suffix: "e9ebfb1451960c1d377f71513c32f6b18b7adad63c8ecab48bb9ef4ac641c233",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b606546c8666867074de5dd102e6f5a6c9212a6290a5a8239640eaf2921df782",
    "sha256" + debug_suffix: "9537339e4763e3cf782fad6e5148ec54ce8b5fee49d17ca3a47a14eb44ba0049",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e319c1a47b8e70bd04a33147d28796b1202f511e3b976b7e5199dd428ba6df58",
    "sha256" + debug_suffix: "e81ec0c497922033ca28895c8a2453b3ff639f4f346c54f9c305999ab160554c",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "7f4f6eb779fb6a6f7294f07d2ca198212a94eca9c7d0e84fd43c37457e83455d",
    "sha256" + debug_suffix: "58a2ded0e2f6b00ec42b965595f5a740fea5af5302ea3e6cd108860f12656b5c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "44d87d314782a3e7c902a94a89d7b0dd98c500bc5a46705f7f8e96732971c76b",
    "sha256" + debug_suffix: "6855fa5daefbcf89759b129fc8301aad41ad436598dc378418ac64b75b55e2cb",
  ],
  "kernels_optimized": [
    "sha256": "c9c51d23602cbdaa9c6983237521c1683049939ddfc63efc21ac0beb608a5e0b",
    "sha256" + debug_suffix: "0542baac2555119ab6b8b821ab568c551b648a06ee02c5ae24d0579ea214f33d",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "62789549287d665a460d92844906415eb9f09dcac05e88b7a7ad250ca2060e6c",
    "sha256" + debug_suffix: "cdfd611833f495e0e0b6e87fca024be9345b82e4126bf94928f93e520eb1a7c2",
  ],
  "kernels_torchao": [
    "sha256": "fc1284569aa52805ea6b0bca725b462ba549823e3438770c92368bb034735160",
    "sha256" + debug_suffix: "a8a956f54296671e7d729612ad9e045fefa98ed73e2d2d65d2dda0b518cd404f",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "fdfd5a25157beed2c383e11744b8d0a4e00ef05cea1ccb2f60f1c9d355feebf2",
    "sha256" + debug_suffix: "9c1f850a41a869c5ff3537a655d10bcc90dc6ec799a2a962ed1e99190a7da562",
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
