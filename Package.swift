// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260501"
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
    "sha256": "e5637cefbbb9cb3f77d48776c660db9f116f03245e0e8ef7a9d56e78fe77701b",
    "sha256" + debug_suffix: "03426fb118ea125edeb3fb3db1759ab6da44d7479319ce90b82b52ac61df70c1",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "ba32729879cd1345141903c1dd824df06f0f554f472440fa77f218e3c3181e0e",
    "sha256" + debug_suffix: "5a21f35ceb9731bf2955f357cccb4194b8d9bd76bd1f6fe68ea6ab616791ac81",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "5650202ea192d0edbec55972ab6088046fbd44ac07f3ca800f2c3aed57997239",
    "sha256" + debug_suffix: "3733a244f0abcc6039dc5782c313298c9bdc566b5cb9b8e3c15a56e63062873b",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d2a11a8c341d6f7dd27058c191473e551e12eb37278e248d8a217e0e3f2d91dd",
    "sha256" + debug_suffix: "205a22ddb622b698890d77626b985762ac6c91c7c7d395b7a71813a5d2ce7bf6",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "81527d6f346b5cce50b54ad072276793e8f71bc51f0224b3544a6984c96c28eb",
    "sha256" + debug_suffix: "9832ab2eb43680c2d582180e3fbcd5eb59d3ea606251734f118347616b9a811f",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "3d0c8baa19c50ad8df2c6ebdcf1a8cd91fca2b9760d2ec2a3c5b40b88b2ed935",
    "sha256" + debug_suffix: "7a61011cf14e0ced4f6ec8901b33429643e9a8c119ef230871487b803504df27",
  ],
  "kernels_optimized": [
    "sha256": "c8c24c618fc9ad5be9aa094cc3e6f17fba7f4cbdfbaf72aaf14923f0ba8c9084",
    "sha256" + debug_suffix: "43de4f0a350c0030801585b44fde39de4a4a2874faf7c0a4c23c7bda37610fde",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "85b1d66189c345c4b90ab68939a2dabcaa0f68e6ebe81677d6fa13a3a0c891b8",
    "sha256" + debug_suffix: "aa5f80d0419f437b233da3a96122e2b9b5a8e90d747b80dc32a2990018475033",
  ],
  "kernels_torchao": [
    "sha256": "ef8f70b5c16fa72d2c7d954fdb92c4800143bc498ffb09ead9cde6d6c865667f",
    "sha256" + debug_suffix: "146cbf4dc18d4d30aff082f594e7aa78ec378911fdcc02a42540e54f0c858ab3",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a6dd969bfe8b0f59938abac781f5d9398755777a600655e24ff1f8b1fdb48814",
    "sha256" + debug_suffix: "462c5756f96bccb1f85a9b576b35d52296287913f9b7b50452572e02b04d6c56",
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
