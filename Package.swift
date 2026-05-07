// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260507"
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
    "sha256": "0a6119465dcf72767bba049e69205fe99577f5bd4d55e1a564e512ce3df476df",
    "sha256" + debug_suffix: "ea18c86d7d4dd3a806128c9d1e12db20e8551da989f5ee7776df918cf561eb34",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "99a43ec97b6ded6fd3ee3a2ce0a8d47c81f44b07f12fc5dd8f4ce841c70bee32",
    "sha256" + debug_suffix: "99db50b4431c45afbe396500d87dc28b6ad9ba073c106f8181222f2c0f408541",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d428bf442a78f614fdc69c3f0666c4c2083820dafd3b5a2491d745325c7540b3",
    "sha256" + debug_suffix: "ae21f22dcf876a58ca8dd8d215e9108b3839ffc251840e33c5314c07bf0cee98",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "7316c623f50122ce43a11bb492daef715ab0553295a032423c446b92180f83b4",
    "sha256" + debug_suffix: "e237aab9fd59a2efd7ad5e4bcdd020fbf138e44918cb56219c810bb64fde0baf",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "48f86fd5a28ddf326f5b81b1a68ba3206720156ac3c6b10c2282a77c3caa03dc",
    "sha256" + debug_suffix: "5093d64f111e7ad0bf0af23d39683ac9d153d1cd6cd7a41dca9d9d3db3a127fe",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "b3d0df29e17e0db414ae57c16c21cdd63837565d20b009d702f198a5c00143a8",
    "sha256" + debug_suffix: "3b65411e7ac477b4230c20f70fa9f66885c752c57137d4b4c7e11b271eda3503",
  ],
  "kernels_optimized": [
    "sha256": "c76050dd4c8a272b0c36150cfd7c1d318a3f4ba40b189b3293eb284bfa611969",
    "sha256" + debug_suffix: "e3b5910ddfbf39943a746053923e5e3c577c1ca523d9e787d79c4aebf085707b",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "5fdcbea0e80790eabe54553cf8e41d5c900773efab58e8b34e9043c2048902db",
    "sha256" + debug_suffix: "9ab77f2943abf6d2b570ce737b2ecf75432739a690949771fb03573157b903a4",
  ],
  "kernels_torchao": [
    "sha256": "52f5f99ddcceb4df5eeb296aa037aad1ed95a3884a38833a1ec43507e4359c84",
    "sha256" + debug_suffix: "de147feef0608af370cf9623b2ccb90946f930d6de8b5164e7cbaff83c88620f",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "afc2d829d22a0e1b6dfa1a5cee48e39e987d2b1c9ef8f93f146b62ad508c0546",
    "sha256" + debug_suffix: "60b908e5afdb96c09c12f460b39e59fbc0cae70b04dbc45923e1bec0e94b805a",
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
