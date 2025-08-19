// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250819"
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
    "sha256": "75e840da818b5a8bcbeccf2dbe100780554a82707ec8ed593cbdca50eb800a2e",
    "sha256" + debug_suffix: "a089c572d0edf4c706a7353d9904fcc73a7734407475f94928103a473a793302",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "0c4c30ab80a47818902c27685c9dd2747da2252e0f5957febce1186342c3a765",
    "sha256" + debug_suffix: "1b533123aff302c4fd11f09a2c4ef18fcd4f2eca983028236fae17177d7d8eb9",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "4af1f491dd5acbf8443cd6dbbdaa20c2f2498e1b8959787aae8009758c53cc52",
    "sha256" + debug_suffix: "bbf9cef8db8e61c692d67def05d58ac379122f589516430c9e31f3fbc1883535",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "44084097aa596fc59c1019e831abedb08307bc890b665550412c515c2a99062e",
    "sha256" + debug_suffix: "e2c21e4f752c3fc095c8c14fc38aae8b3141d828fcb2de2cd9f622f6dd752438",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "9909732cc7b2a288d25368ae369dea7c52e79590f67e9c00f105085b8e9cb886",
    "sha256" + debug_suffix: "60d11e045183012c2ecd8de14a59243c9fa2404d5bd302469583572c7914f5e0",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "f14c4c58248993eab79d26936418b420cd655145438b73c61dd6a87a11e8ebc8",
    "sha256" + debug_suffix: "360ba82fd6929b4e5ef4b7b03ef38f2213bc5cc6588991a78efe478cddf0ea7c",
  ],
  "kernels_optimized": [
    "sha256": "4a21b792b95ca467ed34f67308932dacb1f2ade078fd030bba70d30183a89afe",
    "sha256" + debug_suffix: "c47768946793f508ec5044c8e46e895fcb76740b4bdb71f2fbf8c02a54e24220",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "d68e365e948cff87b76fa934a29e2a4baa9e55e89d3d7aa063ee65009e7c805d",
    "sha256" + debug_suffix: "6b4ce7af55490ebfb7b7b08dff129e0338d346715ca42bdb553445079b7f2d84",
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
    "sha256": "6d8ed39fe1092659fa5f6e6f505fbac5bb1f67598fa7618be93fa541a6c4939a",
    "sha256" + debug_suffix: "7213fff5ff54c98da0b6caf7ca1219322037d6ccb1fdb8e8de67c54553dce33a",
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
