// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260605"
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
    "sha256": "d153b79ee9eb61fefb3fab91d8bbce4af9fe1c8735ba3a363e47ecff9ea827aa",
    "sha256" + debug_suffix: "0ee2267fa829347bf1f5e12369d824560e40593d9fbc0b95a8321fa912f6a8fa",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b8c6b189a0e635050927826e78f54c2b3e951f824940189105ec05bb6457510d",
    "sha256" + debug_suffix: "8a02a36068b94b151eab5fb0bd0660daed89068f26494735dea67d37e0ca753c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "81a81d82b749a52eeee78e8216091ac4353b864b15e70267ca5acf23dd7f26a6",
    "sha256" + debug_suffix: "07cd35cc6ce9eb1f2f65447372eac0c07e7df7028ecf364ed20f12d9adb82493",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "a5bcb514da3eef88d1b71444ff78e521b8163127d328b270504c161a13c24b29",
    "sha256" + debug_suffix: "10af5be7146328ab3d32ff74c4fa53b072301f48e5c4fbb3ae446dd9d14a7c78",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "5524ad2e3a85dba46848861c20f23e097fbdfa84010f64433ab8a1c3c12a4f19",
    "sha256" + debug_suffix: "6da9488a9944541835e267bfabf354c7007e79147f546f9cc532bea541aac813",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "637e57511e786b630835f8285a1a56f4929f1c05c5fd078320b80e5ca2677127",
    "sha256" + debug_suffix: "1c519baf81b3f84d796a77cda39e14b7c5734847e8534dbfcebe29b7162aea87",
  ],
  "kernels_optimized": [
    "sha256": "a6bde6669559f881034a1bc62d5141d2284d4418c5fba18f99aa0c5334bed4bd",
    "sha256" + debug_suffix: "eef58da16379f2ce8ca226fe3de54c16fc0f4d4e543c368a12272265098b729a",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "fb62417bbe118b478762bb3779dd2268988c668f54a4a36cb810a8506504872b",
    "sha256" + debug_suffix: "a6091af2510c19ef565d1e565e2adcf56ef2bd22013f2e605f569f07c0762a3f",
  ],
  "kernels_torchao": [
    "sha256": "79eb5f96c878e1c359dae3b8cec03520b484b98d6bfbcc19ed6ee11897af2ad0",
    "sha256" + debug_suffix: "7f4ba07999a15ce96e4beb382a10d7c78698c36658677cd1f6217b7e5dffe3bb",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "bfb83e8524dd10443a8e8a1cdbb9dc7f66eaa18b987f045d4043d5e42103a8eb",
    "sha256" + debug_suffix: "e24ac2137322961c422c281c583349f1e22b9cd90e29e91237dfef41531ff2ae",
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
