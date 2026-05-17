// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260517"
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
    "sha256": "b07d19bdfc28dffe43a0747ee606e70087abd1ab112fd867525794b43b686df2",
    "sha256" + debug_suffix: "a2a95966b24c2e0fe5146dd8837a545a5cbf15e669ab5b4c01eddd9a1d850f2e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "0525f54a755a45034b5109da422638779b5252abd42e7d86b9b92f6ceee2693e",
    "sha256" + debug_suffix: "483fad02e70fc0ab236bc4897f51a29814aae6470085dbf2ba9310130bfe8a36",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "5e4b6713c40c05140973de017d555ef22eb8f3a3c903f25d660c7a092895ce0c",
    "sha256" + debug_suffix: "cdbad026b6fd4cbcd15797e6795a3f9aee03f175177c3b631d3ab36e2e7a76e7",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d7f2e61a00b398ee126c095579ac50eb572c0bed685a33b8afe41fab6984d2db",
    "sha256" + debug_suffix: "5366b9edd7d16f0b4b28c064abc82b519e2046e44ac86b95617d3db50526d681",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "ae00fdd664f6024d95ffe74cbb321d7f7873190651037b5f50c4fc6202366d33",
    "sha256" + debug_suffix: "e8fee201e83ad4c64b227966490876fe2c7dd04fbfd2e2c79cae7c46bbbda983",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "b18c3a443a158ebe488da0cdf6a5fde47c6b25cc83303f675788d3bca953579e",
    "sha256" + debug_suffix: "1260a0be3a2e5923aa3934ac51c36c82d4c83fdf7248042265d73c572c3892cb",
  ],
  "kernels_optimized": [
    "sha256": "e5e70c1b8d9aa7697fd724c78b3a8bd408fb742632ace6b209f47df0c1465f4c",
    "sha256" + debug_suffix: "cb738b4e5f1ea09b83dc85b497ff2b419a6eb6e6bee72b1342f6ff7a09ad269d",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "6cecec9c22968a3de13609728938fbd7699ea3defccbdeb23eecf4615894bcb8",
    "sha256" + debug_suffix: "57fce5a2e50fb6035f366dc51412ecff94b2885559503b716b1a52d5c32517d9",
  ],
  "kernels_torchao": [
    "sha256": "fec7d44cbe9b5cc2848c1563fa7f7175e768cc5f40c298f10312c9821cf6174b",
    "sha256" + debug_suffix: "1d9569ae6b197cd74e457cfe0d02695d9cde3fa05e50929098979429863d6ada",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "3cecd53a0d206f2632436cdb4fce1bd917d60d6f1b66a56fadd4bb2b0053891f",
    "sha256" + debug_suffix: "74eb90f7d3b105c99e3cf81a7c64522a6898c78c5a89822a7702a718e0afb289",
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
