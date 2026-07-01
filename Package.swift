// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260701"
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
    "sha256": "022c8de4d37e82a23ee4a0ac9dd01cb97bcaccc3f0e822c4afb0e04d6fddcc48",
    "sha256" + debug_suffix: "48ff0350324fc07c69f529a3c7f7738e355174eb857bc5d2e9ec10b4098b2668",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "ebfba625e8b2b3b023a323adbddc42b7d93afb5c01ba51d8fe8d4b6e2e61c100",
    "sha256" + debug_suffix: "397085c119cab0b006de1829748768d60fef3c4098025a8f50822f8206b55b43",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d560d659a46b5ddc2ba1d98da6ae7b781ae171dbe4871f3f749988427a1af821",
    "sha256" + debug_suffix: "3c3bfa91268ada03fffc2eb2c50ae38219285d2774fabe5a6f8370bb51a0706e",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "0b0eb1b48f039a6749216de948312872775b2325cafa5de9db01f3b4912b17b0",
    "sha256" + debug_suffix: "d9801bca7d058dc0f4b0ca01dfd37cc713d3848146005892631c7edc27ec48d1",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "7f8dc2d718eb45176054007a07f13cf21b0d9c47ca25e31cde0546a11c89cb6c",
    "sha256" + debug_suffix: "1fc58d480f70690c003b077fd85d389250584573c7ae9b05dae534769fe4a0fb",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "097a1ea015b5ad2177a97d3cf1e56015c97cd338dddf82eb52d3cb29254dc59c",
    "sha256" + debug_suffix: "d0815b72e9c65a289819281580f6d33b034870245a442408e4d8d3bc75bd00c3",
  ],
  "kernels_optimized": [
    "sha256": "1be3ab78d0ebca2bf30676552dee68a1fb4cf99cd65873d31a8f21f34eacee57",
    "sha256" + debug_suffix: "3285f60dce696f24dd6bb2da3063bbef9d4ac5745fd3ab329db40e79ebcbd411",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "f1cde304b57046794cc193845b48444ad4139a578f5807057901e898bc7d7db6",
    "sha256" + debug_suffix: "7042ea6d704459cc35fe39f457e7cf03bdaec4183c0c97d835c3504968980562",
  ],
  "kernels_torchao": [
    "sha256": "2bfd0cf4b9f0876d2de26d2c7f61d1ad3fa99e0f45d8b495429815447b8e68d0",
    "sha256" + debug_suffix: "e84c57489267ca6759c398b40f6483e1969dacfb70c939680ce098d30cbffb3f",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a7e51697595604627d2adbbf90977ab0df616e8eb4bfc1663f8eb3a239f8cd14",
    "sha256" + debug_suffix: "03a316da4c49b099fd72d488c297826077b52ea4acd7451635e9626da9f4cd6a",
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
