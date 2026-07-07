// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260707"
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
    "sha256": "e7559a4cef7ef28d6f4c9fda3f88219ed3a37675085cd7a88429b92ca41ce65f",
    "sha256" + debug_suffix: "223554b43d84e2b9b3232c63b1bb60b23bada9bf06cc76b75d0c17c0bf807d3d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "3399d9a05cca3c5a075bd04fa38f43625037fb3289df21a7807e3283d141336c",
    "sha256" + debug_suffix: "dfcca4f3ebbe2ee85c2a1311c5e1ff3289709d7675e914bf048b1db832b3fc34",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "cca79a5b78c499e3c0861d870e9ad2922c54276c571f4a5916c1feccd424432d",
    "sha256" + debug_suffix: "41e9562c8dbcf4b28388c2f37a7c189a452c423e803a06879788d7bc0f4cd116",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "21971961465923720bfe6ca3d5338a3fc6e92dda98e226e821731342094f74c5",
    "sha256" + debug_suffix: "aedcf2f401e09b7e95b356c6f5378d751750b8e512f5fc7675e5ecb3ee8733b5",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "6e4227742d8bcbda8137258e63d2b6fcfd987394c32be864e8e69a0a04fb8fbc",
    "sha256" + debug_suffix: "d37bc3ea086029cf66a6389c4030534518f88ede63d0804932f642e1fd55f484",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "f4467ec9fd8383c8ddc2839ea3c1b34300f74822cdf1ccd22e7a6e98dc4318a4",
    "sha256" + debug_suffix: "0a4650b2a5ec3b7187720d579c32674bc7dbfdfd7ef8e93bd38ac42cd22470a1",
  ],
  "kernels_optimized": [
    "sha256": "400463c2e56adfe5bcb1d8fea29e282d99ccdfc09a976ea2a2bcb3c6f8ba116d",
    "sha256" + debug_suffix: "eec0c1014ab118bd3242fea584b1a38aeda7b5ddeab30629f9cee3c22309fe1f",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "0f5854bc4c454eed9d1aae078d357a78c2ca6b83913a25498f95be86947b1e87",
    "sha256" + debug_suffix: "7dd6cf775cbb5171c601ae5942cfe9164fbfa4044d08d341289f0f909954dd4b",
  ],
  "kernels_torchao": [
    "sha256": "39077c28e2c17e243f0166317554e845fff4fb2aa328331de5c75a28d4318bff",
    "sha256" + debug_suffix: "ec54674af77ee879439146cbc5a0c03482cb5723f548391dfba25f84f793acda",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "efea86955d7e3d3cf40f9c99efed2e494f5a64bb7532534fb2e5cb0becc87cd0",
    "sha256" + debug_suffix: "00b9d64b5c5543697c181960416d5f556a48672eebdb790a31f5a393230a7ae1",
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
