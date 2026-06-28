// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260628"
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
    "sha256": "0d36f95ff2616d7713098661da71f3fa87f68383a7281928eca680ada9a68baf",
    "sha256" + debug_suffix: "0c2e4335c876c425f59aeb377532c6daeadd5ab30866a1e525f842e16fab2ed8",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "39fc382c7455dfe2b5a9f4c7717ef81c26834a817321f91750db04fe428dc4b8",
    "sha256" + debug_suffix: "ab82576e25bd482bd7127758858e154356860924f807a86133f469eca9c33609",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "96757e4b07dfff9ff0b52790a9c4df95b6272c8c54d717ca01b5b67b65101507",
    "sha256" + debug_suffix: "7b5d30523e486cc5dcf5299cbbaf1de6430a31fef0ab98ae4667b4ef73aca514",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "5df743ad8f6fb907ae4fa97afeb84b84d4f963067f5a442e4430bf3d223df8a5",
    "sha256" + debug_suffix: "bab34423bd475c4f690259c14198bfaea7101984d5f0bdd359bc6c488ebc8d05",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "3f87f4e91135c39fee4d4b82490169119072e529da1682cdc227ccd455be0099",
    "sha256" + debug_suffix: "97dd6301d930970741d5f56c281cf5a4b360746463b262a7870721e098b370ce",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "557b47ab537f65342e6e0aee373236354223575955696a59d70e5313d732ce53",
    "sha256" + debug_suffix: "b62c3c1c12b0c6b7e3b467cb60258d17fd87d766d011cba56b1318f40110ddc5",
  ],
  "kernels_optimized": [
    "sha256": "27fc5396956fad436b6e82e6592d0ad865fa30ced26631c5b4b89bb30da96c13",
    "sha256" + debug_suffix: "bf2cba587f4d72baeaf52ed6b0df3feaa2aea92db5f218fb4b75816cfb773b73",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "99723521e6f6f425de31af33df18ee4eacf3e3c880aeb22edccdfde7f4ecb63f",
    "sha256" + debug_suffix: "21d3fdedb2a7943e188dadc73fd593b088390fae3183a4cf1806f05288a7e7f2",
  ],
  "kernels_torchao": [
    "sha256": "64c424e079423949972f7235ccf7cbb4273ea4eaf6aac22cb2c4bd9b65f365a2",
    "sha256" + debug_suffix: "75cf69adc31c2fcfdb29ccc2263d4d71b747346a96b6d92ad71a4e60730decd9",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "c3026004bbab8eed17509e4fa28f50872a5b283962f92db3be6f212429972a07",
    "sha256" + debug_suffix: "89eace1498568853d4f6829248934e469a63f2bda94f8003e0b8af18f607dedd",
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
