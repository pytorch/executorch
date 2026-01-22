// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260122"
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
    "sha256": "97ed73cdeeae217857c0c0542c0a7e9a763c2d02d1fdde59becedfbc2127fc46",
    "sha256" + debug_suffix: "da50a4b7ee66a1d643d692064d7acad57f1d85659bff51d02df45ba66aedb7a4",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "3db8416026a8d276aadefa448c52601c4febce45e95ca66aabb55f52ac181ae0",
    "sha256" + debug_suffix: "703a138723b1150a20d2a3ce82efd5392e633f86f5720079281bfd22d712e813",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "92a8910ac7eccc683599bbeb9083ad8768487901fe07d5b564bfd7ce48f489f5",
    "sha256" + debug_suffix: "669305b1731f3151bc3b3cd22c71cfe5887cf8b2573f1acdf2cee3f45e63e0d1",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "3292e1ddc9d5f6c54cd49ac5018f5a967e43d544adc92d78f2c5019a44b15d11",
    "sha256" + debug_suffix: "080fb19d842409212b3107e801433f4864456bf39163b0e06c4e8bd83e5108e9",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e525d465d749abfe6b65af326c328567a3def4e0300442c58e3241c5cd4a779b",
    "sha256" + debug_suffix: "7af5be07ebcd4cda5a0f6d347a38a887f96589f49d8dc60f134e54f9ecbc2259",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "8b70e76e6f5320d2ab119dc6988fd3ce33de1e6c33a6f32c297f39e22580a6fc",
    "sha256" + debug_suffix: "3efa096e432c7ee7616279ded43416a7eddcb64ab919d40a8d82074c2581ddb1",
  ],
  "kernels_optimized": [
    "sha256": "3cc2bb001a661fb56e2423a7e06ee12cf7ea9c77f3943155a0b1935b51b2aeb7",
    "sha256" + debug_suffix: "ddfac3f9a4ea6b42918ae9f30bd30dcd2118b9731747a6e3107d58ff558b57b2",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9b404141e13cab1474038c4e6ed93b7993b0da4542adc05df295026ce866d741",
    "sha256" + debug_suffix: "1773cccc55c33e5b545a81e0bbdd08c0128b12b4228647a4cf1991a94759ab53",
  ],
  "kernels_torchao": [
    "sha256": "0eab5a680db17666784860ee29d242fe88b5d9e1516c8e5b5057efff10a674d6",
    "sha256" + debug_suffix: "30a77f4156db56cebfdfecb9e6af77a961269cd243aeea4fb535f32a567c5900",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "53d011451a91330e24c8c39c915665f46224ec4dd92e1a0b6306e42faa4650fb",
    "sha256" + debug_suffix: "5e61874eafad1d86776ad1519ba5573f459a02916d3535cf2843db23bf1ef8b7",
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
