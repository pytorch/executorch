// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250720"
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
    "sha256": "ffecdab319f2964fff112e09a0837ee218a66013a58587c02fb03471ade8c53b",
    "sha256" + debug_suffix: "83fe69cafb18aacfd52d231a11a1e8c15d049bad170a4d746cb01e080761d387",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "cbf4ef241a17a784dbf4da7c29a50796431fc08de533f740107f377491908314",
    "sha256" + debug_suffix: "c05cafd3be657506083cbf6ce78de017ae8c6d07a64130ab94900c299badb839",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f489c42843837f470bcbddc4d0170554ea5073792e3a8971128ac7504f9a5820",
    "sha256" + debug_suffix: "4eaa443247494c0c2208720ff1e5e1731b05d64b73e590bef02e1dd216799ccf",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f893ef06f3b33e6599c6f7b468894f79f228959187a24140c4360213b4a53edf",
    "sha256" + debug_suffix: "075537b433c9d441df2179e4f06dcdbd82bd9a4da0a9059b826b9aac5728a9b3",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "dab20e7780761ccf16f77bccbe8dde4e6a1467be1b5a53fc7b49c9b2eae02665",
    "sha256" + debug_suffix: "98217612b63b539d9b333f6ce085b5ceac96465ccbcdeaae6934f9097ac8dbef",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "e8ae852acaf19642a8fc4a3e1f4f556de1eae359767eba719ad7e2f17780d04a",
    "sha256" + debug_suffix: "8dd4a4e0f8a08865956a85053141bea15444862b81e664d9e34b93187953cfe1",
  ],
  "kernels_optimized": [
    "sha256": "ba77636532c4cdcf2c09fee29edcfac2fca800f3d470f189699f8d618cc98c2c",
    "sha256" + debug_suffix: "2c9f7307199efee2932f94dd5496ce68b1e461b48be9e90388f7241217c19862",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "af5569179a3f1a73f6b1677034bd0090631ef43a9afdb89fd0fd692172c65be8",
    "sha256" + debug_suffix: "63067f2753230dbcaea704d7f456ec68b0aebd702a1c100eb74db86febcd460a",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "bec99e5dfb1e6efa84c1d86e4c3850fbd3817e32dea43f09792c1d205b110f88",
    "sha256" + debug_suffix: "e5011849a4fe0d0b71366a19d1fed16ce5766b193e7878fd4dda6991d940448b",
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
