// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260512"
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
    "sha256": "1eb91fca05750b1477bb3bb55ec3b1a8d915d3338cd132a5a81a9bbb64ed7846",
    "sha256" + debug_suffix: "e1ebcaac49b7808a8711f514769539f5ebbf5a9509afb420be5bf1e3606cda4b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "3006a1db20e96b1b3418cd3159d6672b1d65318249341a4829b37d4910647d5d",
    "sha256" + debug_suffix: "a6a3d351122fa489b55f17c0e9ee7bf5e83511224a1fb7f8d72f6d3c1e11d0c3",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "8863a800fb63fcde7c4b80434aeabd5f97079bd872c26937e81ecc40af72bb89",
    "sha256" + debug_suffix: "2c38be137b30d853178946ed65ed54b3f6893f5a4e260137d99df7ed9376aaf0",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f170ccd802fb36c1ba3f82fcfb6fec5166bb75c8c01f126b719cb0573f559cff",
    "sha256" + debug_suffix: "fd5402913be4f064705fe4781521d7ad5764192784a815933b2be5bd1154a4b0",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "84e4351ce2955b85210804e0d9f404e59b4b6060138b5d2d65483968cacf9dda",
    "sha256" + debug_suffix: "09ac94ffff03016f84812c6bc2ceea8adbfdbab0a294a3a9f58c7d65e9843fea",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "95e94a435b2b5c6ef6b892c679ecb0dbd8a9b30261936abf7c0bbd163a7a5085",
    "sha256" + debug_suffix: "1dcfa805cb301356c3abca2a53d4b56bd23147208c75b952e37999dcd5b87bbd",
  ],
  "kernels_optimized": [
    "sha256": "05c708ece2fc1f098a852806c974d9f4d43c28ac12f2bea28f66e80faad13849",
    "sha256" + debug_suffix: "51bca08bf3d28deb531c9af2cba3046816cb2b6f33f53d6d8f2bb552e835996a",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "f5adb26fed91adc8ebcfca4cf6fddbeac19b4a897ab9c2f2c635b70ef6aa2fdb",
    "sha256" + debug_suffix: "f46839e7cccd6e471b7f811b5a359332bc4235dbb8c4b96642a8cd3a15e088c5",
  ],
  "kernels_torchao": [
    "sha256": "126703d9c3017d730a192fbd46938938e13cd8b61fb24d20307c525e9a4ae9d0",
    "sha256" + debug_suffix: "358b08b993831a5684654d5c36cea7e2aacdfaeca416fa38831c000095cc74e6",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "2f3c1ad09bec6e866f0d1d72e5571cbde01e1a17a8deb346507dbfc6ba78301b",
    "sha256" + debug_suffix: "368d995a3d4cf522e787b3c01cd86690a6e9edcbb46398aebc7b3b4033a6cce2",
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
