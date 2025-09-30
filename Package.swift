// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.0.0"
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
    "sha256": "880995a4098d1d46c8c1410adf3838b95d4176783834288694e1456761a0c882",
    "sha256" + debug_suffix: "d3edff026e018d0e88ef7760cd003c69eb232fb3c60d54a1cc2601e253ef2b89",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "5b863f989a34407c7dbacf35c311fe81bce724f7bfd7f52708f89052c52f5b86",
    "sha256" + debug_suffix: "8728faab1773fa307c233e08ffa0b17e33446c2a3bbf3d8f069260fbb40d090b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "3fce836839af0c3cd7a42d214ea42adff76c0aac8323e29b598cd11421d16e2f",
    "sha256" + debug_suffix: "c9f0ec3559c99fcaac98184f993672c12aa0c9ccc101ee790702b3bd1bca0ec2",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "676bb84f3c005ee8898dd5672fa370a6c45cfdbce0d3ca057b777d20c2a9ffdb",
    "sha256" + debug_suffix: "ef9c48cf2c3138ea478ef18c523412201a747caf7616a6343cc40c755118cba4",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "cb9c638c9168e0558893fd4db5031440ae7e79f6e55696f96c42bd0a160fd1d3",
    "sha256" + debug_suffix: "7b81832ee720c3e7defd08642ed1d6c881c2a85fcf76231dc0e32ce185d7873e",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "9c3b218e16095aff9926d69b4efc0fa02e58d89763025bd9c0231db018b3d34d",
    "sha256" + debug_suffix: "ee0a72b14e10d259fba37fb71f083f8db697cea0398d329f8be01200be5020c5",
  ],
  "kernels_optimized": [
    "sha256": "a71c200f58f6ce3191fea29fba0158f2df9cb0ec4508d22d4338488bfc2a015e",
    "sha256" + debug_suffix: "f39273a9001f9a0fe363268d05bf558119ce8cb0858847bdf4618ca202ae21db",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "497e17bbab7fbeab0005b74aac183f99675ad6b7e5078aa3a1e4e7030aaaeae8",
    "sha256" + debug_suffix: "dfda710d8bfec6649d27743d867e4791ce3278bfe1009121bdd05d5baf0845d8",
  ],
  "kernels_torchao": [
    "sha256": "b6a427963be25ea36216deb762c182644c63c8d354e5c753350b5346f322cc5d",
    "sha256" + debug_suffix: "6089dbbeefe6069bc921a5672ad7936b6b6e35117653b0208f946f42fda30b5c",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "ce1fa3824e6947047f958d3f5c37b4376ce36077f734b5643b102e414abb45d7",
    "sha256" + debug_suffix: "6f3e265ccaa5ee216af935314ffcaba9b40ca6d8fec86d1987478c42706733c8",
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
