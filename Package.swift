// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260511"
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
    "sha256": "98051fa315fe2ac35f48ef71fb463ca2e503a15fd49af885c46ccc5b6df8329b",
    "sha256" + debug_suffix: "c1c0957e56828465f7f7b96d11fc6023439368b7c93e6a7ed4a02436ecd64e7a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "d057814f8a8975da88ccb64a5580f0ffd6cd1873407a67b6913b6e9ea033a5b6",
    "sha256" + debug_suffix: "ac6511d8ed14578a7953d05d332a706e3bd06d2715e2cbc29bf655ed3327d96f",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "0567bfbf7d8b8003a220c4f00460a8baf5ded10c64fd226c56e6277310d7b7a1",
    "sha256" + debug_suffix: "ebf94a9c14aeeccc4999e349b59efd459ce4349428b50652d8f750fff13b3a69",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "664e8f7d1af493c26e291c0650b36d2b478d29b9b473f51ab0f963c8c60fc9dd",
    "sha256" + debug_suffix: "f20674fa5c01ebd8346eadc10a668dd8656d108505d82096d877da82060acc6c",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "14912dd96272285ce0830b52e6a15b8a3aabd45edb88a120a9db0622f0f0cbdd",
    "sha256" + debug_suffix: "ad856a26fae514cb8cec9ab3be3d3fe24d18fe6e8eb01dde5c81b0c280f23a9f",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "8e4cb70c96e4612d64acbe12013bff2638c823ea9a73ece2fdfe923f8de24935",
    "sha256" + debug_suffix: "7fcd84c0c73428ef0fbcbc9f7bb463cf9fa863f644ad1cf247288e996c3f491f",
  ],
  "kernels_optimized": [
    "sha256": "68c4e92f56ec09bc224ef3e2cbb38a0e374c81346a78a373e1962dae22026a0d",
    "sha256" + debug_suffix: "2ef044db601b969ba55e14ad9cc6f5d1f0b355246144b7170b6a92db2a4f0a4f",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "7b10181342210c60766032a7a45fd78dad1db94d0936eb57f0fed563080c4658",
    "sha256" + debug_suffix: "67cb62ea521b5ed7c48fb7c9159efa147ec40cc5b5525324cae7dbda714cbf92",
  ],
  "kernels_torchao": [
    "sha256": "cdcfefe0a8ae620c91f09e567f795723803b1ea08a50db253b47fa42496ed467",
    "sha256" + debug_suffix: "55e90cbdf8a98b7aa11cf8b7c9bd156920d53fe9c0d12686a3da3191f32ad625",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "3f3c8a03c0d3a265946223ee4c2c9e93056ace33b36fac9a62ece3698eea5338",
    "sha256" + debug_suffix: "0400e9d9e4e6756df0a598f8731fb3721e3e3af54f013cf7887681d37efae502",
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
