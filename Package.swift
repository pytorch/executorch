// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250626"
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
    "sha256": "5983c9f2c59a2fbefe460a61fc3172b3f5576ed44cb805f08eeed469d33fd072",
    "sha256" + debug_suffix: "840fe6d6fe0cba5351c6d02a59e7e418f1eac7da95cd6fda9d5e32b1cfaba1a6",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "483acc5589b2d7bfc5657d125138f766cc9dd4247fc13cf434b1d74db0cb3410",
    "sha256" + debug_suffix: "71beb5055387fdf4d091c4935d6af803d1e40334074dd4f85dbb8e7918af831c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "33854eff12db591e76e7afb758a855524eefa63e8188baffb5fe6e7fa3927404",
    "sha256" + debug_suffix: "e0a34d5d67f738d0a18a0e9e35c3db27045f6d26afb755a2995089895422227c",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "9da07322f55539a72f82e11fd7924d9b6b256996d4ce1315d2a3f5224a21ca2d",
    "sha256" + debug_suffix: "67453c3c2b171ab7af06d955e7e0bebcfd109c9f5e6261b6750f43bb6f7af0b1",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "0a130ad0314e138b0d91b2dbf066156da5766d86c93cccfd2b149fe4acc6d97a",
    "sha256" + debug_suffix: "c96f288950393fdd2d41b6e05d36202802a1a125b60eaf7e65b707f06dc6b1e7",
  ],
  "kernels_optimized": [
    "sha256": "bb3667ed8914a828ac011f8cbfc45454c79859a1d5ba665b44a42a4e2aab4e4b",
    "sha256" + debug_suffix: "3b2680c03d86b271753a5069ba015fd8870dea0a917188d1edaa3d8ee53a0562",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9ca7446bbca6d75c5423bf66605dd57798c143fd885113f52b74334c390a1d17",
    "sha256" + debug_suffix: "0131cbaefb0e0b6d9bba0b341063addaed9a939b35629de677048324da182e3d",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "ffcefbbd704b7651bc6b04a5492482c113684fb3558abdc7f15a7ce774546fe7",
    "sha256" + debug_suffix: "44f4ff5ec406db88d241a505cbc44968cd56a70ddfb02af186ce59ea7ac30840",
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
