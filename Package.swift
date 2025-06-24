// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250624"
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
    "sha256": "828ede664e81c88512943d9737db63bdee41c9a43cfe20658d6be1d38c9d9660",
    "sha256" + debug_suffix: "6441254d4776836892df96060d554ff5f9f1fa91284370f55574c953a7f45cc8",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "d2d71b9d5458e416db46e12061954a4a3de1fab27bdbc7dde11486136e4c2255",
    "sha256" + debug_suffix: "e48984ef4a36b300f8889f8037d04bba04aa16a9f95a07221d0b9cd160941395",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "0404936cd6e63fa5f41f50d0f42e317aa250626509d052bbc8a3a5efb88302c0",
    "sha256" + debug_suffix: "529158e9564f9288736d3c688d60573c42f84b343d8ef8ee1691129c63c037ca",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "af421c8dd2784d557850a14dc9e46403c819567485c19fe22602ddb070b9f573",
    "sha256" + debug_suffix: "f8fb0ce9baf98f602050a5057f010c8e0136fc563e66cf53adbaa7d6003c1a11",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "217d6b1b76e85d74e8db018af0bed01207661150d775b712ae00b524031a9392",
    "sha256" + debug_suffix: "8e899ef526a67342f98aa781072364c0997ca4d9df3c3d2fec9df385d36f5c02",
  ],
  "kernels_optimized": [
    "sha256": "4babc1b51a683139cabbdc48b72a5af74b6c70bc2257a11b4cfe807201b28b81",
    "sha256" + debug_suffix: "31dec81bef79e8379ccfa5d83681269c1d152b03aa594c7e5be43ce33556a2b1",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "52a8ecd29bf93b0886b0447e45df4ba555a7f1604690eefbd691f099ac89ef11",
    "sha256" + debug_suffix: "82656739812384fc6f61dab89cf7104708423732cbd090ef4b681c5260c1e1a8",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "20b44744bdccc5a6bcfff633ecc0b906fec98b2a26d3b080e4625fc7222cb2cc",
    "sha256" + debug_suffix: "7cf0b46c0737971e6183a99fc56247556e738f79ccd43e913066aed16073d0a8",
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
