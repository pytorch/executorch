// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260620"
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
    "sha256": "4510fd5d33474c8454dd755040a76194ec43273065e04b690ddedc63426eb2dd",
    "sha256" + debug_suffix: "37a7a34ab1aa776e4dc374a2fb6cb1ef1092eb1a2f9d510c50fe1a16d6651981",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "3bb0f8510a6a0930a998c053b252274d73ebcffeee130ca5abf70a89bade7386",
    "sha256" + debug_suffix: "8b7d3104e0e6e4398f1b4b952948ac1c01576bb5088c2af1fad904a138a9f45e",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "8a790d1798a4c3db208de0e1467d61e45649b606ebbefea15116e9a906ec0210",
    "sha256" + debug_suffix: "fef5959f81eb51d714aa62421fde0d2efacfc8c8e5b5d0add76c29de97ed9e32",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f3dce4d001ebad69bf942febd382417a60ec64256d203a4ed01d419a03bcbe8c",
    "sha256" + debug_suffix: "0152842ab497e0c4b777f0ed4a3b27b6c2a80c7dd6d0e2cc7a049bd00d684422",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "00900397198a4bf72eb9f34e420b3269f0420c5491ed2e45510e3c9413e9b89f",
    "sha256" + debug_suffix: "04cdb8f889c41c650d11b73e94e7992a788843775501ee2158b126a57be8a2e4",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "604507d06ba45f49b7084cd2837f2017d3ce38aba799b148cef8ba2896291df7",
    "sha256" + debug_suffix: "98be95b40647cf28fa38764ca740200d07ba33726dc768ce86f2c0cffe7fd894",
  ],
  "kernels_optimized": [
    "sha256": "fe695153ef542990cfb2838cebe13144731f7c20cbd3633a8868543f93a88b88",
    "sha256" + debug_suffix: "f9007680ca33b6d9503d2507ec9c1c114843f0e7878309c3a2f72566a65f3212",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "af4f42eeef9099a55538c8b56e36c3a58805b4187b5b61c051dd7972fa03478a",
    "sha256" + debug_suffix: "0fbd02bc787db2b59efa75938278b0a45ca15f78ea0ca78f5f446a7ee8412a72",
  ],
  "kernels_torchao": [
    "sha256": "7691354a6e118a9e0cc89f2df00a05a6fc7bcee067e315a39990ffcc80138f75",
    "sha256" + debug_suffix: "9e6a9c8950beb1b57eb17399eb0dc83a7c9165d33f2ddecf9d9fad5e15c6ef93",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "fa5c3653a8be17e46ee042e962573d2cd97d733c53ba000cc004a61a1a1342a3",
    "sha256" + debug_suffix: "5acdba4a7cd41735aca1c393c3ed7aff8956c259a9581c379ac14a2fc7e01ca3",
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
