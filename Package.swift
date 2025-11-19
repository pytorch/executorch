// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251119"
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
    "sha256": "99a86a6bf110022bed8219a6144729a22844c4f82cd5b3ddf31ca43085ed6494",
    "sha256" + debug_suffix: "7e34e12ef8be2ec4442c619beab4e2cd0203a6ea78783f1376cf2aaabfbd3321",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "f61ff02fa6e7b81febe09cd52c95089a9b65c97e3b70ac115c5a445aac65918f",
    "sha256" + debug_suffix: "782518eefd15a21d93a39e88839d1b3f77c7ce9534808928f6e04c8a40dee584",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e29f39bd9d1351809d6f2b7e072da09cfb22f9743595d89335a1f7e02b61d234",
    "sha256" + debug_suffix: "c1b2389215ed1eb415754a246d8c11b269c65a66036feb06a210839a12c9eb24",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "354d0cb6f1d36615d01cd6ade00eb5c197f521e5f608bc315bcfc2c26cbaefcb",
    "sha256" + debug_suffix: "24cdb12f08c63b54afd251313ae11a7e9fe74c3924c2118e7c3b3914168aa7a0",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "be1894c8f2adea9bd576b729a2ae49314ba352aede75293cea392fe4a797d52a",
    "sha256" + debug_suffix: "4df52ca0102ef3f9592246ea2d6d7435e6e3ba59a98a29c00c2dcbba68bfccba",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "643a0ac5cbbddf5c688e01a3884a1efe8d87736efd06ac814501f825fd3d084c",
    "sha256" + debug_suffix: "210e4bf5d129f34470a937f4f134852a61ce430dd7b9174ea43716db1bf7c994",
  ],
  "kernels_optimized": [
    "sha256": "585a6284593bf264b75c5b6b5dfd4e11d402dd4072565344cbf8c11cb1672444",
    "sha256" + debug_suffix: "6d78ef57aad4ce4985e2899b072985886361264690f961808b8292f37e33c8c2",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "faa1f092b7c72a43c87ccc91ae9c9cde661aa0d9a38e459cf4000ab18da88910",
    "sha256" + debug_suffix: "588d0793f16ac6d0326038ab35e55996151ccd96997a2ef8dd1dc21834dbf6a9",
  ],
  "kernels_torchao": [
    "sha256": "5a951d10d41f990feec0e3c79b0ca2ed7d0960a92c6916d99c7d0a241858975a",
    "sha256" + debug_suffix: "8fbb4970d21be53a8f2d44dbdb70d095033843a9f5421fc436c0f8974337fbb1",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "014add2ab2854748e5a8929a477419fd2f6b63cbab1d8ba8606be475081e888c",
    "sha256" + debug_suffix: "36fe9a1e6cf95662bb5eb55fcf963200ffdf97292eeec46e5ecd0406c4d3fd91",
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
