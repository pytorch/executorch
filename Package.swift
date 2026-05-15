// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260515"
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
    "sha256": "5637c792d1bfb2d05dd1bfa1fa790df8e99b4197e0a454c26ef113c25447fc8c",
    "sha256" + debug_suffix: "b438f00b7df5139cbde448e4402dfed924efec62653242c5d992680667b458e4",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "12b1cd7c0a359c62e1d4ccc9cab66e611a243d757f4b460d759a2061ecab2de0",
    "sha256" + debug_suffix: "78c6d227a307e80e8ba9f4a35c9d46f3855ee92cf36d5a2b8d9bfd86d2474f7a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "cf862492a4faf6cf2733cb4eaa6b91879ac6d5edb38c61fb309ffc4f6c2e479a",
    "sha256" + debug_suffix: "67f0fc4b88acbe632a130b85fe6a68d4c156ae45492d4c48b0729d01c2bd6c28",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "39e45ccfa6798f31d9f94c45b1c4040162709843db9dc2f6eac22fc9813cd54f",
    "sha256" + debug_suffix: "0fb858467596512bafa8e673f533e6c9eddecfaa21040ceeb54492c9393e8141",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "1afead6799cf496fbdbb03e9797ebfe4bada6c271ae1f894b4a27e0910ef5cf6",
    "sha256" + debug_suffix: "4a3cc31c4b1b524a334d1d2dd4fdab939b3b24f6b290a20fd3b9cde8a328ef73",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "ea655db9ceafc4d03e7a734273bd57e1ea179268df02ffc0a78adc9c1e47ab8b",
    "sha256" + debug_suffix: "83edd95fc8e7c6246f698fb67fbc7e124676db7befe517628d0ea51e9ed48dcb",
  ],
  "kernels_optimized": [
    "sha256": "2289565773afdbc5b1bddf5cb4f6c8632dcacaaa34e23f83364e5889d372e9aa",
    "sha256" + debug_suffix: "0646dbb54d74155eeb7e95fa0bc58358505f6105aea17ebf6f331e79ada19091",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "f980ab12688fed3714854ddc24cfbdb7d32c69d801638ac54174baf87ce52c77",
    "sha256" + debug_suffix: "e158b61bb3e2ae53e26a14b71c79e86a108e914321b8ae551f2093a92f123cda",
  ],
  "kernels_torchao": [
    "sha256": "aee0feefaa10882cdb8287c5e92300077a38a1b77c70cbfa572ce0218634a443",
    "sha256" + debug_suffix: "1bb878c420e11e6f2b50c7ef0a167b77c42872ec9921a30e4ec7e57dfa38d5bf",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "2c51d64cbd6d6f7fdae80ebbee2ce2ee7a6ffff80174177ee80b11da41e6f514",
    "sha256" + debug_suffix: "c81dd981f4f2c11568decfdf7538e21a2a4d65ffe5c271be733876fd2d4353d0",
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
