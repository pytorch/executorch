// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260221"
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
    "sha256": "b5de43a3ecfccb7b5ce0ff1d1bd83e00f7f8db4025d821cf31307d624870d704",
    "sha256" + debug_suffix: "ae7b3eab4f482486d4fe44981ae44018502dffb6bb5cba4e68c8b0ecff268dd6",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "6b69befbda9fc70270d33f015a552c79298e08448d67f728b99174d14ad088d7",
    "sha256" + debug_suffix: "a048019a518f05a8b05d2afd2828d0309198085b1ec3898abd18d53453b210c0",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "5c480ad04a9b3f60fa43a573d1bb46c386a1b9659237b350972510d11d48ef28",
    "sha256" + debug_suffix: "a3b2c15beb5daf4a72d45dfc21566e09ba2b87980b5c29e2a46bc1e24eeaf93d",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "bd8fb8aa2086cd7174bcce912ec3e216ed14aa3af15cca64048884df7918dd6b",
    "sha256" + debug_suffix: "b7c9ee3d39368d47507fde7580e5b15b86a7764891c5130a762c9b71abbaa952",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "a98319478115057d9ee505f1c47fca0998834f9152133bf3e0f783a2ce3c4a8a",
    "sha256" + debug_suffix: "69f05466188188e5dbc2a609d37f00bfd5c539534e262eae8375de9884490705",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "f72893ace9b0185eef4a8ab70301fe144d61f75ba84e85ab4343d237c2353c71",
    "sha256" + debug_suffix: "5b37d36b50e1c6542aa3b7f8989849b435841b6a6b8db2c3876e29a02f0f581a",
  ],
  "kernels_optimized": [
    "sha256": "88a8bd99bf93ce356ab4614befde78083c97a73b6ef5810bc5351bad8a8282db",
    "sha256" + debug_suffix: "09302dc55d7e4296bc43c65edfc18ce948cae6e7c42793672e69d70924e07507",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "5753d0c1d3b2bc11e91f23bcba3be0d4c5f9108458537bf4b4f72392f65e724a",
    "sha256" + debug_suffix: "7ced9f7590510d24ece8340e9158c8c793a60e3072f4994fbf4338c50ac04d19",
  ],
  "kernels_torchao": [
    "sha256": "9b8793521e39ea3c1970efd2f3c6338d6d6e27c757085625a9cd5f88d4606013",
    "sha256" + debug_suffix: "8ecd2e5d69c36691f729249fb4173922e709c38bc9f4b2f0d59be8cf38e16d33",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "32ebdf05fc6c22d651ca7c74767f9f7ac52329e590421c216a298d75a4d9201c",
    "sha256" + debug_suffix: "0697f1eb4937817fc96b14dc34a678d0dc2b8b7069618d4330ada80b96325cab",
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
