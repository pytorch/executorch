// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260606"
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
    "sha256": "751843f18b5fb1a91613b04f94cb4eb18670ce17dc8047a0e89f3e23ceb41c9a",
    "sha256" + debug_suffix: "de7b1ec63ddc2b5b89d0b85058684e4bad3fb25b0382f09ec61a582c360fd1f3",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "95eca9dcb6d0365ccd18f5104bb34427b848c841887164fd9d309af9d2721b5e",
    "sha256" + debug_suffix: "504a64b5a89261b1f6a4054d11146438a8670020881ef1f0a4db19e8815d9211",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e70dba541aa085f0dc13797cf8cb7f7df2eeb6ce3cc5a18ba0b4ed9b412b792f",
    "sha256" + debug_suffix: "6d6442db74a602ead55732d4d5c7844a86cedbf43fdce46cbfb88870f05e4c0e",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "c3fc66460fa709ca1e1c71874b331482a99f2a853e5d5568745959699a85d9cb",
    "sha256" + debug_suffix: "a2786bcc88bf5f5d1b943ab06657231b9e898b4c0d14e8ae6777a3370f64e7b1",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "c187a722fd5473c2a7b643e32dab449d30d4d6077421bacf2d86fc14d2276977",
    "sha256" + debug_suffix: "9ac987dfe4d8e7d8a2dcb99059fadea043dd9c2fc1894dcb35beca4f83974bec",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "738e7bab6383f8893ff3794ad9bcf6c0f3f1abaf001a5450d23795c4454c450d",
    "sha256" + debug_suffix: "0730ea41198381c8587f4da0ce9151412a79ebfca9995825ded459897a440111",
  ],
  "kernels_optimized": [
    "sha256": "c1e73454bf15bc51a7f00534fe3dfa2deeea68f0c66a0e22f3560782dd0aad78",
    "sha256" + debug_suffix: "552e7889065a2780f1236e1f21afc1ecfe484b9c6aeeb0f644ac6c9f3f992d5c",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "db888040f77fd68613f4661f490d3d5b4c18a5c4b71dbe19bfd6eade092e6d46",
    "sha256" + debug_suffix: "7db2d0ec31395c1635057c670287e6a1d06f6895cc20703e58c8928dbb3cc732",
  ],
  "kernels_torchao": [
    "sha256": "6398df81cd051a2554794d85d21a6601e1d39024f7998eeb79ff6a75a5527087",
    "sha256" + debug_suffix: "97daf2d33bfd1c670e5100484451b0853e830c410d1c9831b6d53e83df1d8172",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "e74c589d44c5d83281dfe75a15eb1cbc6ab417fa19f65692e7f0c3fa33b495a6",
    "sha256" + debug_suffix: "cf7a0903fb08a310acae0b561d05d469b7a42084733aa0e72a7b4abe945c1e92",
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
