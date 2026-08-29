// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260829"
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
    "sha256": "ba3ff50ce60fcc675637795becdd44df4d80175254f20a976487a1d66927c999",
    "sha256" + debug_suffix: "b1b0b971ddaba11176dac0f041ee6203928a9cda462eed6458ac13d81da27ed1",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "ccd74a2274027b22412d52455e380061416819de86ac1a396658bc933448418c",
    "sha256" + debug_suffix: "884997b6860fce20ad11cc933c33b0824dc589a3f77ebd1c67fb944d67132ee4",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "7e288fb8db1996dafa0a04685b7cd52ba34d91fe61e19dd5bc0b26fb54f69c52",
    "sha256" + debug_suffix: "00aba30f6b90577a4bf72616aaf64da8846786d8fc505303063b6dbc6e895d20",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "73d4f4004c72ca7e0b2dedebf21cba0884050f4c02e14dd33ce44dbe255056d7",
    "sha256" + debug_suffix: "7a79764f820d6742997daea18670a3aeb7124ba789020c839a1f187dc4f8c87c",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_dump": [
    "sha256": "79ab7a1cbea13dbd2f6da0151bf543e72a61051c7d2a07c09bc32ba94cd24737",
    "sha256" + debug_suffix: "bf64a1b69140afa9a365db67dd19f718103ff3f078f8893032ecec4d0d57f3aa",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "7ca4911cc8bc703f015016be21503d382a3acadcbfb34713dd04ee2fa4a8b6fb",
    "sha256" + debug_suffix: "5d16b97780d8d9f101b1c003c87b779b5f82f67bdc856ce313143f9169630170",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "dc89b2cd8f84a95866695039199487109d1b2c5bf284c295d40c8cc4dc013a81",
    "sha256" + debug_suffix: "17d3f67d714bf2a44e6ffabef58f2383311758d59cbb56026297c48572103c8b",
  ],
  "kernels_optimized": [
    "sha256": "2535f6e67c4c4af5f42ebeb71e3e6ad8417d64c013fc1131cf5c5c6341f6045a",
    "sha256" + debug_suffix: "c4f3a3016663549249faa99808da9df54a15567f69491224b8d25ef86f0db4ef",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "c85dfce5ae4cc070049ccbfd361110c184eee1097c9c2cb88798606014ac993c",
    "sha256" + debug_suffix: "08f885eb37572367cb5449fd8ac0279e8c04e6a675f314fda3c7b51695278209",
  ],
  "kernels_torchao": [
    "sha256": "5aa45ea287586ef6e107a613e385c7b3438112fc9d4a6526c97dc68d6403fe4c",
    "sha256" + debug_suffix: "d08fa81b7dcab7b6ef4f2103d6357178700352a55ddbbbdb7c0a9a9baa1e9744",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "70dde166e3f1c717f7047e9765f79bd90fe7eb1f1feee9a84613e29038c640e6",
    "sha256" + debug_suffix: "e06a5a127495540ed3b0aced8483e7e7107abbf2e72b9a7c02ba7d6b86e06c57",
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

// The MLX Metal kernel libraries, one per platform slice, shipped as a single
// resource bundle both MLX products share. Kept out of the generic loop above so
// there is one bundle (executorch_backend_mlx_resources.bundle) rather than a
// separate debug copy, and so the release and debug delegates resolve the same
// name. Each slice's MLX binary asks for its own mlx-<slice>.metallib.
//
// The release job commits all three files before publishing, so they are declared
// unconditionally. A missing one is only reported at package-resolution time and
// does not fail the build, so the release job has to assert they arrived.
let mlxMetallibSlices = ["mlx-ios", "mlx-ios-simulator", "mlx-macos"]
if products.keys.contains("backend_mlx") {
  packageTargets.append(.target(
    name: "backend_mlx_resources",
    path: ".Package.swift/backend_mlx_resources",
    resources: mlxMetallibSlices.map { .copy("\($0).metallib") }
  ))
  for suffix in ["", debug_suffix] {
    if let index = packageTargets.firstIndex(where: {
      $0.name == "backend_mlx\(suffix)\(dependencies_suffix)"
    }) {
      packageTargets[index].dependencies.append(.target(name: "backend_mlx_resources"))
    }
  }
}

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v14),
  ],
  products: packageProducts,
  targets: packageTargets
)
