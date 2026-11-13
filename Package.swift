// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261113"
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
    "sha256": "32548bfda7ee51626c84c74f530a59f56f3bcd42815cbed680d63ba88f52817d",
    "sha256" + debug_suffix: "9067dd363b09b652f963dd2bf07f4535aa715bbcd93c334c1f7bf4280df1e620",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "b10388e3e399c665a3ab5aa7a409d4289851c43750392cf5f8003e98799b94c6",
    "sha256" + debug_suffix: "2aa893da56be771da45c313ea8f99dbbb6d987e42bcb00383862ee73b93fdd62",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6818754a394d34c971f40f8d05e3e5214abd24bf449f20b1f5b9df13ba2548ee",
    "sha256" + debug_suffix: "1fe0ba16ed572a8a15df5d1c73ed52d89eb9cb16a49cc34588af7d3d460207a2",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e74b76e00dc07d991b76dc560eb9aa48853497df2f002cef8466ac4d19f4d463",
    "sha256" + debug_suffix: "20aa26faaf03ead3c1fea2b62f61ab6cbabd3f3dfbaa53c3f28e62b14f2c7d7f",
    "frameworks": [
      "Accelerate",
      "CoreGraphics",
      "CoreImage",
      "CoreVideo",
      "Foundation",
    ],
    "libraries": [
      "c++",
    ],
  ],
  "executorch_dump": [
    "sha256": "b624feed3a942ff8eef0270e13a3ab3f58480b9dd3b0b1254e5a5ec5cd007cec",
    "sha256" + debug_suffix: "9f7264aaf3d6efb15fe7fffd7d0ab02f1f7664423894d004754ca563241a213b",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "447cb3143f2d8b55c8dfa49666ff83d378fca4e1a7d0c5dc6e0fd418a1dd5e12",
    "sha256" + debug_suffix: "aed9dd7dbe98e293ed4ed2bee5c429dc02644315b452ddc428caf9d45d2b3888",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c65c9914649dd647b3f14093289d875e3499f3505bc6147446d7fd903de3c838",
    "sha256" + debug_suffix: "40000cda770a6d8800b037e92d4557a592d93be668b29d678e1715bbf269a144",
  ],
  "kernels_optimized": [
    "sha256": "d197cbec2f59b01ad06a9fae988ee67d0c8046b28d03ece8f612b8413d6f1755",
    "sha256" + debug_suffix: "63b585445a86e38e4a2f1718a89ec835c931fae835dc86e2c6bd976c26e51c9e",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "1628596eaca32a8277cf5732e795a6be8382a6c23b6a104e3802fdb78e00322b",
    "sha256" + debug_suffix: "e4e81a20a52022507bed95e2d01b970ba4330fd4d037a1a1ed3b8ce2b0b5e2f5",
  ],
  "kernels_torchao": [
    "sha256": "7346ce8c0a0b1d679f3bc2f9c07e5d3baccc1631c5b53feace60e37b7152c31c",
    "sha256" + debug_suffix: "0554d97bb5acb789ce846bee31501337caf793890bddfdf32b151458ecef4d5b",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "479874470be376014c8c6c86c757d2feab2f66687208d2e30525770044da714c",
    "sha256" + debug_suffix: "0d4898fc947724820b846f5bb538e573d3eac34e6d6e0500be024e14fb24f230",
  ],
  "threadpool": [
    "sha256": "38c3fe4a8079133b38cc49fbd81a812d49deb97af33c081f1b1fb010ee830c91",
    "sha256" + debug_suffix: "f5130699e3090d965b858bd109fe918b804ce40c21e2a812438cd959c477e345",
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
