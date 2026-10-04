// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261004"
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
    "sha256": "083f8e0d2cc501880a3fbbdc282143a0bb46dbd50f13017c75b1347b904f519a",
    "sha256" + debug_suffix: "7bf03f3213a8c58e3ebcbc368dfed72707a03682eb00c6db2527edc2226741ec",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "d157b9577d7eb4f2c1e67539c941f9042e495644de4b6fbeda37854a4b251564",
    "sha256" + debug_suffix: "40de10b58412623c2da641d0eb02c845108f1a9bd0ea0dd800e085d820776120",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c8f4f0117bf3185dd673c60ec421aece239fb510b9df34ba730803d8ed022fad",
    "sha256" + debug_suffix: "fde76431aad74e899313aa50f556060a6f67c8d8a03bdb0ed6bffe0a51b43c5b",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f03a4377b73ac24e8aad7dcc2465bd14b62e2c93ee2373ebc67057a3e2f9f29e",
    "sha256" + debug_suffix: "ca0e6d6c6152d973952f2f0a7f0981867b895f6bb5ddcb48cbf5d0995be58512",
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
    "sha256": "b4024d995011b826c91997d170fe8427f9a2b37bbe5014bde9c7c2ebde5f5f07",
    "sha256" + debug_suffix: "ca1ad54be287793511963d61cca57bf72cf8719d12519c543f2503f24898ae9e",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "d4f869b8d928eb497da2eec5972c5bb14bcf2a4cbb26adb1ee84294d071444b7",
    "sha256" + debug_suffix: "253524761ed7de3906e0786baa46a6eea93829663c08c541a98532e0de8403f2",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "effa218b00c720cea901db019ad79c9a4446890e575c5141fbf1e3468d2aacf9",
    "sha256" + debug_suffix: "991d3a6f0be5252f6c2eeff06b6da388e7817c5214aabdfe4a04dce5fc32698a",
  ],
  "kernels_optimized": [
    "sha256": "839dbe66ad5d71cc4d8e86e0164f5727de490799380e267dbb29d3c5557b724d",
    "sha256" + debug_suffix: "d277dedd3e21415d84a7acc2e3bd15a5c9f156ff203fb5379cad0767e869bf4e",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "274a5eb521cfba824eca66fff2a847624c1f9106d4dcc151892678465cd7d924",
    "sha256" + debug_suffix: "7c5976394772e83f21a2ae37b43e94cc5fe94733c67d921201d444227c1bb7af",
  ],
  "kernels_torchao": [
    "sha256": "fd0acb4e49c49a8f72de9c01b4d449f291af7da18cdb53fbca97ffdf13bf926e",
    "sha256" + debug_suffix: "9b0291613b55462a0adcdd6ba734e8efb07fc940d2598a20dbf3719bc9c81ee4",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "9c9439e60ce15d55a2c0da837bc36f82517572c4194ddebba47a7a2dec113dbb",
    "sha256" + debug_suffix: "ebfb11af0f07c3e891f1a0532e034484fbfe4c7ca3e76cd20b11606cd10bde28",
  ],
  "threadpool": [
    "sha256": "21816962945bfe55c59a72b698f31a953b4ab2275f497bae2d8c650a24a36f4b",
    "sha256" + debug_suffix: "bc676326bee1658e5571ab8e0b632ce79255d6f2e9de439c5328afd9d5e67f8b",
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
