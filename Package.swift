// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261117"
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
    "sha256": "3d46477e3abb8fcd68f924c74bb78014774b7f0866564bba7782e48ed3503f2a",
    "sha256" + debug_suffix: "e57a1356b2c61c74b850861fbd1583df9305f9affc9a352723c2465dc2518257",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "9a8140e7f8fd31ea143df264b6f95963ffad03fd9b39263ea3fa5795c9f93d8c",
    "sha256" + debug_suffix: "f55c02e2957632d005fb4263e4857f034dec4bad25b41f0c0f904bd00be62d9a",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "abfaf82b4556c1677f0f3b5b9ad704673810a8d19c9b5bc3c4eb6f33d7159e9a",
    "sha256" + debug_suffix: "3fb49039a04a4742395f9ce8f79bb8165ab16a7be04b81e86c5b95d7c86e52d6",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "0f88c9f868c298604cc272e0cab8a0101f57fa3f7f4b3dd5f1c9b6c2f476ac5d",
    "sha256" + debug_suffix: "bbdfb5f37c296303c82029e7657d168a879927009ae2e3efd68796e1cf5102bc",
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
    "sha256": "1d496244548799ad624d0ece0a4f7ebc05a064cb0a9c6d89b43ea03f0f159c21",
    "sha256" + debug_suffix: "3c7b62b5b9dda9915e167fb9679d0e1bd4773a904fb72ec58048c5bd106e5c6d",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "ba4df821110b0fb684902471e7094e09a11901a896a9472b2a109dbd28c712da",
    "sha256" + debug_suffix: "9a027cc3f29ab765aaa3b09b7d3a4cd2d02b83f655f03d6ee47e477960b318f0",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "2a1c0fde7824520f2994b072ac605fb48120024470aebc6b8a76aa8f9e36774e",
    "sha256" + debug_suffix: "1881d7a99d4bcbeb20787fb320311dc14ae1b7c6c9ed8cb924cb924e757e7d3b",
  ],
  "kernels_optimized": [
    "sha256": "766e79907ecea663adfdc8789c13a54abeba35980ccb9ea53b7995c36e3d1069",
    "sha256" + debug_suffix: "c550986035869077be92d15c251c12d425e2ad6e2949f7ad2342d310374cbf02",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "cc27adf93132b99755111dc66d52b52aa4610f67f15ccbb3c4ee5dcea08fd278",
    "sha256" + debug_suffix: "b83da640864e5db4dbfa8fdbd18eaaf2cda0f30d2879b8921824f149ae4e2742",
  ],
  "kernels_torchao": [
    "sha256": "46a2d5b834d13c22e98de3a1e6aadbe60d0c788fa07884181f2d20effe8c7126",
    "sha256" + debug_suffix: "98162dd75c802b3abf5893171708e0403c9885197aa9dc4693c48dd2d2ced8f5",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "9603c30bc4f83c79daae90a3a625d366a06744b600a5845da21ac88aeade42cf",
    "sha256" + debug_suffix: "ba8f1a77339f6d14cbfa8febc0c574c6d231f61576a5364a0140c196f51b0957",
  ],
  "threadpool": [
    "sha256": "2ae389bca50224492d9f0b1474c433620123c8720a130b7d45ea5a537bc1086e",
    "sha256" + debug_suffix: "9eaa3b8b4ac45522cee4e19e1b4b6975457e0b254881a6aefbbd4250ef4b1399",
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
