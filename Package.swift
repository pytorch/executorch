// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261008"
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
    "sha256": "c4327be1b1cbf8b3abb0698ec4f3add7b20488bb653945ee2452aa4aae112563",
    "sha256" + debug_suffix: "5749931bf8c4c33726ec6f5c274d6011dd46639f7ab11f7709307934fdacfdea",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "dc14eaf74647b10da7662ead67b5c5a0c05e4af5317b2c3227689825ee032923",
    "sha256" + debug_suffix: "dbf71cd9badd8799ef1a0f414362b9778e23e07d50af9b034838373ac3e1177a",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "2b18154bd72697b281d1f6122d0bbcd6289805834df6cfbde8a868fdd3d7bf32",
    "sha256" + debug_suffix: "85e2f5b90e058e92db594339ac947073d99b400714480f320aa55381d7ed05ec",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d4c48944c86a2d032e0e54ab596be75364f27a631366496cdbd83a3da9ab1b12",
    "sha256" + debug_suffix: "61df6c5d652a23bc294545e1c20499998a9480141b4e5f1c28426801c43c9fef",
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
    "sha256": "4d7a3fbf7f26968b0ffe942256d854a397021c9ceac716af9e88980a6bfe94b9",
    "sha256" + debug_suffix: "d2f04e06eae2dfd873ad7c062a599139f37809b484128737d4a4edb8884ac9d8",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "5a1f5e698bc1a57a5ed7f47f3f5f7cdb2a54ba43c751d624f7db51014ef9c41b",
    "sha256" + debug_suffix: "340fff795cef206b2648f2cb4f4d7829c58f15de205d66ee0d6a5048a7d9d227",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "b96be1646b7d75222abd7bf71cca345fc532607272d991880b65bfa2ef0d5840",
    "sha256" + debug_suffix: "42df5d0b2cb695879e528fe1092153795bfa93b391c7e285d081d04e7bf9467f",
  ],
  "kernels_optimized": [
    "sha256": "7c3e55b8eb81f1038b24ae084d8a849e6500046e42be1c95a4577770a77bc871",
    "sha256" + debug_suffix: "0a0bf9835a77578bd106c844405d71c65c34e3f99406795609b26664b44bd956",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "e19c8949105bfcf6f4f159e3583a69889057eb29f67ed9da8ec09662c670805d",
    "sha256" + debug_suffix: "3825814ce0b459d4e4fc29b34d83e7ea6b7520e8cb90ac8008c7fea829224ebc",
  ],
  "kernels_torchao": [
    "sha256": "9535ac149d832d7a32ad1e9c4bccc430a9fe16d0d219213a7d97fcdd7d072461",
    "sha256" + debug_suffix: "66cbbb290b2948320000a532d9c95190e92084f3e60e10fdd4ac989d3f7f8c0b",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "d1c89e7317a94254f8cd4200d03d88046dce9f88b7b83d40ec6aa6328dcf3bd1",
    "sha256" + debug_suffix: "92bad6b3dfd1da86c053e1c2368167afacc0366b25e257430e7bac027df2ed33",
  ],
  "threadpool": [
    "sha256": "2f45919656277ac177127766216e27199130cba05132bdb0551b14e5e64eac85",
    "sha256" + debug_suffix: "0bbf39d468cc2f98aa465b382d1a7c7baddd3e8a656f29383b1163217bd2a4d9",
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
