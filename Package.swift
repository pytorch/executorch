// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260909"
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
    "sha256": "4abfa545b9058123ef0fcdb2820226c8b79336c253a90a59231d76796d0a23dd",
    "sha256" + debug_suffix: "232950a7bcc0e4c43578e731f7900de8a425bbc4f66ae3dc197084284a34305d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "61513afdfa1088661b3ef4ab26be45ebc4e41ad56e42d4396590428fb91bfe27",
    "sha256" + debug_suffix: "d33244d015171709f78551ba97e0bc2093ef1414a12b2dd67fdb46b43e2a80d3",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "fd4f4cdbae3d0d5b3f70d3ebe748e665c9ff1362df4465ea9769455949729ec4",
    "sha256" + debug_suffix: "38294c9a5d307d10424e8086c9d17cd08b00482a7c180daccea219cffa74c450",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "ebea11052584b62843bd86bfbf5815d95ae969023af79836cd577fa200ae733e",
    "sha256" + debug_suffix: "7885b821fea8dbcdf2c4a55d6531f0a652024ea18cbc7033bff0949268c33092",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_dump": [
    "sha256": "12aa2a0c2964d5f27be60ea49d0ab730a2a9871108dc1fb8b6ac1c485f1747b8",
    "sha256" + debug_suffix: "5bb3ff94202b1f75d770265d6ca91fc85837c7b573547b2714b611646bd595f3",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "51c6b37ceb142465b4a6be0f39a315d64abd7c0219ac34f0328172fb4fb01949",
    "sha256" + debug_suffix: "7be651d7b2de618e06517f08481cdd4b147f931cd4ae38b0ccd004de0a18dc88",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "0de3904ee93d8172a2a4b7c69ded00796cd02d3d01f57ef12822e74e518c9dac",
    "sha256" + debug_suffix: "d245075f1b875131f072662df6b9698a36f9b2ddd4af182f9890f12703f566aa",
  ],
  "kernels_optimized": [
    "sha256": "49b290b8f0b978e6f0ce5e57a0ec50a2cbd4c3cc41efcd2e6238a0b3ad268c57",
    "sha256" + debug_suffix: "4cb00329d59112ca6044bf84a7321a5d2a889dbe0d5f289f5b91e8b8b5aed5f9",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "a18e8624857c88210c67af133359d1cb6fa1b97edbe6f398b07217a817620876",
    "sha256" + debug_suffix: "35ce38f0362259acada4c14ac5eb5a1f94e080b19327bdc6aea1454c12e00a7f",
  ],
  "kernels_torchao": [
    "sha256": "9e0c403fbb778d32158925aeae43174b7a7ff21bc57d40dbf0e27c0df6cfd1bf",
    "sha256" + debug_suffix: "2356af3735ef9da57a6a4e94df04ce3d08e41c5994a202873e873c3e71927960",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "2ddf066859095c73b63dd6cf7f36210edc7cb92aab1fbd3f84df9d651c0dd92d",
    "sha256" + debug_suffix: "1208699a3736a84c30dff83e50d0812ae45ff3a1fff3145d280611509db93f47",
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
