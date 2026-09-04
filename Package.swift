// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260904"
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
    "sha256": "3058268f8e4bd410a3734cd342d3533c8b47ef8fc387ab5b6d09edaf9a6c35d3",
    "sha256" + debug_suffix: "b6cc547579d4ace3f7bbaaba96c436eb2a335a7aee9d782f6219a61a77967717",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "4b1fccf8ab4236aecc5043828b9b2497166e5694ac52e4667ff5a79d4f748dbc",
    "sha256" + debug_suffix: "f3894f1aecd8133c05a03795e5058cc814a55f03abaae9ef727228a27951c95b",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f570e1d2a685ff5c52606acc5fedca4f90d6a41f6f8dd9a08747f8c369c4e8a0",
    "sha256" + debug_suffix: "4e843b5911baf058c14ecc72927b3247dbca2c3155ecf7fc329a0b652a046265",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d13f43c8ead2052127e699ada7e82b81b49b48eeb0085d88aa316e367095f439",
    "sha256" + debug_suffix: "08e4bd05ed9e93101d4ec6b12cce6ce0cc89e4c6e8cc71795896753a4aafc484",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_dump": [
    "sha256": "b485ca1f885b5b0396ea59635a145236d28a100ff1b281e0aded79e1a06babaf",
    "sha256" + debug_suffix: "f439b2550aeafd73177150f95f092ef13a9452a871df0ebf69fae1c37e697852",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "996e5fece5a542dfc413b77a832ee4a9810a73bfce904e6ed180d83b3da30d46",
    "sha256" + debug_suffix: "05f2e5f9aac123627005d93443d5fcef7e7026b61b3cd5c8246d928797b51dd9",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "f6d5fa6d50694ee78b20b9483771615c55a34dc7ff6c44e1be1845b16c82b9d3",
    "sha256" + debug_suffix: "5b77c6d6c1e1523d8bbb92ae8c2d666633eec570d6322d2535708d25dcecb906",
  ],
  "kernels_optimized": [
    "sha256": "812dacf51c3f22816130b5a461f30281cc67373940510038218143fbf3b9639a",
    "sha256" + debug_suffix: "2b19949b7be0d15a93d56d54d6d48cbb39263445ce2e40614e5eaef0c1352a32",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "2b713375c965f8fe3e93d9f6d6ccd2554db236ed52ce92060ef582b2ecb77310",
    "sha256" + debug_suffix: "6327cc8730cefd4d340315911a04c794dac2a52b712051d6280b3abe145adbd5",
  ],
  "kernels_torchao": [
    "sha256": "ec553e93904ed5471e9b321519cc7224c2c7e3bac474ef26afaa6cc7447c9d12",
    "sha256" + debug_suffix: "fbc3501fedcab67351e5977717d86488e085ef34d8577cd65c3c9aba34512066",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "dd76b4467f4e826e41b2de666bf8062a87d33a43b751fb075914c15083c1ba77",
    "sha256" + debug_suffix: "6aebaae4b1f745dce5b0651ac60c792eb5a7f3b8f706820f56056d57c5ab0062",
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
