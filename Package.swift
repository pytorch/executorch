// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261123"
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
    "sha256": "7438cea673cb652b4bb767c33c5f7b14cea105bac3e741b3b17375c2c48bea38",
    "sha256" + debug_suffix: "4099684e4d7791272c2342108b1122e2e50c6883af0f910c48b7512f1c91e379",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "34e8be22c3e2b5512b3a572df7ea5968d80a6dc09bed091e8392de0f292a97ac",
    "sha256" + debug_suffix: "65ce7c51e953424eb9474a6abdc69ea2f76ea098ae16b947a8af7db82cbb0b87",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b219e877b3492018914ca2df8e5d9d89cb60424e41932c7d3561266d2e629f19",
    "sha256" + debug_suffix: "6ff9b6c071f8e728d8569a2be4b8a86ffb48ad2d1403bfa957ae509ceb90396d",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "8898accb4319f3f8d00cca5fc9a0ebc49479aaa64386602776436f8e375df4ae",
    "sha256" + debug_suffix: "c1da1202e7a1b3ec3bda33feb1edf05d0dc546d8481db531231907bca770d3e9",
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
    "sha256": "8ccb95973261415c05b85b7dae718f34900088f420027303ac15b1cdea0aa3cf",
    "sha256" + debug_suffix: "6239b8309c1a8c7e1c02c1288b7db8f78bf7eee5d409848c7fe238cee6381d2c",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "86580855125fdcaeac6b362b9ed3eae0ab846d3c98034a2405ffc5e4b640d621",
    "sha256" + debug_suffix: "469cb6b4933f0ed71644f6214bde9ae0be07eb3129599195b049257626d9a41c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "deb96bd72bb678824eaba21a93f8feefd32f660d19cfe688857a856bfda3c0f2",
    "sha256" + debug_suffix: "cddac99b6d2b4dd7750124d23348c1ccbf1230e34c442e19d9ecb4b8f1b4723a",
  ],
  "kernels_optimized": [
    "sha256": "37f26a64be627cf0d09de166b03831ab9c9a8a4f4d5399664609b53a8018a3c2",
    "sha256" + debug_suffix: "2ae976b1b5a2b4a41d4d62ee19079e4aea3ffe17a4255285157f36e444659ce9",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "e274bb466e6ead8560a2d2f9303bc1cf3a9f4474e406daca714c65f2f23c21e4",
    "sha256" + debug_suffix: "3f0f374ddccb0a4cd89f5136b0fe52c98d39ff686cbbc343ff0d80dedbb7794d",
  ],
  "kernels_torchao": [
    "sha256": "e10b3b2cc9a8093844d623af9f9e0f1ebc977963b3898ca29cd570df9800f2fe",
    "sha256" + debug_suffix: "52584062129c4b81462269c43a343964d4840dcac208be99e543e5106255a0ef",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "da5737692fe7988a171913a6caf880d5f8b6857acc325b721343439a69ef3799",
    "sha256" + debug_suffix: "32e5c142e6fed056906206c926e5ed34553263b8789c68fbf2e5a1fb45f876d9",
  ],
  "threadpool": [
    "sha256": "1514a2ea97a0c0887bbae715b3064a7b4b4f9b794ee0ea2c54b4f6a7516f548b",
    "sha256" + debug_suffix: "998ee0fa87f1e45d73989b01c3f5fbce7dcf7c00c57e395645a2dba04b64d1c7",
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
