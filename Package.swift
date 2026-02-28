// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260228"
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
    "sha256": "4c9cd36eb87413a23df34289d187915f9bc6a603685f9768dab1084046e121d2",
    "sha256" + debug_suffix: "01733c7485b9abef1e9d73d5270ad6ce7e93d06ad599ec937bdd5b85d1b8f2f4",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "250d7caaaafeb4ed49c3c5935dc32e4ab1ea99f5a0f2bb5980b1ecd18a4e18f6",
    "sha256" + debug_suffix: "042f654ab131c5be5a1cc24ee1a91bb0b1360f3f8ec3272906d03f13bc6ddba3",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "dd5a2ccfd17be994a49c7732697f6bf1cffac613ef46ebdebec4d6163f5d71b0",
    "sha256" + debug_suffix: "8aaad0522b04371d71cf4325e9284b4faffe15e0d9850cb096491a3081d0abae",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "4710041046848e026f7a3a3e344aa45a943f8619bac3ff290fc77b20965386bb",
    "sha256" + debug_suffix: "f2a68afc73e7b3c554658dfee6c12aa1f1d024b8dc57648795eb32ed8e38fda4",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "cbffb36f61981d35f8432c1d4b8e4c5c854da93c30720983699dd337aa40be3a",
    "sha256" + debug_suffix: "31299f912651e9cd76e61e71a76f1b1e19b4bbf3ffa33bcc2c40425235296c15",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "361a8f72927a05006d46c27c30b33c2c3accc113337e232facb86c33fdd1c79a",
    "sha256" + debug_suffix: "55d5688ca9c7bc56d06ae8bd73af02a4c4fb0f39f3f4baca4560f24a4550486a",
  ],
  "kernels_optimized": [
    "sha256": "5abd0734ce02fcb42789c6dfdeb8ab27b617e77a30dddc32c6753dadd381e283",
    "sha256" + debug_suffix: "11c3a6bfd24fa8e2957ce4cecb14562ceaca80fd28e2294e6d3572340e4f7eca",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "2b677f10ac06b4035bf20e2565cf2ff4c69531ea028f05cfc4a846e65f3907ac",
    "sha256" + debug_suffix: "af646d18210af0eee3a231f3ca45ca532849724502fce5e2a4ed2769a1c973b6",
  ],
  "kernels_torchao": [
    "sha256": "8d3c79aadbf2c00df708332eff2c04510bd1bd4301642cd8a26386fa712a8e8d",
    "sha256" + debug_suffix: "738393a180eb2732c4cb9dad486cc661bc32b23e227a54c0c554a2d86782644a",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "770296a983a0919b2b1391c7f4a22c04dce156cc376f99b7388bce6b6bc32d2a",
    "sha256" + debug_suffix: "67be571f074928b5add3ff5316a8ab14bca362d282bbf55fe7b01d6fcfb0d567",
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
