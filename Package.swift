// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250918"
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
    "sha256": "5d78add8dbb17218b1aabec07c2bab39bbb99b0d434783d5ee7153094f94ad44",
    "sha256" + debug_suffix: "0649f884fa19ed1fc5b09113c8be8f448a01e5f4f9487b26e227a7d3183d7604",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "5465d48c610301f494b0139be8604bf1ef7ed75a24c1ab7b10243e877b6dcd0d",
    "sha256" + debug_suffix: "b5727df68074252e0ad07700c08804e67fd3f234c35fbaf2a9210f42516e3a17",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "5dbd5e5dc3deb894b95de76a055615737e0b014ec0156c8fb5e2996caf73ce05",
    "sha256" + debug_suffix: "8f5fecadc25d5da48f9b329e0c7a44aa37c1cd01d5f1c568980add97f7125ec3",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "ecb5f56e3915937597432a23e64617c8239e5529cc310cd92f94320d87ae3ca6",
    "sha256" + debug_suffix: "e113d15354d766424e7f1d58533f57ab07e475a577b8aaa5a9c98de994e37023",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "86e355afc562de806ac6e51105f476d1ed79ce6793496428d825976e8344dd4e",
    "sha256" + debug_suffix: "5979dc37ea6f33ed3df84de8273ff198fde5d42d4ede409ba014dca5a91e83ab",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "bb34ef73878673aa9c1e035cc0a5613a3f5655324299c7ed0d0c01f265599344",
    "sha256" + debug_suffix: "80088864dd61e5317c108e7b6df3bcc9ad8b88e80ee21184cf637b1eeb34caa3",
  ],
  "kernels_optimized": [
    "sha256": "10c2619a666fd879fafa7d74f183d285aef7abebd40e00435a9e5d61f1e37db3",
    "sha256" + debug_suffix: "72efa00b208770ac4b606024e782f9c3ab0dd9ad22fd80c3906e6b7898def3f0",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "5cb61fe48fd1d81e82127579b06cde2bee9cbfdf1cee9084c801b3ddd4076fc1",
    "sha256" + debug_suffix: "d117f99028f0db13a6d8d807741edc377e834af0eb392c1b5242cd7f395aaeed",
  ],
  "kernels_torchao": [
    "sha256": "be9a8eb26aa4365ec7cce6e580d8ad0ae1dd06f98727d32d14645aa4137de138",
    "sha256" + debug_suffix: "21a27e14d115cf53f6d746de6780188e16ce23ab6ac77f89d9bf371c89f18253",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "6da6488a961032c89070d80975e1d4b255a54ac62804c6ead8aa88bad5ca3db8",
    "sha256" + debug_suffix: "80e27be13fa293168f761b4b904c89afe6d1958a25afd6a08a428006b64c6887",
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
