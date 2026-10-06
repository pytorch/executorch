// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261006"
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
    "sha256": "772494b122b1f5f1571de41d5d9d961320097c8b913487d1b802cf8227ca1782",
    "sha256" + debug_suffix: "7be5c203e419888086cdec8ca5cf451018551827d7de0fe0854eefda5daf7a14",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "75270002114714964c5bade2fdb7da17c099b12b7fe4fe5cd84bd4b15b78e668",
    "sha256" + debug_suffix: "8ee81fbefc5d46bd03a8c7aa215aeaff91d4eeda43a330b4187bd87a9ae2c133",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a336021830d82b107993f7cf972280e056cc38d303a8bb268e2ce0fbe5687114",
    "sha256" + debug_suffix: "87c63cb784f740c7e5f0006b7cf40c5cffed87f328d9a70f34ab56de26d089f4",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "22394251fbae383291e1823349837fcff237c34d9ef83552e233f9593a65d38d",
    "sha256" + debug_suffix: "fe88978fc27046973622c87ad0eba15d205e3309df402507a4113113e21ea2cb",
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
    "sha256": "bae9fa501a9fbe302a8b37b84ec9f710cf6249263212905c0f46189555254fb2",
    "sha256" + debug_suffix: "885b59f1c8fdcb9553ab37dfd7c12ed3e5fbf8b0c70e52b1d90abc587e29d0c7",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "b1cea945aeecbf88f410289331657d9fd81faad9d5fe234ffa6230f3dbc63b32",
    "sha256" + debug_suffix: "ffa25b802099fd8ab487c9c8a8cc58523cc635af7c26c38694e5ca609bd85395",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "6f2c070ed59ba4915844b263693815705ff4ab12d3e2e3bd3a15f45a4524f7fa",
    "sha256" + debug_suffix: "8eaec63729349f431820058ddc241b1bcf37160e538a0c511d74d0236c81db28",
  ],
  "kernels_optimized": [
    "sha256": "d200cb76bc9eff76a912d2489d931ff9ac86241f5ef5423f4f1c792055d79b20",
    "sha256" + debug_suffix: "ec9e3749ccbc3a2522457b2b04a4831df36a85c64267aa3623e10fc37d767e71",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "e8a213a3c6a9e0f8164f56ea8ecdaf124028aa614865d962b47c9bf5cfac6be8",
    "sha256" + debug_suffix: "d8a3311f2d3e22cdb6183c457a443359e5156dea4f788dc23dcdf52d1374d92c",
  ],
  "kernels_torchao": [
    "sha256": "79f03e248cb806865208f5a6712a90528f46252d8f54d45a54459bf2adfe0b2f",
    "sha256" + debug_suffix: "c6d518fc4156d29325f0317c0b0750c5fd3866c088e9d8b76489e97591f91b45",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "7b56a1cbc77a334504455f340a56592d6f7b670a52fbcdbbc38f959e9699dbd3",
    "sha256" + debug_suffix: "6262d6a57c3741bd850b9d8b55ffc2b82aad0e143c8fc4e4cb002e021758876c",
  ],
  "threadpool": [
    "sha256": "2f783a12499b6ecbba4207d71a8e5b20d385098a4f52782ae64301e5c635de19",
    "sha256" + debug_suffix: "f6e70e667cc5fda7d10f31e6abb26d130c352c722f3a2852843b0cf6db347010",
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
