// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260830"
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
    "sha256": "8e75a0389035275cc49b23cff7e27b8d8f9ceb2ce4a261c7ad34a2bd9ef2589a",
    "sha256" + debug_suffix: "8ffd32f4552b7dc710650f923f13f876ad09092a7c32cc25073d784509d7148a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "4987a7555229d0dbd70bbc4351c78344e664527fc40bf11accb41fcaf55f773c",
    "sha256" + debug_suffix: "013e064f86bcd3befd7bcaf9809af2741b398d62b280e292ca951e7e665ee486",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "87d6ee02ea484df6df729b47e3c413bc606adaeba0968cf8cb76384e7a9aa014",
    "sha256" + debug_suffix: "a9ea7010b4b368f6bec3f0d9fe5e6e3371f8aedd14a4a06d25cabd60a5696976",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f3549d8946a62b4be108f7eb0285beaf811b5322c989b8f72be5ba301dc2bd14",
    "sha256" + debug_suffix: "84d96c4ecb48db65edc0e9fd4bab26d6c44b3c3d9d025ed8f0209bd9d8cb4ae5",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_dump": [
    "sha256": "ecca5e7b8782b79e86570ccba12bcb4f21de30c20279bc99722d0c39864b54da",
    "sha256" + debug_suffix: "c40db6cab5da00f3b3a11d19b2301d5fc5b3d693671bb9fe0526a15fba0d4ac9",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "ac49e52d4c8b06ac9db7b18939c62e21b5b74e94c1945cb0daa3dd38ae259051",
    "sha256" + debug_suffix: "f87972ebf77add28d46dbac224776684472aa8b12b5d2df53797e32065f3d516",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "ba4c07b9b8396bb5474733b2ff54e340e22fd24c3a5985e6428e0ff103f458cd",
    "sha256" + debug_suffix: "b6f2b19ab387f1b582735094a97676bb570614dacf194edc83e87b65a226c8b7",
  ],
  "kernels_optimized": [
    "sha256": "523720475105c18bca48f7862f3023df319289d0b97e12a76f6da2c74bc1fb57",
    "sha256" + debug_suffix: "4dcf2f1e3e3a621c5826c09af853186ffbf7bb33af7828abbeb9dd35c596dc88",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "43a131a7b9302d45c61c666f719e1d2c9573936512167fe717d76ea3d38cdaee",
    "sha256" + debug_suffix: "befce6517e5c9303d3c931da9c35ebae6eb0cda89764b91d65ec84ff319a305c",
  ],
  "kernels_torchao": [
    "sha256": "510caac56f8d0f57b8feb57e69a4080e2c210e4125442908f857faeaf7cb6c74",
    "sha256" + debug_suffix: "cace78554cf5a1777a1acedcdb8f138d683539700e5a10b9e1ce1a27a30e6578",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "48db88abeab1566c71cdcf268f546ef85b608f01e8175387b05022574807261a",
    "sha256" + debug_suffix: "6224afa35d6a2c77b22988abde660688eca810083b9d54dcd4f520ab9e34a9a5",
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
