// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261121"
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
    "sha256": "653d1ac6afe7806cb93c53a55ede698a4b5adada910ce636e6307d11f836862b",
    "sha256" + debug_suffix: "7fb9a94287b2675e2edb594edb9bb1612bdc3ccbd9b5b675656046135650e79a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "4a3291210f1d67a2f5665b327cf4b749ec50715a213d7939eb59d9d134d03a16",
    "sha256" + debug_suffix: "75c1e4ab0bc49a3d7018606746cfb47c9ec6f3396c8d5e85f3a83a5dace90655",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "430d8655e782f37ee56d8583f7cbc53727650aaa70618392645ae6c26c8c59ee",
    "sha256" + debug_suffix: "a89c775d2a1dfadef1626af350c5c7d2f296b69e551dda2689bdc5ddbc6a6c02",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "ca557eac833065b7755904c041f5d066e8f40a0f7b2c2f3287853c245a9339f9",
    "sha256" + debug_suffix: "a2d45e6da1ac372c9e8879cae4d579e2a19333c8f0029160066f65e4732cc717",
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
    "sha256": "7934cc6415f9578576a2eae4de020741b9b059d77c10f4429ff65defe493f8ae",
    "sha256" + debug_suffix: "7dca98e51a94b54668d9b9c7da2e230287d99d5e37d4124404810fa930b03f04",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "c1c874a4d2cbd61f45aa6e689b441f277a76b2602c7d6a7b0de80cc92fdb010d",
    "sha256" + debug_suffix: "f9f78c494e3c88a221e74e62aa54e870ffed0d063e5d43eb6b98c7be36c498e2",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c7e9c28e5055c31af86151e8f263a551ae0e2ae10318f88fafcd99edbb98bc16",
    "sha256" + debug_suffix: "1770ea310b94055c45f4144a1c9b3f3451116142be5149a23cd10512d2881768",
  ],
  "kernels_optimized": [
    "sha256": "762308d0b648eb9e58e21c1df64e41e0c13eed9e79b6bccf6503b41d01405ece",
    "sha256" + debug_suffix: "6e9602a2d6b2a76cfb2c51c0a40e5925c14fe28e29717a2844960069a5fa5def",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "4b617cd8ae457fa45e0e3eb27b150ae119afaf60c8b385b163572cc99ddb5592",
    "sha256" + debug_suffix: "a11126217e337f30ae4e08354e19015cd4275b0c2040fc41eea0694c49974206",
  ],
  "kernels_torchao": [
    "sha256": "2ffbc68229ccc0f6f279144573b836a729b823a2e78d56623f02c72c2be0feb1",
    "sha256" + debug_suffix: "96207c952f736e7a1ae610921c047ca915f69bf71fa35fce1e260d1162241423",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "0e39c9929a5ca96b6348cf2cbdd5a5da8c5ed95d1083b60f0e7b01b5a606dd10",
    "sha256" + debug_suffix: "b66ea472064eddd3fcedb176a62c52426aabf64ad07da4909dcb4e61e694a5e7",
  ],
  "threadpool": [
    "sha256": "693a02c45496aa46d032f1988a86138e537853223fda9996d9780cb67e9f9a1c",
    "sha256" + debug_suffix: "8dc8e16f70e44711f40817b858ab0b0ae139dc73a9dfa64cf9dd7680a07d614a",
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
