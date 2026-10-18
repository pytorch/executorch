// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261018"
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
    "sha256": "0d30349bad93bf6d93293864abc5cc38238bc5999da25c9e6e33610a5407b7f4",
    "sha256" + debug_suffix: "4140e4641ee4591ad2d1e97f7a516d6f56f37e03d3d37df60d0c3adf8e749bee",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "13618bf0cc30c41584ee70a611124230b23d8a0f31e3addda9f3b530dec3f913",
    "sha256" + debug_suffix: "4023cfd3a8d31d682b218781b9b43a3b5368b9f87667cc3bf8b4b9f337e9aad4",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "49a17c6e6f3a3936bfde1a2c1cee0f0dba1b309cfba20635c3eef79eb90af080",
    "sha256" + debug_suffix: "1401413aeb79d97282c6c895e926ccfdec37f833238ba51ab95531aca907a831",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e404baedf73ef5daa63b571070ab2680d78ea332e821ee358ba01748ff401883",
    "sha256" + debug_suffix: "0722f72617600ebe66b6482170f12e8b55ac765877c0a8883b94aeafffef3fcb",
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
    "sha256": "de7590b8a1c544a157d63347cb64c8251b6a7c26bd47e9f4012912557fd7a4a1",
    "sha256" + debug_suffix: "329a99f3abdca86cee6d6c9906c3606f314777b42733db4cd88f09bfefe18348",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "291e5d704bbf9b11563868ce0baf928cc7b3c5684bd6d52d42d2b8ed70873e67",
    "sha256" + debug_suffix: "5712a63ba78a5708e5d79c3d890621ebd1b333ff9148b1671edcb95c5a94342c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "f1af086840f2329fb43be69716623aa398870944330b667ab67763f363f652b7",
    "sha256" + debug_suffix: "adf530edcd57d113fec6afac8ef263fab491b953fc834e16e6944e798beef9ff",
  ],
  "kernels_optimized": [
    "sha256": "b2b95f4345e827326dc5051bc01c95ae3c25d434a2218a717f6e81eb855cc7a9",
    "sha256" + debug_suffix: "4919a1fceffb90f3a1a9012f532d568484c9de1f69424ebb1b9ca45cc2b13d5e",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "1ddfdb22dec0d1ae709c917a5ce2e48d2ffe548c426a38e48bedcde3d287fcc8",
    "sha256" + debug_suffix: "7dc77cf5d0725e8344dad1cad985fa8cdb16bb62a394c86caaa0ce2b65d4e5d4",
  ],
  "kernels_torchao": [
    "sha256": "a054963c5a3d8768255af62d7877b423da198697399b6e9cde1b89235bb65f14",
    "sha256" + debug_suffix: "fba608334a835f2fc0db544327bba6984766d46338b5e4193563ebcc4e5349ee",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "26cac80d7c6d96599064d49987c271aec6adcfb9a0ebe55d675e1d63a731dda7",
    "sha256" + debug_suffix: "eaf7b8b5b75902b84a42e1cd9994ac7652e16aeee20e0e91fe3d05196044bc14",
  ],
  "threadpool": [
    "sha256": "56a13058d741567ab5a4cfaf1bb843f2c10e368a3281f21e47dfc4393a37c659",
    "sha256" + debug_suffix: "4201a869538b3ebde624b7bb8ad5b6b98b0a56607a902912f632adeca060528f",
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
