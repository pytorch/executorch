// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20260916"
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
    "sha256": "d6f773f5a276c078e0816351a0f2eb194bea293ff448c83b2112cea4394c7d5f",
    "sha256" + debug_suffix: "badf7974d737f3935e2896f73fa34dc509d2c3b144d66bf8fcff7480811a0846",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "dd5c6e9d8ea6885cc53ce8ab989a1153a749f9f2a10159e9d671857ae8c8fd46",
    "sha256" + debug_suffix: "99740dd36aefe448a96584d5d910dcc73087fee478799db0c4088dcf6603eddd",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1af568ff705e7f5a78c41b7de9d14b6fba4f36c884317fdce342fc2493d6cbb8",
    "sha256" + debug_suffix: "721f384c5cf3ebe21271534d144f0203925a057591c5264b5474d06720747186",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "176ff61433f7bf67d972d12588cd423c79ac6be189f365284db5c614862ac54c",
    "sha256" + debug_suffix: "bc73843cf05d5556e2a65bad29c758e46083e7f5973ec02501ca13a1a9a777dc",
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
    "sha256": "b010f87d9bc58fd1be61baf12df0aacbdedd60b07b16b19dde6fa71d2caa08dd",
    "sha256" + debug_suffix: "8d98e7c9ec6d5b5fe7e866a932547da84cb5dcf12150a525c2ab75d5171b00d3",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "5b03d147cd0c17ffe070846688a6df30fd5dd6269408f559e94c631bc15cd1b9",
    "sha256" + debug_suffix: "b7010305c66b2433a49813eb1630dcc143c1eaebdb7256e40907d5887e529142",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "f8baa845b45e527f2353cc11edb8884f6df5d1726dfc4869429bfb4e98f68fc3",
    "sha256" + debug_suffix: "828416dc3341ef987c57128d7d346e5b06c051c9fbb042ba0ac63982ab84cdce",
  ],
  "kernels_optimized": [
    "sha256": "3ae9042f2aba842ab4be4c0555b82c4c2dea6f4bfa6852adae3fb5357e9fcc24",
    "sha256" + debug_suffix: "8c48ee08cbbbd08ba6c013f6c75aa13754ebdc353f76afdd91982d5481208da9",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "490ce787abfec92c479da774db17c6d1039b5e63d629e1263c142becb981d5de",
    "sha256" + debug_suffix: "d54328fcb8fcf4b97ef68960cb66a37b4cb0147b103cd0bb43e838d99ca056a6",
  ],
  "kernels_torchao": [
    "sha256": "fc0adfa989422f84f3c1ebf2dad7c3c0fcb12ede2fa8bade9362db3e23166211",
    "sha256" + debug_suffix: "9b7a54f92761cebf060efb9c282e204cd19900c420e3a7c18f7fd5bef701b204",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "5d8c0b8eb65941740f05f62fa1da02724149dec163d880b982a581da8b596ac6",
    "sha256" + debug_suffix: "188ac783b681611d7e17b02e50f111ac787207b681296a1e6919441404df48d7",
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
