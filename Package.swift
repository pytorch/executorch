// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261007"
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
    "sha256": "966371e8e3120f807053adc2f76555cf9f3437754c9d2a08c96144aa7d09bd5e",
    "sha256" + debug_suffix: "6d10359361a713c8d914753eb201eadb48a33cec8113203db685b73f328d1247",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "4742cd74bdcec54061eef1bb35e2153baf7539af24206f5a75b873a922218b89",
    "sha256" + debug_suffix: "cc0feac22b24686d3b2cc6f4dce05ad6447af1925fb859b09fcf60cc658c5d68",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "736dea53a01a4e867abf369d7fece6d67d17629b6930e1925455a99694ad8269",
    "sha256" + debug_suffix: "d364e275a57e283ee22eda2e74c26a8c43fedfdb00a24c46ebe7353eb4d21abc",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "679bf5f3af2b68f13228f38db762e0d557604ad5bbb93f688728e650ed37a518",
    "sha256" + debug_suffix: "da05a74b19a49b439a54d1cc141dae2b2b0c770e8b2bc4555e411925cee20aab",
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
    "sha256": "5a8a2b603b8928f4db0ba0234b855cb73b5d723dfc54e7516723b09abe1ba5ed",
    "sha256" + debug_suffix: "d6c095ffa6b96f539605c830ad9ff1daa951b973d4bfcfd1feadfbed4a9de035",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "376f8825a2b42dfb98342c7e24b36dd8890f542427df9381b0757ae523ef67a5",
    "sha256" + debug_suffix: "5f425708c806b54c7eb4d38fe6a486a944d25e2f235c9abbb9f6bc5853a647e1",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "453a10dec00e3b73b8de7faca26b7c0cb3eb10e5f75f7d3e67ce9129020cb5f2",
    "sha256" + debug_suffix: "f4fc7086113eefa88f904aeeaf177e1dacaf96baeaa76fb7d52413c547f8d5a6",
  ],
  "kernels_optimized": [
    "sha256": "75c344c349b48ead66958718d6739f654e551f883c119b87c5f7523628a83eb2",
    "sha256" + debug_suffix: "af8df3a493d4ed50ba7c84c0480df40521d4671c5f962a8bc0bfb0de6b4a20f9",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "77f753664540f3696a8296150459abe8006e6005b1c3d9bdb58be12550a24cd1",
    "sha256" + debug_suffix: "c94629b958ca29b85943a1f460ff3e2c486d9a11f435f4bba1bec5dab27a78f4",
  ],
  "kernels_torchao": [
    "sha256": "b80060d9ebf4560bb7d204b8bc4adabbce42147d40fc9afc2c1e2444ac2972c4",
    "sha256" + debug_suffix: "ffd84b760b726738182eba6508a070f26d41c431aab218f2db42f345e74b32c4",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "d77d3af4b88e450de8f3827b76f59f250951a58f072d0efd9dc051e927d4bd79",
    "sha256" + debug_suffix: "cdbd3296f5e4f0c0840bdeb02ecf8f194f38ab7d6a21e5c54b6e4c8cabcea0a7",
  ],
  "threadpool": [
    "sha256": "9b5f2845d8b28785dba8fc0a7941958106dbb44eb43321b45c9f718fb128ccd2",
    "sha256" + debug_suffix: "6e72979484d76c73486695bd7e9a69715edbbc620d5892d10fd29a26ca307efc",
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
