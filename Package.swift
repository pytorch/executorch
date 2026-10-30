// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261030"
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
    "sha256": "97f9805d21a4ecf10b1501ce1077d0b55a3f70f08de7a62da892775f558e05a7",
    "sha256" + debug_suffix: "663b596a48eba31ffa052f2cf1affff105343fb3e653b001bbb64759b3869098",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "1f17ea86551480a32daca338f02e47017c9bd53d51d4f7098400bd0c4d7c4e1d",
    "sha256" + debug_suffix: "5c64da7adb87685f8800ff0c2476091b852a31289ad8fc24f7d58989d10b5ab9",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "bd8360c4aa2bd4ca1cc1fc5c93acfe62fe3ee98c27a3a55a07683692416944fe",
    "sha256" + debug_suffix: "4ea1fa44ae8bae86f0d5311373f13c7cabf5867dd3c4e0f9ea1f67b882f3798a",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "baf9e9fe173ec4e8b81256588c27d810128431701d1bcb9f41ee8ab83e75c82a",
    "sha256" + debug_suffix: "44cad386d07ac5e6f6375eeb14ccd4db2c9ed39c1a34f6db9aba44b98b5abc4b",
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
    "sha256": "871425749578079bb9598f923c6b75231845cf48f551a513205dcbb76fddc4e2",
    "sha256" + debug_suffix: "9ea15c8d8ddc8eeda31e3c6665b6f33fd558790270dc33b40da22b6975f2876e",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "d9d10e6c8bf344bd759c92cad0ae84a3c7da33174d2065e95d2fe182753efbe4",
    "sha256" + debug_suffix: "6bc0f36ef2ce7d17ebec83cfe1e6e61e2955cb9bf962d1614308088475ff553a",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c67e2038d1b7426c6011d72d2a1d044691025449f9de357d2b19336f9592aa04",
    "sha256" + debug_suffix: "f23d4e85c1fff16f374ede7156fe9a45da519156cadee3b6b41a8df83478e6a1",
  ],
  "kernels_optimized": [
    "sha256": "8781b80dc8e2a032e9e51a9514124738b3ef35a6072a360755a79e797fbaae67",
    "sha256" + debug_suffix: "12ae79d7c3df422a7a467e80c69e9a9428e8ba7d77aa13da880c69964bc74499",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "0bae3a9d0594a00a481425925a06c9dbfd5a7ddf8b31e83b3e50916b9427a8aa",
    "sha256" + debug_suffix: "a6d4fb19b0b2e364a3f728ccaaa0c279c020f138ac831ec07bbdc425bc03d9eb",
  ],
  "kernels_torchao": [
    "sha256": "32a1764e7e9befef99cf2b567b74c5a072607e12cd7ea0729710f266a850165b",
    "sha256" + debug_suffix: "1e57603452d07c1e6ed3f50f8bd0625108351f11aa981a4a4c982dc1ed489a29",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "641b8dad7ce87202e91194b5cbaed2051190bdc4ad71dfed87d076917a29265b",
    "sha256" + debug_suffix: "97d6db00b7e3e5f181a699d3e9d6e72240d0f991a914deb36eefdb026c8556b9",
  ],
  "threadpool": [
    "sha256": "8feda1dd6fd08aece3652c2fbb16a0b5ebf1a78f05f365aaf4c8ae2d3573e472",
    "sha256" + debug_suffix: "b3f6e9d8a2d8bc1a30fc3fc15545ed05198856be9517de53ca67b337386b5740",
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
