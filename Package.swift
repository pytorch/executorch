// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20260915"
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
    "sha256": "ded1c1a024e25be519f05b75c9ab88b5b134e8601015f87811223327d951383f",
    "sha256" + debug_suffix: "4203a69e673cda9a02c764cc36676efef589725514d31f471298b726f8190948",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "81c740651136005982ffa465046c3698069125faa0d80c07488a2ce1888943fe",
    "sha256" + debug_suffix: "6831424e36f67e4f0111e8c8bed5d1cc61ce8972df0959ae0e2bc1f26e9ace36",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6f6785c055e59e8c4ee2e95698a2fdad742ea35384cb7584badbcffadf1b6edc",
    "sha256" + debug_suffix: "bb303829723f707960f13b4f11516ab0259648903b762671edf6c65b2ab43e08",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "374252fa81585bee3225085bb5869590f35e16a3553d7c2010bd736e3323452a",
    "sha256" + debug_suffix: "9dcc03471d8bc280e365bcac27131ae26938190c9dbbf6b7b6e2f8158493e248",
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
    "sha256": "e489f12615974a3cfb81ae04c157c3a4cf137867d6b7e15bbf17a8cbd62e8b4e",
    "sha256" + debug_suffix: "8312db44604229ecdb285a29f9d72f4bb47c3bbcaea831f162340887a3d504a1",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "d4f4cc541da03d5fcaa5ee0ae21f04d6f852265aed22e6c1b7e7f58695e536a0",
    "sha256" + debug_suffix: "de95706fe70c0932d44dc225c2641490059b9da2fc58616817e34d7fab2c4803",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "ad47ebfc96450961ce6661dc848b6fe9a92f4d74f43a34c5dba57978c81da2a8",
    "sha256" + debug_suffix: "cd07f289830105134a883cfae4e72bf3160c9c8abb0dcb1b3c734b6ee024a880",
  ],
  "kernels_optimized": [
    "sha256": "fe628fd57275341e0eaf22524fcff503470ada57d9640aaf64c1f36679b59c82",
    "sha256" + debug_suffix: "79b252ec9745aafd84ffb56ef6481e4333e5931ecd07a344a0de518819cc6913",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "95933f24958b168483c6db4014bd8f50a70daaca8cf1ab00e92b68363b665820",
    "sha256" + debug_suffix: "f681957d776531a3a8c9b5db4201ec9ebaa274f3e7e3cabaf3c326380de41c40",
  ],
  "kernels_torchao": [
    "sha256": "bc426e3dd783450eb8fffcc22329cbd7a48dca773de27ed444a09ef8c10235c8",
    "sha256" + debug_suffix: "e27bbce0b4aa7882fcad53379a2d1edc3869b852de6099f0d1374048eee4e48f",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "c31d8e8eedbb6bae7906304ac8c986f35d49f233b1142a5931b2b069de10811a",
    "sha256" + debug_suffix: "172b10814cb138976ee1208926e0f35b644a849dffd0afe986cd26a9b3e2a0f6",
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
