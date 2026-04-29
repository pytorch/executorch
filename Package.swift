// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260429"
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
    "sha256": "e438ab2a9eb1c1e4afa3a788b455fe799a0a2a3f1bb3c1fcaf52387c1347a3d5",
    "sha256" + debug_suffix: "f66baf1465df9356e693946151277983f1b2beef803e42e65fd1a9094855e716",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b4b1e83e528c8ebfddce7de6f2dc1fe6fbd91faca882013c57340704b214b838",
    "sha256" + debug_suffix: "c850adf72b215d502881936b3df7ee06c5b7efe49c4738f207f2cfbf2198ab38",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c0c57b89696500ecaf21883d67da804df2d06617632856ef0c8eb0feee9035c8",
    "sha256" + debug_suffix: "9b856f0cb9d95195e3a01edb1e475364ae1aaee45bd219c33ac66e1387d59d6c",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e150c23f3c609ae5d4e20b945b629eed4688098cf39f99607ab5bc91d2ce4a69",
    "sha256" + debug_suffix: "205e2673ca86ed09266eb4a31f4d0a5c749900e0db9cad67b0b05ef5a2723d1c",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "28f547c6b6b518caa6ad887acab31342fde8535b577f5d6f2680f13dfdc63df8",
    "sha256" + debug_suffix: "a31620e5028e2e1f6dba5dad44cd03c83bee9de9152cf4a68eee042222a2161c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "add8666b3afc27ff7585c409ce4bded7085f88236c89552530638bcbe2cee81d",
    "sha256" + debug_suffix: "0087e8f087325a6d74052a82abc0c9e95fae9e7e3ec3d555136eedf3ce8ad6c4",
  ],
  "kernels_optimized": [
    "sha256": "48da639122745de0f645d159d3440f423a1d55f5facaab6262b2ec4bb40ddd19",
    "sha256" + debug_suffix: "f81a679123a2c810f0ec6fcb6d80ddd2aa0b0d230ff204e7deb60e6beee64d36",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "64b0e6c3e826c196ff21b275b2958d0ecfc92fe5489c2920b2a2547117878c22",
    "sha256" + debug_suffix: "bd79616be1c6c57fab8101b05be7f4a9004b9d08ad51a606a509b3fd69c37427",
  ],
  "kernels_torchao": [
    "sha256": "018b659203139766f42c42a6cf96f9780141f7632bf40b4098a6261e0fd38190",
    "sha256" + debug_suffix: "a6c5c6325f909c72f01559446b62a5d50d229317198e33b9465580ec0b6dd0bd",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "cc4279e1bf6f4b6af95c2fe6406f4d980938855fbfebc9714a1afd0048a4f4ec",
    "sha256" + debug_suffix: "a8f0468ff7a023159afaf20a4982fcc1388d203e2a46873f9911bf2743b22fc7",
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
