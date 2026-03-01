// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260301"
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
    "sha256": "0e8b82667f5a403d5402180c71bb8ef65cd7deb22186c720d686854fbac7fffe",
    "sha256" + debug_suffix: "a6436dd2915d21f292d4c0033d17ec58377dfa4a3a9b79e1c6644c74a6bd1659",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "bccf627fb0cbb57ac27b9814ba8ad9eefdf8156154a8f177ff124320822cd5c9",
    "sha256" + debug_suffix: "0cb78af0174e26aedfce7a8d67decfeb6cb5d3e42e80c91b4760ab328f311cbb",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "364900539883b415296e9eed18055b0736b3b05e47c37364f6c8cc19d6ef1c4b",
    "sha256" + debug_suffix: "4d5c69226880e33ba7f44e14f3030708fc3052d88b8cd6f4b1f4c30122147885",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "629de6d9561c4f948ba05fe5e09b5fbaf0f52158cd1ad7ac8219e494e00d3aa7",
    "sha256" + debug_suffix: "cb465f82dbb04d6eae7792cc8a2e62b09a722c7ec60a05d079241637ce4a9506",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "f99331e158f7910d3659f76f86b8c97abc2339ddd07dafc1b81a493deea9e0cb",
    "sha256" + debug_suffix: "7c9d77ea7a3cb88fba93b1fe2fb23f3c51743764434ed82f9c2d3878032042d4",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "a3ed86329db6d5bc8fb4037bbb1c2dc40343467d186dac454a1aa68a779f7870",
    "sha256" + debug_suffix: "a51591003a94746c213a2ad130cbb635c0a1036ddf60e32ef615c54896b9058f",
  ],
  "kernels_optimized": [
    "sha256": "b4d0bc69713126bf1c6e9403af1238d81518b7e93f49f167f5366ffada8828f6",
    "sha256" + debug_suffix: "4e4e9754e79f9070ab4bf1273c6c90500f5d9d9411849580669e7a3bbe591f3f",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "5f3efdf08246c7ccaa891b9820343612aa56c617c7e6d0474d671e1aba5db176",
    "sha256" + debug_suffix: "1b9b5a0be872777dc8ac7f8fd02daf5727d03ec1347773c2e8086b7fcb3634b3",
  ],
  "kernels_torchao": [
    "sha256": "66ca82eb0f45c856ec9caa3b1408fabc8c16e4f29c3cd2e6c3d1e6ba36af3461",
    "sha256" + debug_suffix: "39db11f5724cc6fc39e708ac089d2b3222c3a086405847c748c821e77fa0e88a",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "ff6fad0f205916b91e90cc507104bf0e8e90c46163d537d2f218fbeb4c5591f5",
    "sha256" + debug_suffix: "fd6d67dd75241e3e87e30a989d81fff8b93f6b8fa24e0573cee04b12d9f792d3",
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
