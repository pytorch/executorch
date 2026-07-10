// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260710"
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
    "sha256": "54a925585132707a1fa9e6a6c7f9279667848cd3acf3861ee8e05c4d4e96692f",
    "sha256" + debug_suffix: "ebe3007904b87d79d9fd989125e3ec60c8214e0ac121bc4645b681960277444c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "bd75794631b199aaab78c0cd27b2e8dee9eed19dba896184e2900705c0b77b2f",
    "sha256" + debug_suffix: "4127b9919539e5e3b2c8923e3c01cd072b1b23b3aac52817957fe7a1c72f65e2",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "69bfa585fdb91f3df003c520077e10c34a251da9eb0f52ecaf46795f49b39fb1",
    "sha256" + debug_suffix: "ee08d0b5515199a0d21e1e00301450d30ebfcb38b73d323b092423d50edcdb52",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "8474a79fcb14ba438f79804056d3e96fb0c39608c7159b562629eca1bcde3bf6",
    "sha256" + debug_suffix: "1eaafba47c3a4d7bf7bd9c878ba7aba2f11ab7fef7a0816a457b8bae8b43480b",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "bac0aa0f2d3eea349e87f012f88f816be4c9f4debbbb2a68fc161f8964f84e33",
    "sha256" + debug_suffix: "6f8d017485f8f78e0cd130368d5f217102b8d5c5eb3630d76f61e869cad39727",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "d1000f1ef8e47005c21dab9fc9a974f28e37bcc0c15f8cb0eaf1997e249b8528",
    "sha256" + debug_suffix: "ae12741e450315a8f864a98c653ca801d70a08f7892d8388bae90806bc98e61e",
  ],
  "kernels_optimized": [
    "sha256": "09d73543066cfc3843be823997e9cd3ee8ff62dfc4566befdd4d3b2db0aa5764",
    "sha256" + debug_suffix: "a382eeed59872493366188e4f2a308e8fea286f3cfb59839a3ae14b421d7e6b2",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "83f5b16e9ef437894589953a4266e21a32d034082e31b1d1678cda3aa9627131",
    "sha256" + debug_suffix: "4a40f0392a782245e6d1e3013dd7e625617052616e593693c124f936f0172bcc",
  ],
  "kernels_torchao": [
    "sha256": "622544859a60aff99bb7844b9c8b6fc3c340381ea85a8059684bbfa4f77f1130",
    "sha256" + debug_suffix: "9e3d4f9c600158764b38a9bfe28053616a0b6c37993b6f7234e70dbbc3da4bd9",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "213413ea643643acfced49b0956bb447ff2cd0b5402ede21b4b6d431ba925f19",
    "sha256" + debug_suffix: "f3459ba773c9e8e46de27e335699da75453ae89e0f8bc997eb1934b5ee762335",
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
