// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250722"
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
    "sha256": "3a358895b68a938f50205fde4d2f11c86da7efaed366183d55465f5d52ad4215",
    "sha256" + debug_suffix: "cfc00dcf10f3aa538b0308f3a750932456fa75e2288ce38eaefd50551fee6cc5",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "d9c9010c5ac804782138bce9366bf5cfc9dc3f0b0870476733594a366be49720",
    "sha256" + debug_suffix: "232a450e2a307ac4413a0f8573b6ed6840b11daafbcbc07ffa5882a8fdde6929",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "12ed234dfcedd81928d3177968cd02928d7505b43487453027484b945330db50",
    "sha256" + debug_suffix: "f4796758a6e7b968ec87313aab8f6e0d1d111230201d4629a6c0f9da8c9ef7c5",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d7d017e2e384154727f76297457931012bcb1cd214de6be15676306026b2ed5a",
    "sha256" + debug_suffix: "14969aa893fc142ffaa325723726486299228125f9ab8f956d58d8be164773b3",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "198316aacb1724e2c47f67c3dfc02ffdf43af213a3f72acd0b2c3d5771560cec",
    "sha256" + debug_suffix: "8bd48c61ed7ec43fc17488c54b6cd30ae697a40305ee5b78dee90cb732bc102c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "add2c5c17d98f76e4aba81ca4466fccd6e3587a40784c0ef1a13637d153e2f8e",
    "sha256" + debug_suffix: "b5fe63375b6af4adbe3a6efaab82d30dc559cc68b4182a70d05860fe77f29bbc",
  ],
  "kernels_optimized": [
    "sha256": "544b5a90a9a98746e8a19ec97ddf294e4960e784cabb72c22f92f789beb1da1b",
    "sha256" + debug_suffix: "e5e3cb0d726948233bff72f80ed255cd3481e7cb0f8a5860fd20fba2500597e6",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "08d98c1bd9cf4a48479d9c56998c6de4e3b95d505102890a436d49fb248bddee",
    "sha256" + debug_suffix: "19c3a859a684c695234ae3842aa4fe1969f8b110c533cbf32e4194c637b851a8",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "7cd545055d9044aa7737f25d864a258bd1e3c16bf2855d06b305f9a16140b686",
    "sha256" + debug_suffix: "5bcbf960fa041dab35a49f30076367468c0bed6ba530211391d465eed5bc0d20",
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
