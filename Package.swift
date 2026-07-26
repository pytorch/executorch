// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260726"
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
    "sha256": "ff8d77d8877b55f73fd9dc40237a814840d84b0e822c1fc8bafd1cb9948f36f0",
    "sha256" + debug_suffix: "416468e6c248b5c80580212d24d5652c5e02c507ffab403b6c1ab587dd16f265",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "4e716287b4f2d8afcf9db26f503edb0549d946842ae91b88b0558fce0502ea43",
    "sha256" + debug_suffix: "51f7e983fafa0121d49c6a83af7504b06fc1862c8b6c0ad797e8ebf161972c07",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "269017c3b8502abbf875837ab7e953b62241a6c9aa72b84d068a4f59c9486f18",
    "sha256" + debug_suffix: "dc8b885d9108aecb4d62f710604263c7fef91a022aa27525c5db047bcb34ac96",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "852e83ef096c6c0818e2e8da23685c156892db6637d0858e1315817933e587dd",
    "sha256" + debug_suffix: "e6b4eb2f280c5b9bcf28361f5eebf5b8122b714dfc4faaae0967ea090aba64e4",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "9816f5aa9a7ebacedba5765e252aa3ed04ee9ebad6c60ddfa3955ab4f16cbd39",
    "sha256" + debug_suffix: "cd7326d0f485e643ee7e6eee216ba17090c51a05881ef242f58fb111fbac9223",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "91ddd7a0862aef8e81762d19254e839060d0ba2e834e0bec50d866731b93175e",
    "sha256" + debug_suffix: "0e64edd16d936c141575d29a2bb0ffd4f8f7da2df303250392a696211fd1dbcb",
  ],
  "kernels_optimized": [
    "sha256": "ca88fbbd54b763bc0fea6ef82edbb205c984358708109d1f05849abf114e2daf",
    "sha256" + debug_suffix: "888967269f0f2b36b68ee913adf5fbbc618b94789421039de20fb5ec5edf87ec",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "e4b2424585b0bf9149d599c8278c94d3d7408aee5effb4f0cb48eeaeb7c8a87c",
    "sha256" + debug_suffix: "d6c5f395326d5b97d402a6eb043441a2e3957d5ada8852b2df5057a7860ab433",
  ],
  "kernels_torchao": [
    "sha256": "d402ccaabf63e714e2e61464e464d663a67b7d57f84ab3642cef760d4724be2f",
    "sha256" + debug_suffix: "a6658a68222a03db03d016517513b06a1eedafd7c4bf8ab1b22faa5f22dbb49d",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "7c2f1dca6dfe482ed5ecd1fc936ba2cc0354992680f228d12031c7bab0a600eb",
    "sha256" + debug_suffix: "774ad830ef2bcf9544ed4883dbe680ee96fa23e2cb0031b3f4f1fb427a598f9f",
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
