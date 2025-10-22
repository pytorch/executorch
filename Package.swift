// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251022"
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
    "sha256": "8a200d74ea194e561c358410fdd088565f1a043d24a49596e72bb51c6a7e60c8",
    "sha256" + debug_suffix: "e3acd3e192f71084f1c23790ce1629ca157472e0047d573ff9039392a4b2f18b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "0cb1d4772bc1f25d0106832d3bb3f3c9eca6000528fa76beff629b46102052a6",
    "sha256" + debug_suffix: "b0b24b12f8e5e0517b77122ce74afaff55b616878bb602fd9c43730cf9593894",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "8a4fce8ce1dce87337867d35875decb01b6a981cc9800f8e8ef3139df388ab4a",
    "sha256" + debug_suffix: "29b71c04861e08685e96eb5e76e93b2372f725b81806c60ef259a243cbfd5f1c",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "5fb787678e1fd3dade4345c9bd3897f0c168f015075ca7ee0fc50f182ad14eec",
    "sha256" + debug_suffix: "3e6e05a2834af1a1886fc10711813c45aba7f4855e812baa5bd9a6061d869f4e",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "96e40bd0eafcd449a7ee14ba86ad96a2168f92fd6f2624f47ebba4300ddef4c3",
    "sha256" + debug_suffix: "a24a80b0d6b7480badfb59773548600b0bd6485765320ea0ee0c5bdb0d27cee5",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "7e9393056b8a0628dfc258785d5bd21f60ec0a8eb1f940a73b290a4453306e74",
    "sha256" + debug_suffix: "503f0ffc12efdaa3b2fbbce5477f2c8f9b281c0ca259dd4bc4690842960c2a7b",
  ],
  "kernels_optimized": [
    "sha256": "0d5c421cd88f2dc895af0d9dc12b35b39efed693a0b43182c4bf7ecbfe3c4cae",
    "sha256" + debug_suffix: "094f9c28a1bb7f625d09c6f7e89a23d1ee3f2e7af8b2af88334acc1b19b9999a",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "373db0a1e4d9db39140161fa35d100c615d3ac8cf89d193658880a755d2312b7",
    "sha256" + debug_suffix: "9956b3577263c18631009ee0ce4e4c9292f0e4289532667ecdaa84dfcfe612b8",
  ],
  "kernels_torchao": [
    "sha256": "a9065154c839c9e7cfd2de177911f4cae3b00b2ae14d8b374b514d3ebf82ab54",
    "sha256" + debug_suffix: "6252a0cfb33a8f3fe662134226e08ea6f240f6109533c8906b693235ff2a50ba",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "82f4475deda412f6ac5a931e24dc436b6f9e8bd05fa1ec6eb62e3b1176122a21",
    "sha256" + debug_suffix: "634d2cc36366516d041cacb8f24e74ab54115091154298e4d890495e5e8f2707",
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
