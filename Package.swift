// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.0.1"
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
    "sha256": "e80e6e546dad182ca4564f3c4b3887fa613b89fc524e2f0e49b2ea17bfb911bb",
    "sha256" + debug_suffix: "393d8f28a64169a64514bc674db3160d121ebaf3ec555f50d8b84a2800edce2c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7ae31c42dca0b45417e584be1891aa49b26dc64dfb44d7b73adcf27af82f4bb5",
    "sha256" + debug_suffix: "5b1d08924b7c2170f992d0fcdbb88306021ca7c916b35e88b58640975d34b729",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "420d501f39d93df38d924de726a17bbf7711ebba90edfcfc225d3033963d6168",
    "sha256" + debug_suffix: "a972e5b23de7b4d4fe03f81aa930d60ec1604a78d58578d63b3e43ec8854a715",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "640299f2e98455b99a579c80b2e4af31beef152e6a4bd91d33d21dba07bc8d93",
    "sha256" + debug_suffix: "3f7866d439d4fcb880035041c00b1de0656dd014153af0419aa074f71466b7e7",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "adac791f711e233d6b1dea429aa507105377b931a00b49fe029b47ab7e62456b",
    "sha256" + debug_suffix: "f789c98b1a122dfde1f68d60571cfb61470f46ba382d0c6617772bf528e4fdd2",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "f510d1f40d56cbbb0d47a372041dcb4c96c670273c2a25e451d6dfe5e1825621",
    "sha256" + debug_suffix: "650e8d87fa1dbd554e4d61f4cb14186bb0bd12f357f3cae2989830e1381c8e97",
  ],
  "kernels_optimized": [
    "sha256": "0cfcb42170274e38b921d56f32b5a8ca793704285cfdf200c74f52d68a1570c0",
    "sha256" + debug_suffix: "7ca9203724c38544478b940105853f32fb06d24b072b5cdb41007b8e832803a4",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "a59b7c5aeb6b523790bfe2b2e21aa33ad86f5c7386e87c7146a0040a01c568fd",
    "sha256" + debug_suffix: "e5b69b479556a910b2992dfedea6d21a6a298330f7a72dd288d8620bf4620670",
  ],
  "kernels_torchao": [
    "sha256": "4b8e298fe2ab666163079cb1a7cdd254d7043e65adecee3c33f9ae54d7a781e1",
    "sha256" + debug_suffix: "93d4f3d64dcfba8ff40eef1624896fd5beaa6a726bf81cb2fc32876939952f9c",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "e397d74e6e09507a180d3bfea09ca49c60b77f0ecf40e8cfa59a02a641ddacc0",
    "sha256" + debug_suffix: "4cea78c39e807aafae444d397add7bad596b75427b8aed6797f741c16f600092",
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
