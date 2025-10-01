// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251001"
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
    "sha256": "3c592766098d33b5d09dfb526faac45aa64cac4fb9bf1df010701017a64ee81e",
    "sha256" + debug_suffix: "aa76a3494c598e81af295a25155dae2033478a4f95e84ef5b5f722582cbc2663",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b0390e40782fcc3af87ca2c9179a1e15ade2cbddefe9712854658542184ac906",
    "sha256" + debug_suffix: "0f51c16f01a20e6339748d19bea519b2412a80ce7b9e7839d7bd2d6dd2463e70",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6f112ebb8ae452e91b04b1e9c0089ab35ff115f23ba7a9abd7e23c5de3087aad",
    "sha256" + debug_suffix: "053f282ef1a6bf71b1551b83a80f470def47417a72daf9e0a5225dea434b9fbd",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "3fc782894922e6fcb53328de771eca5335a059953ee95b0eaf99e7364789a1dc",
    "sha256" + debug_suffix: "00d934dbc961a0fd662ffc7767ce7f862dde1c48a94e3defcfa7181371a5debc",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "742b2fdc5b3ad168c856a1086c9fe9abd80919e110ee084a57c315849cc050f1",
    "sha256" + debug_suffix: "c070eca619289a9d6c9a515a67b5a89429838f5dfd4f42d348c88d661b3db16f",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "fa823e05766b8b5ecd0fdabfae651a9fde607e12c414accca32cc435a39d4c74",
    "sha256" + debug_suffix: "1fb56f443171bfc946a82f62cd2d4f68cbd18ffbbcd44a60162d1fce181c06fd",
  ],
  "kernels_optimized": [
    "sha256": "28127889a555ce1444e2913f34d504a3acc61bb0c5dd2f8a1b1a2ed63ab00d9d",
    "sha256" + debug_suffix: "d40c3844aecdd5cebaa99d8daf9f99104407b1952bfafe06c803c0e332fc8b5c",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "8e9a71817e73a694eafc2a1a3497f2cdbd841ea12649b824458d4cef300b78b8",
    "sha256" + debug_suffix: "bd6ad58ffaf849ef0cd402755f571ce7ac52578a4e62a1609d4d7b3bbf673e5a",
  ],
  "kernels_torchao": [
    "sha256": "0b6e3b9f5276eaa5dc9641e1adf1d5c5a4b1794671816dd6b3cdce7b174d579a",
    "sha256" + debug_suffix: "922c04ea3671b738803a50f9a3bb3593daf11acf126804042ec501a1fff99440",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a48e5271ed2cf3970157c8469eef49a6cef07cb8320623765b4bb8faaca47e6c",
    "sha256" + debug_suffix: "4e3c9f07192c15246cdac5f4c7c529c181a6786f7d6624c2bb6784af530cbe0c",
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
