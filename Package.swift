// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260124"
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
    "sha256": "cfae1dd3c6de1bb05c4ca4c9cf1de22935afe405402e2b8a19d48dff8be52ae4",
    "sha256" + debug_suffix: "191a2bd18df1d8eaf88615534185bdc8402f3d242638137729e7e3fffb7e72d3",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8719b06a6b8fa413c1960322c17620bc0ac964f9de42bd433e760d13147bb894",
    "sha256" + debug_suffix: "07d3ee5517154865d70128f580a5751c25884899b8075c8d068a8cc7613a5829",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a892dacd4a7f3628a68e5e2118c8ac804757461f6c39a26768bc7372975ba202",
    "sha256" + debug_suffix: "2caa3955be9a7c8238db69e4fb0a7315365310d8ecbd87b8a42463a84f428bf9",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "0b3fdf75ce76ce2036d8906cd04ddc09becfd8cae76a871781ac8e0f75b25c8f",
    "sha256" + debug_suffix: "bed7f801218666f49f89dbf2b4df6ddec3c3b984afb6291d8a7fb7fc1f732656",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e13f5c704482068ff421c24c933fc57d9163428e19a74be1ab1e447df4cfcb51",
    "sha256" + debug_suffix: "7b2c82267658e63191c1ecf419a7805eba8b23a1670dc0f7d92c9a8b04af3fae",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "ccfe126d3329dbafedbceb7664dfdcb29d9629d8b6b5ad674b66add81fd5aaaa",
    "sha256" + debug_suffix: "b09b50352b0cd4285b2146c086c94d561a49179099e679a994488f78deb85fc4",
  ],
  "kernels_optimized": [
    "sha256": "2d2812ab4c0fd88e58a02ed6b34112ef9e608d3d108525eeef830aab20ddc794",
    "sha256" + debug_suffix: "5ab8e74de0e8c104cf79fc51a35099153e5a855287509789f14a4b525ff48871",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "111f3162cb79065bcc08026cd74f78d84d9855e7a841722eb3241c7065c4fdf4",
    "sha256" + debug_suffix: "f995578df8b1c4fb118ec42a4bbfb2a83d681ba05e3a54dc0e1d0e52f79b6229",
  ],
  "kernels_torchao": [
    "sha256": "f1bb03737d005292b528e0b834ff6da6205f870152f1354a6be039652bc847ef",
    "sha256" + debug_suffix: "f4c7f8345d992f43cbc79ac0ec631dfc4f767735dd78477ed0c888d885e23067",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "b73f5de89cda8c4d3b5d551f162fccc70234142783c9eac2e3e1827a33167cf6",
    "sha256" + debug_suffix: "9d1470e74162bbbc92a15db2563d036719fc8bf1fa9bddabefb7035bd6e2bc33",
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
