// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260723"
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
    "sha256": "82d81b0a188d016387767eb6bd1cc80679eacd4f5ae4c1e0444c5d08ff16a21c",
    "sha256" + debug_suffix: "150fda19a53f08b312c2a2e2d3414e1571ce7084a8fdc85443f2afb105090af2",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b17cbc2302c9a0fd859688decc313a2ae53815ab3fe2a36b823b578624524812",
    "sha256" + debug_suffix: "dd9ff10a5603cff10761362cd14f09aa483dcc19ae4fb2fed84e335f618decdf",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "7a7b2f839a121143d08857e998872048924ecf94c4a9c2f2c1c4a3931371d326",
    "sha256" + debug_suffix: "d88e517eb9117adacf1ffa4118341e1aa72d1943e22b241186ff0f36a07c8e57",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "760aed5ee1cea3cd7feb1681964c38161d4755ac848b411ac2789a4397f7afba",
    "sha256" + debug_suffix: "cfbfab669f5bfdc98ad0d345fe0cc1130f23a1233a0565bc2f8424f982499687",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "55a7834bf62b88176fdd33608770b10af9c4ef4248b0f02c4a97544e0b50b8cc",
    "sha256" + debug_suffix: "be511a72e2ed58ab1d00cd0702578146a13d9210980147db81915a9706cc15c5",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c6672cd5631c15320fbbfc2083fde75f1b685f45351599f550c8b46bafe30d11",
    "sha256" + debug_suffix: "90993c02907bb5205d40a720fc620ecee8031ff726be7c73e8d8d17883960001",
  ],
  "kernels_optimized": [
    "sha256": "59d5675f0e66b78181243f08eb59f208ce75fdef9e8baa0986e47e67c3c6643a",
    "sha256" + debug_suffix: "da981cdb2b5abaf78d2fccc97a144b8143173c1d022c5c4c7190468c0e8d8572",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "0619f916de8d5c162c92c9e0d7eaf5123db0498a70c82e7019bb2cadabe0bddd",
    "sha256" + debug_suffix: "534c436252f944c354e68613d6df37cf48eb7bf42041f0a58b27563d7f248136",
  ],
  "kernels_torchao": [
    "sha256": "eb747d92923353337cd0fa6f2975f7966e99f8140cc8a3c5e2e6a9d5e423cc38",
    "sha256" + debug_suffix: "f6448e9791e1637d0b86580640b23f0ee0d792629f4cc85bcdd1239aeaede535",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "f038e990efe130ef77d0d5205995dec354b9241c4b21811c0f24279c2ffb2c5d",
    "sha256" + debug_suffix: "b33c480275e5aad3fcb430012a6aa823ed75b885a80fbdeddd1048196e85b732",
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
