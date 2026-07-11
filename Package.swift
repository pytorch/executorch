// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260711"
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
    "sha256": "3bd4ddf6dc5a250119dd24a4839da98e4a2c7ea6b55e198ad37d44f5a9702519",
    "sha256" + debug_suffix: "c18aa993429e18bde219b3eb13837393970cdf72f8cf852dfc62a9f83e05db9f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "737230110c6ab819bc384bbc115724f3d5d6de37a7439fe81017e1d1f2bc12aa",
    "sha256" + debug_suffix: "c5aa55d2c4a2a22e158d0ee4a1f14fe6b1a2f892de839cd2eaac22adcb59f6ec",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "5985d40382106949e43ef7cc851fe8b3d8d0f60a948dee93377e941afb70d405",
    "sha256" + debug_suffix: "0b1aa681ee4c70463b301e2f61857c96642efbbfcee078a290eff70cbaabb6bd",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "59d28645d299dd604ede7cf1c56ecf54b0b3ca944a0e67c6ee3501a0da850658",
    "sha256" + debug_suffix: "85eed3db4ddfb8aeda6851c6cdfb50b7f1d3ba9dfe2e9436d8caee2af44f6514",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "2511ba40281adf71c4b95e734b4e590d16e7f83d893c4a6bf20dcb3c35c1f021",
    "sha256" + debug_suffix: "63e17c7a247c07e77bb91c60f1c90b7dfa0645c04def4d93f2c051beb6f8296c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "31f73fec8c79a241a72cab8498208124c5fa814c07f04836a8eb36f36e42e113",
    "sha256" + debug_suffix: "0017d939db96635e63ad9ab40dd2a8faf02fc862506101093c303aedc801ba70",
  ],
  "kernels_optimized": [
    "sha256": "bd3c95c0ea37f30f62912108fa348565a44a35b576e2814a06c96ceb39a848d2",
    "sha256" + debug_suffix: "830a333df402e89e66825b79ca1745762fae9eed9d333fd430cce3497cebc416",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "f21c3073483cc4bfba70a692f111db0d8619d147abe77f7c7324bb1aede1abdf",
    "sha256" + debug_suffix: "dbd2c128572f6fa863f99b258a1cc902305bbddf38ec8594b0cebd2c56a319a0",
  ],
  "kernels_torchao": [
    "sha256": "dbbbd01f0881100873e4ae47cd8b97f3bd42cf6a8bb02f1affda0824ba401194",
    "sha256" + debug_suffix: "86a7976135efd6c00411c9bf22556db7a2cc6dc4f46957829d99ee9ef6ed6674",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "33aaa242353c189ea4105cc9ec00c191db5df1494d8e8e5b3998e049de6b55ae",
    "sha256" + debug_suffix: "16b5e5f01f17f1988c7c8fe39178c95023b77afd50335bdbc8784942f8d4d750",
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
