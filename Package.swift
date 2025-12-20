// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251220"
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
    "sha256": "ba5c8eef1913c3bbde88a90dc38d752b276da2f58365d7f306b4749b7205c251",
    "sha256" + debug_suffix: "72a0050dfa058295e688567c2ac616b32fa3e8393650907f131b8b2c2026fb24",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "5700fc6bef6146a077b88e81cc3a2d335bf36d4acc6b770ec71b78a58fe1f1ac",
    "sha256" + debug_suffix: "cb05d777574235ace274241817ae6c0ad74b7d0f02cab2f9f53ca5383482ee4c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d6013103392d02163b0b0cc8e7c6a3fbc5dbaf3e54d9cd3c4186301b5e8dd8c8",
    "sha256" + debug_suffix: "7cdc6f1daca1d4973165985b7b97f6dce167c0328414bd444ce3ca62c5b09a05",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "43bdfe60d16bfbfdf1fe5413b96d77e02f8865a6cad883253451b1f0c15caaa2",
    "sha256" + debug_suffix: "f9bd75608489f1d909118827b7e269b41c5b50f215e6be30ea794a88f5944aee",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "49dd5401aefaf4298d67524d8fafbd9858abb61f19535001419a9d887eac3be7",
    "sha256" + debug_suffix: "0e9056e91791911e320b0e47c837bd38e3cd428fe7c542676bdfca48daa392a7",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c1e6a90473b5bd3e4141646304da8157b0efd42ab3c67b85a529ba472f2549e5",
    "sha256" + debug_suffix: "b669f60114b38613b3f5eb0089e15f65bfc92405a5c5424deb19567073d07721",
  ],
  "kernels_optimized": [
    "sha256": "ff29cef962c83180eca1552b89e61d373e16904a004a5fd274b886975996ff96",
    "sha256" + debug_suffix: "79ffc2eb36195b05ae520f198279bb6765ae9b945473cf2b177c02d8186065b1",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "fc21acaf76f684ed4775f0b38aa617128a508fb0f558ad3f06a8e1d082058042",
    "sha256" + debug_suffix: "e79952de88cf5d1f5ea733cec5ffeee16058df7a38ade1f5d07d3bf85d8810df",
  ],
  "kernels_torchao": [
    "sha256": "9392e889b632596fa4afe222ffd97b4edcefdda06fdfc8dcfdf0dba7e23b2c9b",
    "sha256" + debug_suffix: "a07dbb473062c0e0c70678496d392cc6291c8b8546a618ac29d71b2723372e44",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "6f68de85517a47795f795a083cc6e539ca0477a3c081bd34b98fc69da98b94c2",
    "sha256" + debug_suffix: "1b7368c6feb36795b0ce00c3cae42f563a064a746192f800271f2c673c2d4400",
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
