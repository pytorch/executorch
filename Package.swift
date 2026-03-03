// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260303"
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
    "sha256": "a1493d95df3b7845cc30278433ab0e87f73e97d57db12220ee0a5b88742aaf46",
    "sha256" + debug_suffix: "c18b548df8d38c06ba612f2b26bd5eabbb2fef368fae3ce2fca7bee63a02a351",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "59bcd53cab52bb51abe6b35b2f6b19fe1ae1191bf1a126e2e1b9191d256d1190",
    "sha256" + debug_suffix: "266c15ad60c6c195ffeb3c487cd3267196282a08b51d30a6861dccb6230d2b03",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "181fbc7e088b17fbe9b30d5798040b30cafe611fa22e5682a0ee50b519829caf",
    "sha256" + debug_suffix: "bfe031411d07f0bb4f9d72fdac5b003ff3ae4426a7a88a37b1c65f417b9475f0",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d09da265fc07aca862991201c2ca6cae8242c51f9f9588deda395fc943d1add3",
    "sha256" + debug_suffix: "79f530073fb514eae0014ade94eaa1ca0a296d21c7d0a4937575b06ad2f1aab5",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "cfb7af5a52397b1a4a68c5a33e5fe0b38f4592896c1ff30ffd248bfe938be1b8",
    "sha256" + debug_suffix: "153ba7128abb85fe553cf083d2e9a45c4029f80bda899fa4208cac881cd0fe93",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "80a978c46be767ef49569e68a845b9db2043097a4835c9604ebadc3efec6bdea",
    "sha256" + debug_suffix: "f7bc7653c8dabeb10ff05c2b53a8758c5838b915b9c640e99305a33d73ab736a",
  ],
  "kernels_optimized": [
    "sha256": "c7e6c27218ed93f3ea984f9b6975259f1912445541c6b1e750138ccfbc9b895d",
    "sha256" + debug_suffix: "16b0602f73f533721e491b459928fe273aa2f9f12aa23c4277bb8df166130dfc",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9a7fc4a6ad59fc790db92ea2c5a36247e675df44e918347e92ba477d6299a4dd",
    "sha256" + debug_suffix: "f6f6f0978c0cdb1d7ef347f5e263005d86a5826428badbde9d024edb0e4cdbe0",
  ],
  "kernels_torchao": [
    "sha256": "b02a20e90127e2957ebbf348611bdccd0f3f297a92b7372b02bdc495f0638f36",
    "sha256" + debug_suffix: "00a0e258f9618e0040f78bf0fa561559230389b2a67d96c0599b1c5da0f60816",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "60d5cbe213ed2824f553155d1ab5d9a5c548432e95d92c74d7e535c2372955d9",
    "sha256" + debug_suffix: "289501a9e8f96ccdf47237e1124401e3558f166febb7b02022e2e788ff6eb800",
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
