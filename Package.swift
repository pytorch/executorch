// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260314"
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
    "sha256": "b0c154923e2cca57ab63926df888132f10c527a02f7c9ea91e58ab2352468d00",
    "sha256" + debug_suffix: "01952d3d81888b4b328495fa1b7a0b56c9d28dbedc238a0063af0ca48ca35d6d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "d34c87323d9e6adc3b3e3db46be70283dccd6a2760559bb3fe188ff0e68eaa21",
    "sha256" + debug_suffix: "f6ab2efaaed5a1bf68f7e472231d4c288dd3e342ca429a036c69dd572866f617",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "47a845141cdd966e86baaee30bc6d31524a73d56f3dc8211347d405911610d5a",
    "sha256" + debug_suffix: "22db80bd2a3a1d94a925625a90047afb77e888297f1fa9a7258d21bb2a8abe7e",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "38501c74eb0797ee6febee798c5960a9399d621e5de7bb3214cca194d45d3326",
    "sha256" + debug_suffix: "85851f25b3f863d8a9aebbe2108ac57de3934a70c97cc84ed758403feb6b586b",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "957caa67c91662d2857ae9c018960b04fd8e0b848146b079f4ad47366f976ae8",
    "sha256" + debug_suffix: "eae36d51a59626a67ab2be63a42de49f9100d0cc2173f6bb936fd1d2c49a6e00",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "2f5ad3f1acade39b35cd23fa41d3f70f86df3d9a5162e27e12c43d6306722064",
    "sha256" + debug_suffix: "cf8184f51179a38ddbca481bd6ab56d73e1ce3413364513041f4918574b31cb2",
  ],
  "kernels_optimized": [
    "sha256": "804ffd2472fa742aaf1a7e19a8779e0f4f36be1abeb38d6322521da76704d8b6",
    "sha256" + debug_suffix: "d536fb467a637fddb888a4ad0e1afd1f84eaab0ee35dbf40df71ea80020cfbce",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "bcfb2d399de094d9de6864c04eb5134a632eb8436081dfc2edfea96dad7f459f",
    "sha256" + debug_suffix: "054abf78386f74686038beec9bddf9df5ee1fc086f8ddffe67f01a6d38d6e28b",
  ],
  "kernels_torchao": [
    "sha256": "90bfdf2c7cfd9e7d74721c4c9232361ddaebb3dcd32716d858131cb5c2dc0db5",
    "sha256" + debug_suffix: "a9689c37e9569f99830d6823b3ccf549639734f368a5536b69debf6e5c13e2e7",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "5b7020df2e16606595d199e53314da36872cfeeb7c3fb363d449e5e3440cb5b3",
    "sha256" + debug_suffix: "3a4697a96d57d3e61e2b1e117f4a1d04d48bce4808cee2d96ac3bec730cb911a",
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
