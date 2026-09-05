// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260905"
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
    "sha256": "9d187618358e1e84b9fc93be6a362e9c884ec2562153a3573accf989c9825c15",
    "sha256" + debug_suffix: "e5ca92304cb6d35be0c072637ec2edd357782907fe3a262f4e40712a886c60f2",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "82d5c30e20dfef306bc482ac8480d05a976275ed77295445be43559ef76ae49c",
    "sha256" + debug_suffix: "949d3cb290a88c472aeae0d1c03846e12aa2293e5d65f4c4c47b7d6a9bdcbe61",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "2de1a7121574435153bfb1420f54f14e565be8b65fc6d6e355221c7d87902d4c",
    "sha256" + debug_suffix: "ca04765d243f24f082c896db57426cdc260252eb4d951d21ec187bbdfbc343f3",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "79145a695e0e4061afe7d6579c58dbab1576e58d1ad13191f1afe94ce73dd602",
    "sha256" + debug_suffix: "295c3f1d090008ebe2eb69d0d362498b770308efd4a3de702bb186f322cb567c",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_dump": [
    "sha256": "3c66d78851087584afee636d6e159dc85792fc80d3b56290fbfebe623aa4e948",
    "sha256" + debug_suffix: "92c41a586cb42068ec46891658d67e3f3de8a0ebf19cffcb2abc3a6c2bdf5ad8",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "344e94af92361dced3fc570791250659ed187cab8768958675fac32cab4984c3",
    "sha256" + debug_suffix: "d81099953fbc4c5227f3183f9606197a650973e0158887c56a5f538089702563",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "1abb81635aadcbfd632bd91819e03a74295a7007872404d7d30665c96bbf4f74",
    "sha256" + debug_suffix: "601ed1a6ad06a289b51afb3ec6152fdc0fdd1136c09c09811326e1e6c099cbf9",
  ],
  "kernels_optimized": [
    "sha256": "4ef620b572b31dab7e884395f02a4414f472e309ec1f74e195205816d7df41fa",
    "sha256" + debug_suffix: "14bca748bbebf1fdcfd7182178a8ccb121c3025f126b9db92082a00b484501f9",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "2a35cf2c908d7cfd49011e15e84b0f3a3a68c4894dd171bf18ffd5466297e17b",
    "sha256" + debug_suffix: "3dfa3d4eb42ba234bca9c4fcc75e91873ec4f52991fc2d9296c84e2a0b76e293",
  ],
  "kernels_torchao": [
    "sha256": "2c0c40887ba4a1fa1e399cc96a34f8368b35bafe9312065955930aec47ce9393",
    "sha256" + debug_suffix: "fdc33f63100111a364f8f266113d7298c082da5860ea7ec084cc84afd8f9aed8",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "56a17dd91a2102ead9b97bec548de81cefb7402e5896c2273180bc491ecb3213",
    "sha256" + debug_suffix: "74278d8c1ba60a633d9fe1f8118f2f02f38050743b3ad4d1b7da65ba42e594a6",
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

// The MLX Metal kernel libraries, one per platform slice, shipped as a single
// resource bundle both MLX products share. Kept out of the generic loop above so
// there is one bundle (executorch_backend_mlx_resources.bundle) rather than a
// separate debug copy, and so the release and debug delegates resolve the same
// name. Each slice's MLX binary asks for its own mlx-<slice>.metallib.
//
// The release job commits all three files before publishing, so they are declared
// unconditionally. A missing one is only reported at package-resolution time and
// does not fail the build, so the release job has to assert they arrived.
let mlxMetallibSlices = ["mlx-ios", "mlx-ios-simulator", "mlx-macos"]
if products.keys.contains("backend_mlx") {
  packageTargets.append(.target(
    name: "backend_mlx_resources",
    path: ".Package.swift/backend_mlx_resources",
    resources: mlxMetallibSlices.map { .copy("\($0).metallib") }
  ))
  for suffix in ["", debug_suffix] {
    if let index = packageTargets.firstIndex(where: {
      $0.name == "backend_mlx\(suffix)\(dependencies_suffix)"
    }) {
      packageTargets[index].dependencies.append(.target(name: "backend_mlx_resources"))
    }
  }
}

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v14),
  ],
  products: packageProducts,
  targets: packageTargets
)
