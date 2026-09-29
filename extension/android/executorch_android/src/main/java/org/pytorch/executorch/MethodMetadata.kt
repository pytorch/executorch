/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch

/** Immutable metadata for a method in a Module. */
class MethodMetadata internal constructor(val name: String, val backends: Array<String>)
