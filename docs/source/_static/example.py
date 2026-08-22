# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
A sample python file
"""


class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    # start

    def introduce(self):
        print("Hello, my name is", self.name)
        print("I am", self.age, "years old")

    # end


person = Person("Alice", 25)
person.introduce()
