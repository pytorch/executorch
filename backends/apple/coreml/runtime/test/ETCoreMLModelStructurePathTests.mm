//
// ETCoreMLModelStructurePathTests.mm
//
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <ETCoreMLModelStructurePath.h>
#import <XCTest/XCTest.h>
#import <program_path.h>

namespace {
using namespace executorchcoreml::modelstructure;

Path make_path_with_output_name(const std::string& output_name) {
    Path path;
    path.append_component(Path::Program());
    path.append_component(Path::Program::Function("main"));
    path.append_component(Path::Program::Block(-1));
    path.append_component(Path::Program::Operation(output_name));
    
    return path;
}
}

@interface ETCoreMLModelStructurePathTests : XCTestCase

@end

@implementation ETCoreMLModelStructurePathTests

using namespace executorchcoreml::modelstructure;

- (void)testPathConstruction {
    Path path;
    path.append_component(Path::Program());
    XCTAssertEqual(path.size(), 1UL);
    path.append_component(Path::Program::Function("main"));
    XCTAssertEqual(path.size(), 2UL);
    path.append_component(Path::Program::Block(-1));
    XCTAssertEqual(path.size(), 3UL);
    path.append_component(Path::Program::Operation("x"));
    XCTAssertEqual(path.size(), 4UL);
}

- (void)testPathEquality {
    {
        Path path1 = make_path_with_output_name("x");
        Path path2 = make_path_with_output_name("x");
        XCTAssertEqual(path1, path2);
    }
    {
        Path path1 = make_path_with_output_name("x");
        Path path2 = make_path_with_output_name("y");
        XCTAssertNotEqual(path1, path2);
    }
}

- (void)testModelStructurePathConstruction {
    {
        Path path = make_path_with_output_name("x");
        ETCoreMLModelStructurePath *modelStructurePath = [[ETCoreMLModelStructurePath alloc] initWithUnderlyingValue:std::move(path)];
        XCTAssertEqual(modelStructurePath.underlyingValue.size(), 4UL);
        XCTAssertEqualObjects(modelStructurePath.operationOutputName, @"x");
    }
    
    {
        NSMutableArray<NSDictionary<NSString *, id> *> *components = [NSMutableArray arrayWithCapacity:4];
        [components addObject:@{@"Type" : @"Program"}];
        [components addObject:@{@"Type" : @"Function", @"Name" : @"main"}];
        [components addObject:@{@"Type" : @"Block", @"Index" : @(-1)}];
        [components addObject:@{@"Type" : @"Operation", @"Output" : @"x"}];

        ETCoreMLModelStructurePath *modelStructurePath = [[ETCoreMLModelStructurePath alloc] initWithComponents:components];
        XCTAssertEqual(modelStructurePath.underlyingValue.size(), 4UL);
        XCTAssertEqualObjects(modelStructurePath.operationOutputName, @"x");
    }
}

- (void)testModelStructurePathEquality {
    {
        ETCoreMLModelStructurePath *modelStructurePath1 = [[ETCoreMLModelStructurePath alloc] initWithUnderlyingValue:make_path_with_output_name("x")];
        ETCoreMLModelStructurePath *modelStructurePath2 = [[ETCoreMLModelStructurePath alloc] initWithUnderlyingValue:make_path_with_output_name("x")];
        
        XCTAssertEqualObjects(modelStructurePath1, modelStructurePath2);
        XCTAssertEqual(modelStructurePath1.hash, modelStructurePath2.hash);
    }
    {
        ETCoreMLModelStructurePath *modelStructurePath1 = [[ETCoreMLModelStructurePath alloc] initWithUnderlyingValue:make_path_with_output_name("x")];
        ETCoreMLModelStructurePath *modelStructurePath2 = [[ETCoreMLModelStructurePath alloc] initWithUnderlyingValue:make_path_with_output_name("y")];
        
        XCTAssertNotEqualObjects(modelStructurePath1, modelStructurePath2);
        XCTAssertNotEqual(modelStructurePath1.hash, modelStructurePath2.hash);
    }
}

@end
