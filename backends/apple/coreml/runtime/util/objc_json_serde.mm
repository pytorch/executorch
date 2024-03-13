//
//  objc_json_serde.mm
//  util
//
// Copyright © 2023 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.


#import <objc_json_serde.h>

namespace executorchcoreml {
namespace serde {
namespace json {

id to_json_object(const std::string& json_string) {
    auto bytes = const_cast<void *>(reinterpret_cast<const void *>(json_string.c_str()));
    NSData *data = [[NSData alloc] initWithBytesNoCopy:bytes length:json_string.size() freeWhenDone:NO];
    NSError *local_error = nil;
    id object = [NSJSONSerialization JSONObjectWithData:data options:(NSJSONReadingOptions)0 error:&local_error];
    NSCAssert(object != nil, @"Failed to deserialize json string=&@, error=%@", @(json_string.c_str()), local_error);
    return object;
}

std::string to_json_string(id json_object) {
    NSError *local_error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:json_object options:(NSJSONWritingOptions)0 error:&local_error];
    NSCAssert(data != nil, @"Failed to serialize json object=&@, error=%@", json_object, local_error);
    NSString *json_string = [[NSString alloc] initWithBytesNoCopy:const_cast<void *>(data.bytes)
                                                           length:data.length
                                                         encoding:NSUTF8StringEncoding
                                                     freeWhenDone:NO];
    return json_string.UTF8String;
}

} // namespace json
} // namespace serde
} // namespace executorchcoreml
