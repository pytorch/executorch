#import "HelloWorldIOS.h"
#include "HelloWorld.hpp"

@implementation HelloWorldIOS

HelloWorld _h;

- (NSString*)getHelloWorld
{
  return [NSString stringWithUTF8String: _h.helloWorld().c_str()];
}

@end
