#include "HelloWorld.hpp"
#include <TargetConditionals.h>

std::string HelloWorld::helloWorld()
{
#if TARGET_OS_MACCATALYST
	return std::string("Hello World from Mac Catalyst");
#elif TARGET_OS_MAC
	return std::string("Hello World from macOS");
#elif TARGET_OS_IPHONE
    return std::string("Hello World from iOS");
#endif
    return std::string("Hello World from an unknown platform");
}
