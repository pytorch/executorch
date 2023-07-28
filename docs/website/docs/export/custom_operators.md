<h1> Custom Operators </h1>

To ensure a successful export of your model, it is necessary to provide a META implementation for any custom operators used. Custom operators refer to operators that are not part of the "aten" or "prim" namespaces. You have the flexibility to implement the META functionality in either Python or C++.

Note that the official API for registering custom meta kernels is currently undergoing intensive development. While the final API is being refined, you can refer to the documentation [here](https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit#heading=h.64r4npvq0w0). In this document, you can find detailed instructions on how to write a Python meta function by searching for the "Out-of-tree" section within the "How to write a Python meta function" section.

By following the guidelines outlined in the documentation, you can ensure that your custom operators are properly registered and integrated into the export process. We recommend staying updated with our latest announcements and documentation for any updates or improvements to the official API.
