# Backend Documentation Template

This template provides a standardized structure and starting point for backend documentation. It is intended to provide a uniform experience for users while allowing for backends to customize their documentation as needed.

## Template Structure

The template includes the following files:

### Required Pages

- `backend-overview.md` - Main backend overview and introduction

### Recommended Pages

- `backend-quantization.md` - Quantization support and API documentation
- `backend-partitioner.md` - Partitioner API reference
- `backend-op-support.rst` - Operator support documentation (RST format)
- `op-support.csv` - Operator support data in CSV format

### Optional Pages (and Subsections)

- `backend-troubleshooting.md` - Common issues and troubleshooting guide
- `backend-arch-internals.md` - Architecture and internals documentation
- `tutorials/backend-tutorials.md` - Tutorial sub-section
  - Use this sub-section to provide tutorials for your backend. Tutorials should present information about a use case in an end to end manner.
- `tutorials/backend-guides.md` - Guides sub-section
  - Use this sub-section to provide guides or how-tos for backend-specific use cases or functionality. Examples might be static attention or device-specific memory management. These are intended to be used as a reference.

## Using the Template

To use this template for a new backend:

1. Copy the entire `template` directory contents to your backend's documentation directory
2. Rename files to match your backend name (e.g., `backend-overview.md` → `mybackend-overview.md`)
3. Populate the content for your backend.

### Additional Customization

You may need to:
- Add backend-specific sections to any file
- Remove sections that don't apply to your backend
- Update the operator support CSV with your backend's supported operators
- Add backend-specific images or diagrams
- Update cross-references and links

Try to keep the landing page (`backend-overview.md`) simple and straigtforward. Use the child pages and sections to provide more detailed information.
