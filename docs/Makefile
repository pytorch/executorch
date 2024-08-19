# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
# SPHINXOPTS    = -WT --keep-going TODO(T165752164) fix sphinx warnings around preprocess macros in cpp like ET_DEPRECATED
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = ExecuTorch
SOURCEDIR     = source
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

#html:
#	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
