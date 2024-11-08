#
# Copyright (c) 2024 Cadence Design Systems, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to use this Software with Cadence processor cores only and
# not with any other processors and platforms, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

QUIET =
WARNING_AS_ERROR ?= 1
MAPFILE  = map_$(CODEC_NAME).txt
LDSCRIPT = ldscript_$(CODEC_NAME).txt
SYMFILE  = symbols_$(CODEC_NAME).txt
DETECTED_CORE?=

AR = xt-ar $(XTCORE)
OBJCOPY = xt-objcopy $(XTCORE)
CC = xt-clang $(XTCORE)
CXX = xt-clang++ $(XTCORE)
ISS = xt-run $(XTCORE)
CONFIGDIR := $(shell $(ISS) --show-config=config)
include $(CONFIGDIR)/misc/hostenv.mk
GREPARGS =
WINNUL =
IFEXIST =

has_mul16_tmp = $(shell $(GREP) $(GREPARGS)"IsaUseMul16 = 1"  "$(XTENSA_SYSTEM)$(S)$(XTENSA_CORE)-params")
has_mul32_tmp = $(shell $(GREP) $(GREPARGS)"IsaUse32bitMul = 1" "$(XTENSA_SYSTEM)$(S)$(XTENSA_CORE)-params")
has_mul16=1
has_mul32=1
ifeq (,$(has_mul16_tmp))
has_mul16=0
endif
ifeq (,$(has_mul32_tmp))
has_mul32=0
endif
CFLAGS += -Wall 
ifeq ($(WARNING_AS_ERROR),1)
	CFLAGS += -Werror
	ifneq ($(CC), xt-xcc)
	CFLAGS += -Wno-parentheses-equality
	endif
endif
ifeq "$(has_mul16)" "0"
	CFLAGS += -mno-mul16
endif
ifeq "$(has_mul32)" "0"
	CFLAGS += -mno-mul32 -mno-div32
endif
CFLAGS += -fsigned-char -fno-exceptions -mlongcalls -mcoproc -INLINE:requested -fno-zero-initialized-in-bss
CFLAGS += -mtext-section-literals 
CFLAGS += -Wsign-compare

OBJDIR = objs$(S)$(CODEC_NAME)$(DETECTED_CORE)
LIBDIR = $(ROOTDIR)$(S)lib

OBJ_LIBOBJS = $(addprefix $(OBJDIR)/,$(LIBOBJS))
OBJ_LIBOSOBJS = $(addprefix $(OBJDIR)/,$(LIBOSOBJS))

ALL_OBJS := \
  $(OBJ_LIBOBJS) \
  $(OBJ_LIBOSOBJS) \

ALL_DEPS := $(foreach dep,$(ALL_OBJS),${dep:%.o=%.d})
-include $(ALL_DEPS)

TEMPOBJ = temp.o    

LIBOBJ   = $(OBJDIR)/xa_$(CODEC_NAME)$(DETECTED_CORE).o
LIB      = xa_$(CODEC_NAME)$(DETECTED_CORE).a

CFLAGS += $(EXTRA_CFLAGS) $(EXTRA_CFLAGS2)

LIBLDFLAGS += \
    $(EXTRA_LIBLDFLAGS)

ifeq ($(DEBUG),1)
  NOSTRIP = 1
  OPT_O2 = -O0 -g 
  OPT_O3 = -O0 -g 
  OPT_OS = -O0 -g
  OPT_O0 = -O0 -g 
  CFLAGS += -DDEBUG
else
  OPT_O2 = -O2 -LNO:simd 
  OPT_O3 = -O3 -LNO:simd 
  OPT_OS = -Os 
  OPT_O0 = -O0 
  CFLAGS += -DNDEBUG=1
endif

all: $(OBJDIR) $(LIB)

install: $(LIB)
	@echo "Installing $(LIB)"
	$(QUIET) -$(MKPATH) "$(LIBDIR)"
	$(QUIET) $(CP) $(LIB) "$(LIBDIR)"

$(OBJDIR):
	$(QUIET) -$(MKPATH) $@

ifeq ($(NOSTRIP), 1)
$(LIBOBJ): $(OBJ_LIBOBJS) $(OBJ_LIBOSOBJS)  
	@echo "Linking Objects"
	$(QUIET) $(CC) -o $@ $^ \
	-Wl,-r,-Map,$(MAPFILE) --no-standard-libraries
else
$(LIBOBJ): $(OBJ_LIBOBJS) $(OBJ_LIBOSOBJS) 
	@echo "Linking Objects"
	$(QUIET) $(CC) -o $@ $^ \
	-Wl,-r,-Map,$(MAPFILE) --no-standard-libraries \
	-Wl,--retain-symbols-file,$(SYMFILE) \
	-Wl,--script,$(LDSCRIPT) $(IPA_FLAGS) $(LIBLDFLAGS)
	$(QUIET) $(OBJCOPY) --keep-global-symbols=$(SYMFILE) $@ $(TEMPOBJ)
	$(QUIET) $(OBJCOPY) --strip-unneeded $(TEMPOBJ) $@
	$(QUIET) -$(RM) $(TEMPOBJ)
endif 


$(OBJ_LIBOBJS): $(OBJDIR)/%.o: %.c
	@echo "Compiling $<"
	$(QUIET) $(CC) -o $@ $(OPT_O3) $(CFLAGS) $(INCLUDES) -c $<
	
$(OBJ_LIBOSOBJS): $(OBJDIR)/%.o: %.c
	@echo "Compiling $<"
	$(QUIET) $(CC) -o $@ $(OPT_OS) $(CFLAGS) $(INCLUDES) -c $<
		
	
	
$(LIB): %.a: $(OBJDIR)/%.o
	@echo "Creating Library $@"
	$(QUIET) $(AR) rc $@ $^

clean:
	-$(RM) xa_$(CODEC_NAME)$(DETECTED_CORE).a xgcc_$(CODEC_NAME)$(DETECTED_CORE).a $(LIBDIR)$(S)xa_$(CODEC_NAME)$(DETECTED_CORE).a $(LIBDIR)$(S)xgcc_$(CODEC_NAME)$(DETECTED_CORE).a $(MAPFILE)
	-$(RM) $(OBJDIR)$(S)*.o
	-$(RM) $(ALL_DEPS)
	-$(RM_R) $(LIBDIR)
