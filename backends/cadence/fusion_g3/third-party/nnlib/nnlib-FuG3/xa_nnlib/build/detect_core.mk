

ISS = xt-run $(XTCORE)
CONFIGDIR := $(shell $(ISS) --show-config=config)
include $(CONFIGDIR)/misc/hostenv.mk

GREPARGS =
ifeq ($(HOSTTYPE),win)
GREPARGS = /c:
endif

ifeq ("", "$(detected_core)")

fusion_g3="0"
fusion_g3_tmp:=$(shell $(GREP) $(GREPARGS)"IsaUseFusionG = 1"  "$(XTENSA_SYSTEM)$(S)$(XTENSA_CORE)-params")

#check if the detected core is Fusion G3
    ifneq ("", "$(fusion_g3_tmp)")
        detected_core=fusion_g3
    endif

ifeq ("$(detected_core)", "fusion_g3")
    fusion_g3=1
    CFLAGS+= -DCORE_FUG3=1
else
    $(error "$(fusion_g3_tmp)" Core Not Found)
endif
endif

xclib_tmp:=$(shell $(GREP) $(GREPARGS)"SW_CLibrary = xclib"  "$(XTENSA_SYSTEM)$(S)$(XTENSA_CORE)-params")
ifneq ("", "$(xclib_tmp)")
    xclib=1
else
    xclib=0
endif

