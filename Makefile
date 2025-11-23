current_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

PREFIX ?= /usr/local
DESTDIR ?=

HIPARCH := gfx1201,gfx1100,gfx1101,gfx1102,gfx1103,gfx1151,gfx1012,gfx1030,gfx1031,gfx1032,gfx906,gfx801,gfx802,gfx803

ifeq ($(OS),Windows_NT)
    dllend := .dll
	exeend := .exe
    fpiccuda :=
    fpicamd :=
    plugin_install_path := $(APPDATA)\VapourSynth\plugins64
    exe_install_path := $(ProgramFiles)\FFVship.exe
    ffvshiplibheader :=  -lffms2
	ffvshipincludeheader := -I include
	fatbincompressamd := 
	fatbincompresscuda := 
else
    dllend := .so
	exeend :=
    fpiccuda := -Xcompiler -fPIC
    fpicamd := -fPIC
    plugin_install_path := $(DESTDIR)$(PREFIX)/lib/vapoursynth
	lib_install_path := $(DESTDIR)$(PREFIX)/lib
    exe_install_path := $(DESTDIR)$(PREFIX)/bin
	header_install_path := $(DESTDIR)$(PREFIX)/include
    ffvshiplibheader := $(shell pkg-config --libs ffms2)
	ffvshipincludeheader := $(shell pkg-config --cflags-only-I ffms2 libavutil)
	fatbincompressamd := --offload-compress
	fatbincompresscuda := --compress-mode=balance
endif

.FORCE:

buildFFVSHIP: src/FFVship.cpp .FORCE
	hipcc src/FFVship.cpp -g -std=c++17 $(ffvshipincludeheader) --offload-arch=native -Wno-unused-result -Wno-ignored-attributes $(ffvshiplibheader) -o FFVship$(exeend)

buildFFVSHIPcuda: src/FFVship.cpp .FORCE
	nvcc -x cu src/FFVship.cpp -g -std=c++17 $(ffvshipincludeheader) -arch=native $(subst -pthread,-Xcompiler="-pthread",$(ffvshiplibheader)) -o FFVship$(exeend)

buildFFVSHIPall: src/FFVship.cpp .FORCE
	hipcc src/FFVship.cpp -g -std=c++17 $(ffvshipincludeheader) $(fatbincompressamd) --offload-arch=$(HIPARCH) -Wno-unused-result -Wno-ignored-attributes $(ffvshiplibheader) -o FFVship$(exeend)

buildFFVSHIPcudaall: src/FFVship.cpp .FORCE
	nvcc -x cu src/FFVship.cpp -g -std=c++17 $(ffvshipincludeheader) $(fatbincompresscuda) -arch=all $(subst -pthread,-Xcompiler="-pthread",$(ffvshiplibheader)) -o FFVship$(exeend)

build: src/VshipLib.cpp .FORCE
	hipcc src/VshipLib.cpp -g -std=c++17 -I "$(current_dir)include" --offload-arch=native -Wno-unused-result -Wno-ignored-attributes -shared $(fpicamd) -o "$(current_dir)libvship$(dllend)"

buildcuda: src/VshipLib.cpp .FORCE
	nvcc -x cu src/VshipLib.cpp -g -std=c++17 -I "$(current_dir)include" -arch=native -shared $(fpiccuda) -o "$(current_dir)libvship$(dllend)"

buildcudaall: src/VshipLib.cpp .FORCE
	nvcc -x cu src/VshipLib.cpp -g -std=c++17 -I "$(current_dir)include" $(fatbincompresscuda) -arch=all -shared $(fpiccuda) -o "$(current_dir)libvship$(dllend)"

buildall: src/VshipLib.cpp .FORCE
	hipcc src/VshipLib.cpp -g -std=c++17 -I "$(current_dir)include" $(fatbincompressamd) --offload-arch=$(HIPARCH) -Wno-unused-result -Wno-ignored-attributes -shared $(fpicamd) -o "$(current_dir)libvship$(dllend)"

ifeq ($(OS),Windows_NT)
install:
	if exist "$(current_dir)libvship$(dllend)" copy "$(current_dir)libvship$(dllend)" "$(plugin_install_path)"
else
install:
	@if [ -f "$(current_dir)libvship$(dllend)" ]; then \
		install -d "$(plugin_install_path)"; \
		install -d "$(header_install_path)"; \
		install -m755 "$(current_dir)libvship$(dllend)" "$(lib_install_path)/libvship$(dllend)"; \
		ln -sf "../libvship$(dllend)" "$(plugin_install_path)/libvship$(dllend)"; \
		install -m755 "$(current_dir)src/VshipAPI.h" "$(header_install_path)/VshipAPI.h"; \
		install -m755 "$(current_dir)src/VshipColor.h" "$(header_install_path)/VshipColor.h"; \
	fi
	@if [ -f "FFVship" ]; then \
		install -d "$(exe_install_path)"; \
		install -m755 FFVship "$(exe_install_path)/FFVship"; \
	fi
uninstall:
	rm -f "$(plugin_install_path)/libvship$(dllend)" "$(lib_install_path)/libvship$(dllend)" "$(header_install_path)/VshipAPI.h" "$(header_install_path)/VshipColor.h" "$(exe_install_path)/FFVship"
uninstallOld:
	rm -f "$(plugin_install_path)/vship$(dllend)" "$(lib_install_path)/vship$(dllend)" "$(header_install_path)/VshipAPI.h" "$(exe_install_path)/FFVship"
endif

test: .FORCE build
	vspipe ./test/vsscript.vpy .
