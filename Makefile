current_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

PREFIX ?= /usr/local
DESTDIR ?=

ifeq ($(OS),Windows_NT)
    dllend := .dll
	exeend := .exe
    fpiccuda :=
    fpicamd :=
    fpicclang := 
    plugin_install_path := $(APPDATA)\VapourSynth\plugins64
    exe_install_path := $(ProgramFiles)\FFVship.exe
    ffvshiplibheader := -I include -lz_imp -lz -lffms2
else
    dllend := .so
	exeend :=
    fpiccuda := -Xcompiler -fPIC
    fpicamd := -fPIC
    fpicclang := -fPIC
    plugin_install_path := $(DESTDIR)$(PREFIX)/lib/vapoursynth
    exe_install_path := $(DESTDIR)$(PREFIX)/bin
    ffvshiplibheader := $(shell pkg-config --libs ffms2 zimg)
endif

.FORCE:

buildFFVSHIP: src/ffmpegmain.cpp .FORCE
	hipcc src/ffmpegmain.cpp -std=c++17 -I "$(current_dir)include" --offload-arch=native -Wno-unused-result -Wno-ignored-attributes $(ffvshiplibheader) -o FFVship$(exeend)

buildFFVSHIPcuda: src/ffmpegmain.cpp .FORCE
	nvcc -x cu src/ffmpegmain.cpp -std=c++17 -I "$(current_dir)include" -arch=native $(subst -pthread,-Xcompiler="-pthread",$(ffvshiplibheader)) -o FFVship$(exeend)

buildFFVSHIPcudaclang: src/ffmpegmain.cpp .FORCE
	clang++ -x cuda src/ffmpegmain.cpp -std=c++17 --cuda-gpu-arch=native -O3 -fcuda-approx-transcendentals -L/usr/local/cuda/lib64 -lcudart $(ffvshiplibheader) -o FFVship$(exeend)

buildFFVSHIPall: src/ffmpegmain.cpp .FORCE
	hipcc src/ffmpegmain.cpp -std=c++17 -I "$(current_dir)include" --offload-arch=gfx1100,gfx1101,gfx1102,gfx1103,gfx1030,gfx1031,gfx1032,gfx906,gfx801,gfx802,gfx803 -Wno-unused-result -Wno-ignored-attributes $(ffvshiplibheader) -o FFVship$(exeend)

buildFFVSHIPcudaall: src/ffmpegmain.cpp .FORCE
	nvcc -x cu src/ffmpegmain.cpp -std=c++17 -I "$(current_dir)include" -arch=all $(subst -pthread,-Xcompiler="-pthread",$(ffvshiplibheader)) -o FFVship$(exeend)

build: src/vapoursynthPlugin.cpp .FORCE
	hipcc src/vapoursynthPlugin.cpp -std=c++17 -I "$(current_dir)include" --offload-arch=native -I "$(current_dir)include" -Wno-unused-result -Wno-ignored-attributes -shared $(fpicamd) -o "$(current_dir)vship$(dllend)"

buildcuda: src/vapoursynthPlugin.cpp .FORCE
	nvcc -x cu src/vapoursynthPlugin.cpp -std=c++17 -I "$(current_dir)include" -arch=native -I "$(current_dir)include" -shared $(fpiccuda) -o "$(current_dir)vship$(dllend)"

buildcudaclang: src/vapoursynthPlugin.cpp .FORCE
	clang++ -x cuda src/vapoursynthPlugin.cpp -std=c++17 --cuda-gpu-arch=native -O3 -fcuda-approx-transcendentals -I "$(current_dir)include" -shared $(fpicclang) -o "$(current_dir)vship$(dllend)"

buildcudaall: src/vapoursynthPlugin.cpp .FORCE
	nvcc -x cu src/vapoursynthPlugin.cpp -std=c++17 -arch=all -I "$(current_dir)include" -shared $(fpiccuda) -o "$(current_dir)vship$(dllend)"

buildall: src/vapoursynthPlugin.cpp .FORCE
	hipcc src/vapoursynthPlugin.cpp -std=c++17 --offload-arch=gfx1100,gfx1101,gfx1102,gfx1103,gfx1030,gfx1031,gfx1032,gfx906,gfx801,gfx802,gfx803 -I "$(current_dir)include" -Wno-unused-result -Wno-ignored-attributes -shared $(fpicamd) -o "$(current_dir)vship$(dllend)"

ifeq ($(OS),Windows_NT)
install:
	if exist "$(current_dir)vship$(dllend)" copy "$(current_dir)vship$(dllend)" "$(plugin_install_path)"
else
install:
	@if [ -f "$(current_dir)vship$(dllend)" ]; then \
		install -d "$(plugin_install_path)"; \
		install -m755 "$(current_dir)vship$(dllend)" "$(plugin_install_path)/vship$(dllend)"; \
	fi
	@if [ -f "FFVship" ]; then \
		install -d "$(exe_install_path)"; \
		install -m755 FFVship "$(exe_install_path)/FFVship"; \
	fi
endif

test: .FORCE build
	vspipe ./test/vsscript.vpy .
