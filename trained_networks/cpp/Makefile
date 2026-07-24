# Build flags for embedding Python 3 in C++.
# `python3-config --embed` is required from Python 3.8 onward to pull in
# -lpython3.x (the linker flag was split out of plain --ldflags so that
# extension modules, which don't need it, aren't forced to link it too).
# Older python3-config versions don't know --embed at all, so we fall back
# to plain --ldflags for those.
PYTHON_INCLUDES := $(shell python3-config --includes)
PYTHON_LIBS     := $(shell python3-config --ldflags --embed 2>/dev/null || python3-config --ldflags)

GCC=g++

all: library exe

library:
	$(GCC) SCYNet.cc -I. $(PYTHON_INCLUDES) $(PYTHON_LIBS) -shared -fPIC -o libSCYNet.so -std=c++11

exe:
	$(GCC) run.cpp -I. -L. -Wl,-rpath,'$$ORIGIN' -o run -l SCYNet

run: .PHONY
	LD_LIBRARY_PATH=. ./run

.PHONY:
