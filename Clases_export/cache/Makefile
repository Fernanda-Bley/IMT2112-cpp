CXX      := g++
CXXFLAGS ?= -O2 -std=c++14
SRC      := src
BINDIR   ?= bin

KERNELS := dot \
           matvec_row matvec_col \
           matmul_ijk matmul_ikj matmul_jki

.PHONY: all clean

all: $(addprefix $(BINDIR)/, $(KERNELS))

$(BINDIR)/%: $(SRC)/main_%.cpp $(SRC)/common.hpp | $(BINDIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BINDIR):
	mkdir -p $(BINDIR)

clean:
	rm -rf bin
