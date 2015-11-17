################################################################################
# Makefile for PCP 
#
# by (CPD-Minho)
################################################################################

SHELL = /bin/sh

CUDA = cuda
CPU = cpu
CXX = g++
LD  = g++

CUDAXX = nvcc
CUDALD = nvcc

BIN_CUDA = cuda_bin
BIN_CPU = cpu_bin

CXXFLAGS   = -O3 -Wall -Wextra  -fopenmp
CUDAFLAGS = -O3

SRC_DIR = src
BIN_DIR = bin
BUILD_DIR = build

SRC = $(wildcard $(SRC_DIR)/*.cu)
OBJ = $(patsubst src/*.cu,build/*.o,$(SRC))
DEPS = $(patsubst build/*.o,build/*.d,$(OBJ))

vpath %.cu $(SRC_DIR)


################################################################################
# Rules
################################################################################

.DEFAULT_GOAL = all

$(BUILD_DIR)/$(CUDA).d: $(SRC_DIR)/$(CUDA).cu
	$(CUDAXX) -M $(CUDAFLAGS) $(INCLUDES) $< -o $@

$(BUILD_DIR)/$(CUDA).o: $(SRC_DIR)/$(CUDA).cu
	$(CUDAXX) -c $(CUDAFLAGS) $(INCLUDES) $< -o $@

$(BUILD_DIR)/$(CPU).d: $(SRC_DIR)/$(CPU).cpp
	$(CXX) -M $(CXXFLAGS) $(INCLUDES) $< -o $@

$(BUILD_DIR)/$(CPU).o: $(SRC_DIR)/$(CPU).cpp
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) $< -o $@

$(BIN_DIR)/$(BIN_CUDA): $(BUILD_DIR)/$(CUDA).o $(BUILD_DIR)/$(CUDA).d 
	$(CUDAXX) $(CUDAFLAGS) $(INCLUDES) -o $@ $(BUILD_DIR)/$(CUDA).o 

$(BIN_DIR)/$(BIN_CPU): $(BUILD_DIR)/$(CPU).o $(BUILD_DIR)/$(CPU).d 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(BUILD_DIR)/$(CPU).o 

checkdirs:
	@mkdir -p build 
	@mkdir -p src
	@mkdir -p bin

all: checkdirs  $(BIN_DIR)/$(BIN_CUDA) $(BIN_DIR)/$(BIN_CPU) 

clean:
	rm -f $(BUILD_DIR)/* $(BIN_DIR)/* 	
