#
# $File: Makefile
# $Date: Wed Dec 11 18:57:54 2013 +0800
#
# A single output portable Makefile for
# simple c++ project

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
LIB_DIR = lib
TARGET = gmm

CXX = g++
#CXX = clang++

BIN_TARGET = $(BIN_DIR)/$(TARGET)
PROF_FILE = $(BIN_TARGET).prof

INCLUDE_DIR = -I $(SRC_DIR)
#LDFLAGS = -L /home/zhanpeng/.local/lib -lGClasses
#LDFLAGS += -lprofiler
#DEFINES += -D__DEBUG
#DEFINES += -D__DEBUG_CHECK


CXXFLAGS += -O3
# CXXFLAGS += -g -O0
# CXXFLAGS += -pg
CXXFLAGS += -fomit-frame-pointer -msse2 -mfpmath=sse -ffast-math -lm

CXXFLAGS += #$(DEFINES)
CXXFLAGS += -std=c++11
#CXXFLAGS += -ansi
CXXFLAGS += -Wall -Wextra
CXXFLAGS += $(INCLUDE_DIR)
CXXFLAGS += $(LDFLAGS)
#CXXFLAGS += $(shell pkg-config --libs --cflags opencv)
#CXXFLAGS += -pthread
CXXFLAGS += -lpthread
#CXXFLAGS += -fopenmp

CXXFLAGS += -fPIC

#CC = /usr/share/clang/scan-build/ccc-analyzer
#CXX = /usr/share/clang/scan-build/c++-analyzer
CXXSOURCES = $(shell find $(SRC_DIR)/ -name "*.cc")
OBJS = $(addprefix $(OBJ_DIR)/,$(CXXSOURCES:.cc=.o))
DEPFILES = $(OBJS:.o=.d)

.PHONY: all clean run rebuild gdb

all: $(BIN_TARGET) $(LIB_DIR)/pygmm.so

$(LIB_DIR)/pygmm.so: $(OBJS) $(LIB_DIR)
	g++ -shared $(OBJS) -o $(LIB_DIR)/pygmm.so $(CXXFLAGS)

$(LIB_DIR)/pygmm.o: $(OBJ_DIR)/$(SRC_DIR)/pygmm.o $(LIB_DIR)
	cp $(OBJ_DIR)/$(SRC_DIR)/pygmm.o $(LIB_DIR)/pygmm.o

$(LIB_DIR):
	mkdir $(LIB_DIR)

$(OBJ_DIR)/%.o: %.cc
	@echo "[cc] $< ..."
	@$(CXX) -c $< $(CXXFLAGS) -o $@

$(OBJ_DIR)/%.d: %.cc
	@mkdir -pv $(dir $@)
	@echo "[dep] $< ..."
	@$(CXX) $(INCLUDE_DIR) $(CXXFLAGS) -MM -MT "$(OBJ_DIR)/$(<:.cc=.o) $(OBJ_DIR)/$(<:.cc=.d)" "$<" > "$@"

sinclude $(DEPFILES)

$(BIN_TARGET): $(OBJS)
	@echo "[link] $< ..."
	@mkdir -p $(BIN_DIR)
	@$(CXX) $(OBJS) -o $@ $(LDFLAGS) $(CXXFLAGS)
	@echo have a nice day!

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(LIB_DIR)

run: $(BIN_TARGET)
	./$(BIN_TARGET)

rebuild:
	+@make clean
	+@make

gdb: $(BIN_TARGET)
	gdb ./$(BIN_TARGET)

run-prof $(PROF_FILE): $(BIN_TARGET)
	CPUPROFILE=$(PROF_FILE) CPUPROFILE_FREQUENCY=1000 ./$(BIN_TARGET)

show-prof: $(PROF_FILE)
	google-pprof --text $(BIN_TARGET) $(PROF_FILE) | tee prof.txt

