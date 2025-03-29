# $Id $

# Copyright (c) 2007-2015, Trustees of The Leland Stanford Junior University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

LEX = flex
YACC   = bison -y
DEFINE = -DTEST_SCALE_TRAFFICMANAGER -DDEBUG_BIGRAPH
INCPATH = -I. -Iarbiters -Iallocators -Irouters -Inetworks -Ipower -Idsent
CPPFLAGS += -Wall $(DEFINE)
#CPPFLAGS += -O3
#CPPFLAGS += -g
#CFLAGS += -Wall -O3 -g
CFLAGS += -Wall
CPPFLAGS += -std=c++11
LFLAGS +=
PYBIND11_INC = `python3 -m pybind11 --includes`

OBJDIR := obj
PROG := booksim
LIB_BOOKSIM := pybooksim.so
STATIC_LIB_BOOKSIM := libbooksim.a

########## DSENT ###########
# Define the directories that will be compiled
DSENT_DIRS_TO_COMPILE := util tech io \
				model model/timing_graph \
				model/std_cells \
				model/electrical \
				model/electrical/router \
				model/optical \
				model/optical_graph \
				model/network \
				model/network/ATAC

DSENT_DIRS = $(patsubst %,dsent/%,$(DSENT_DIRS_TO_COMPILE))
DSENT_SRCS = $(foreach dir, $(DSENT_DIRS), $(wildcard $(dir)/*.cc))
DSENT_OBJS = $(DSENT_SRCS:%.cc=%.o)

DEF_FLAGS =
INCLUDE_FLAGS = -Idsent
OPT_FLAGS = -O2 -g
WARN_FLAGS = -pedantic -Wall -W #-Wextra -Werror -Wno-write-strings
CXXFLAGS = $(OPT_FLAGS) $(WARN_FLAGS) $(INCLUDE_FLAGS) $(DEF_FLAGS)

# link libraries
LD_LIBS += -lutil
LD_FLAGS += -Ldsent/libutil

# Other libraries used
LIB_UTIL = dsent/libutil/libutil.a

########### DSENT ##############

# simulator source files
CPP_SRCS = $(wildcard *.cpp) $(wildcard */*.cpp)
CPP_HDRS = $(wildcard *.hpp) $(wildcard */*.hpp)
CPP_DEPS = $(addprefix ${OBJDIR}/,$(notdir $(CPP_SRCS:.cpp=.d)))
CPP_OBJS = $(addprefix ${OBJDIR}/,$(notdir $(CPP_SRCS:.cpp=.o)))

#LIB_CPP_SRCS = $(filter-out main.cpp,$(CPP_SRCS))
LIB_CPP_SRCS = $(CPP_SRCS)
LIB_CPP_HDRS = $(CPP_HDRS)
LIB_CPP_DEPS = $(addprefix ${OBJDIR}/,$(notdir $(LIB_CPP_SRCS:.cpp=.d)))
LIB_CPP_OBJS = $(addprefix ${OBJDIR}/,$(notdir $(LIB_CPP_SRCS:.cpp=.o)))

LEX_SRCS = lex.yy.c
LEX_OBJS = ${OBJDIR}/lex.yy.o

YACC_SRCS = y.tab.c
YACC_HDRS = y.tab.h
YACC_OBJS = ${OBJDIR}/y.tab.o

OBJS := $(DSENT_OBJS) $(CPP_OBJS) $(LEX_OBJS) $(YACC_OBJS)
LIB_OBJS := $(LIB_CPP_OBJS) $(LEX_OBJS) $(YACC_OBJS) $(DSENT_OBJS)

.PHONY: clean

all: CPPFLAGS += -O3 -g
all: CFLAGS += -O3 -g
all: $(LIB_UTIL) $(PROG)

# debug
dbg: CPPFLAGS += -O0 -g
dbg: CFLAGS += -O0 -g
dbg: $(LIB_UTIL) $(PROG)

# dynamic library debug mode
libdbg: DEFINE += -DLIB_BOOKSIM
libdbg: CPPFLAGS += -O0 -g -fPIC
libdbg: CFLAGS += -O0 -g -fPIC
libdbg: CXXFLAGS += -fPIC
libdbg: ${LIB_UTIL} $(LIB_BOOKSIM)

# dynamic library
lib: DEFINE += -DLIB_BOOKSIM
lib: CPPFLAGS += -O3 -g -fPIC
lib: CFLAGS += -O3 -g -fPIC
lib: CXXFLAGS += -fPIC
lib: ${LIB_UTIL} $(LIB_BOOKSIM)

# static library
staticlib: DEFINE += -DLIB_BOOKSIM
staticlib: CPPFLAGS += -O3 -g -fPIC
staticlib: CFLAGS += -O3 -g -fPIC
staticlib: CXXFLAGS += -fPIC
staticlib: ${LIB_UTIL} $(STATIC_LIB_BOOKSIM)

$(PROG): $(OBJS) dsent/DSENT.o
	 $(CXX) $(LFLAGS) $(LD_FLAGS) $^ -o $@ $(LD_LIBS)

$(LIB_BOOKSIM): $(LIB_OBJS) dsent/DSENT.o
	 $(CXX) -shared $(INCPATH) $(PYBIND11_INC) -o $@ $^ $(LFLAGS) $(LD_FLAGS) $(LD_LIBS)

$(STATIC_LIB_BOOKSIM): $(LIB_OBJS) dsent/DSENT.o
	ar rcs $@ $^

$(LIB_UTIL):
	$(MAKE) -C dsent/libutil

dsent/DSENT.o: dsent/DSENT.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(LEX_SRCS): config.l
	$(LEX) $<

$(YACC_SRCS) $(YACC_HDRS): config.y
	$(YACC) -d $<

$(LEX_OBJS): $(LEX_SRCS) $(YACC_HDRS)
	$(CC) $(CFLAGS) -c $< -o $@

$(YACC_OBJS): $(YACC_SRCS)
	$(CC) $(CFLAGS) -c $< -o $@

.SECONDEXPANSION:

${OBJDIR}/%.o: %.cpp | $$(@D)
	$(CXX) $(CPPFLAGS) $(INCPATH) $(PYBIND11_INC) -MMD -c $< -o $@

# rules to compile networks
${OBJDIR}/%.o: networks/%.cpp
	$(CXX) $(CPPFLAGS) $(INCPATH) -c $< -o $@

# rules to compile arbiters
${OBJDIR}/%.o: arbiters/%.cpp
	$(CXX) $(CPPFLAGS) $(INCPATH) -c $< -o $@

# rules to compile allocators
${OBJDIR}/%.o: allocators/%.cpp
	$(CXX) $(CPPFLAGS) $(INCPATH) -c $< -o $@

# rules to compile routers
${OBJDIR}/%.o: routers/%.cpp
	$(CXX) $(CPPFLAGS) $(INCPATH) -c $< -o $@

# rules to compile power classes
${OBJDIR}/%.o: power/%.cpp
	$(CXX) $(CPPFLAGS) $(INCPATH) -c $< -o $@

$(OBJDIR):
	mkdir -p $@

clean:
	rm -rf $(OBJDIR)
	rm -f $(YACC_SRCS) $(YACC_HDRS)
	rm -f $(LEX_SRCS)
	rm -f $(CPP_DEPS)
	rm -f $(OBJS)
	rm -f $(LIB_CPP_DEPS)
	rm -f $(LIB_OBJS)
	rm -f $(PROG)
	rm -f $(LIB_BOOKSIM)
	rm -f $(STATIC_LIB_BOOKSIM)
	make clean -C dsent

distclean: clean
	rm -rf $(OBJDIR)
	rm -f *~ */*~
	rm -f *.o */*.o
	rm -f *.d */*.d

-include $(CPP_DEPS)
