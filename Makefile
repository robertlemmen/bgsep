TARGET:=bgsep

CXXFLAGS?=-g -O2 -Wall -Wno-unknown-pragmas -std=c++17 
CXXINCFLAGS?=
LDFLAGS?=$(shell pkg-config --libs opencv)
CXX?=g++

OBJECTS=$(subst .cc,.o,$(shell ls *.cc))

$(TARGET): $(OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(CXXINCFLAGS) -c $< -o $@
	$(CXX) -MM -MT $@ $(CXXFLAGS) $(CXXINCFLAGS) -c $< > $*.d

.PHONY: clean

clean:
	rm -f *.o *.d
	rm -f $(TARGET)

-include *.d
