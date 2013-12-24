CXX        = g++
CXXFLAGS   = -std=c++11 -Wall -Wextra -O2
LDFLAGS    = -lpthread
TARGET     = threadpool-example


all: $(TARGET)


$(TARGET): main.o Threadpool.hpp
	$(CXX) $(LDFLAGS) $(CXXFLAGS) main.o -o $(TARGET)

main.o: main.cpp Threadpool.hpp
	$(CXX) $(CXXFLAGS) -c $<


clean:
	rm -f *.o $(TARGET)
