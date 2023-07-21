OBJDIR=build
SRCDIR=src
UTILDIR=utils

USRCS=$(wildcard $(UTILDIR)/*.cpp)
OBJS=$(patsubst $(UTILDIR)/%.cpp,$(OBJDIR)/%.o,$(USRCS))

SRCS=$(wildcard $(SRCDIR)/*.cpp)
OBJS += $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCS))


CXX=g++


FLAGS += -I$(UTILDIR) -I$(SRCDIR) -g -O3


TRG=class.bin


$(TRG): $(OBJS)
	$(CXX) $(FLAGS)  $^ -o $@ -lpynq -lcma -lpthread 


$(OBJDIR)/%.o : $(SRCDIR)/%.cpp 
	$(CXX) $(FLAGS) -c $< -o $@ -lpynq -lcma -lpthread




$(OBJDIR)/%.o : $(UTILDIR)/%.cpp 
	$(CXX) $(FLAGS) -c $< -o $@ -lypnq -lcma -lpthread


clean:
	$(RM) build/*
	$(RM) -r utils/__pycache__
	$(RM) $(TRG)
	$(RM) -r data
	$(RM) image.tensor






