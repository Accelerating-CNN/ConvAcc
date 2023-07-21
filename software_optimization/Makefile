OBJDIR=build
SRCDIR=src
UTILDIR=utils

USRCS=$(wildcard $(UTILDIR)/*.cpp)
OBJS=$(patsubst $(UTILDIR)/%.cpp,$(OBJDIR)/%.o,$(USRCS))

SRCS=$(wildcard $(SRCDIR)/*.cpp)
OBJS += $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCS))



# LDFLAGS = -lm -lprofiler

CXX=g++


FLAGS += -I$(UTILDIR) -I$(SRCDIR)  -O3


TRG=lab3.bin


$(TRG): 

$(TRG): $(OBJS)
	$(CXX) $(FLAGS) $^ -o $@ $(LDFLAGS)


$(OBJDIR)/%.o : $(SRCDIR)/%.cpp 
	$(CXX) $(FLAGS) -c $< -o $@



$(OBJDIR)/%.o : $(UTILDIR)/%.cpp
	$(CXX) $(FLAGS) -c $< -o $@ -I$(UTILDIR)





clean:
	$(RM) build/*
	$(RM) -r utils/__pycache__
	$(RM) $(TRG)
	$(RM) -r data









