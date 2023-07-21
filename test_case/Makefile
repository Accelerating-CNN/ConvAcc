OBJDIR=build
SRCDIR=src
UTILDIR=utils

USRCS=$(wildcard $(UTILDIR)/*.cpp)
OBJS=$(patsubst $(UTILDIR)/%.cpp,$(OBJDIR)/%.o,$(USRCS))

SRCS=$(wildcard $(SRCDIR)/*.cpp)
OBJS += $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCS))




FLAGS += -I$(UTILDIR) -I$(SRCDIR) 


TRG=class


$(TRG): $(OBJS)
	$(CXX) $(FLAGS)  $^ -o $@ 


$(OBJDIR)/%.o : $(SRCDIR)/%.cpp 
	$(CXX) $(FLAGS) -c $< -o $@ 



$(OBJDIR)/%.o : $(UTILDIR)/%.cpp $(UTILDIR)/%.h
	$(CXX) $(FLAGS) -c $< -o $@ 


clean:
	$(RM) build/*
	$(RM) -r utils/__pycache__
	$(RM) $(TRG)
	$(RM) -r data






