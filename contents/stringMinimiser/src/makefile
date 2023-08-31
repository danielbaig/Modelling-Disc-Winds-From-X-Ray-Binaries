CVERS=c++17

IDIR =../include
CC=g++
CFLAGS=-I$(IDIR)


ODIR=obj
LDIR =../lib/eigen-3.4.0/


LIBS=-lm
LDFLAGS=-I$(LDIR)

_DEPS = ExtraMaths.h Random.h Energy.h MC_minimiser.h DataManager.h Timer.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = lineMinimiser.o ExtraMaths.o Random.o Energy.o MC_minimiser.o TensorConvert.o DataManager.o Timer.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/%.o: %.cpp $(DEPS)
	$(CC) -std=$(CVERS) -g -c -o $@ $< $(CFLAGS) $(LDFLAGS)

run: $(OBJ)
	$(CC) -std=$(CVERS) -g -o $@ $(OBJ) $(CFLAGS) $(LIBS) $(LDFLAGS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ 
