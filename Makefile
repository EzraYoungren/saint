CC      = gcc
CFLAGS  = -Wall -Wextra -O0 -g
LIBS    = -lcblas -lm

TARGET  = saint
SRC     = main.c

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
