#ifndef _STOPWATCH_H
#define _STOPWATCH_H
#pragma once
#include <time.h>
#include <iostream>

class Stopwatch
{
public:
	Stopwatch() : start(clock()) {}
	~Stopwatch() = default;

	double elaspedTime() const { return static_cast<double>(clock() - start) / CLOCKS_PER_SEC; }
	void reset() { start = clock(); }

private:
	clock_t start;
};

std::ostream & operator<< (std::ostream & os, const Stopwatch & timer)
{
	os << timer.elaspedTime() << " seconds";
	return os;
}

#endif