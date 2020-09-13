#pragma once

#include <string>
#include <cassert>
#include "MersenneTwister.h"


class ArgParser
{
public:
	ArgParser() { DefaultValues(); }
	ArgParser(int argc, char* argv[]) {
		DefaultValues();
		for (int i = 1; i < argc; i++) {
			if (argv[i] == std::string("-input")) {
				i++; assert(i < argc);
				input_file = argv[i];
			}
			else if (argv[i] == std::string("-size")) {
				i++; assert(i < argc);
				width = height = atoi(argv[i]);
			}
			else if (argv[i] == std::string("-wireframe")) {
				i++; assert(i < argc);
				wireframe = true;
			}
			else if (argv[i] == std::string("-gouraud")) {
				i++; assert(i < argc);
				gouraud = true;
			}
			else {
				printf("whoops error with command line argument %d: '%s'\n", i, argv[i]);
				assert(0);
			}
		}
	}

	void DefaultValues() {
		width = 500;
		height = 500;
		wireframe = false;
		gouraud = false;
	}

	std::string input_file;
	int width;
	int height;
	bool wireframe;
	bool gouraud;
	MTRand mtrand;
};