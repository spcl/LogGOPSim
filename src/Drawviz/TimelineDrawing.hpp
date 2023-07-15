/*
 * Copyright (c) 2009 The Trustees of Indiana University and Indiana
 *                    University Research and Technology
 *                    Corporation.  All rights reserved.
 *
 * Author(s): Torsten Hoefler <htor@cs.indiana.edu>
 *            Timo Schneider <timoschn@cs.indiana.edu>
 *
 */

#include <string>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <libps/pslib.h>
#include "cmdline.h"

class overh {
	public:
		int type; // 1 = osend, 2 = orecv
		int rank;
		int cpu;
		int start;
		int end;
		float r;
		float g;
		float b;
};

class trans {
	public:
		int source;
		int dest;
		int starttime;
		int endtime;
		int size;
		int G;		
		int r;		
		int g;		
		int b;		
};

class TimelineDrawing {

	private:
	gengetopt_args_info args_info;

	PSDoc *psdoc;
	int psfont;
	int fontsize;

	int numranks;
	double ranksep;
	int numcpus;
	double cpusep;
	double timesep;

	int width;
	int height;
	int leftmargin;

	std::string content;

	std::vector<overh> overheads;
	std::vector<trans> transmissions;

	void calc_arrowhead_coords(int sx, int sy, int dx, int dy, int *x1, int *y1, int *x2, int *y2);
	void add_ranknum(int);	
	public:

  TimelineDrawing(gengetopt_args_info _args_info) : args_info(_args_info) {};

	void init_graph(int numranks, int numcpus, int width, int height, std::string filename);
	void close_graph();
	void draw_everything(int maxtime);
	
	void draw_ranklines();
	void draw_osend(int rank, int cpu, int start, int end, float r, float g, float b);
	void draw_orecv(int rank, int cpu, int start, int end, float r, float g, float b);
	void draw_transmission(int source, int dest, int starttime, int endtime, int size, int G, float r, float g, float b);
	void draw_loclop(int rank, int cpu, int start, int end, float r, float g, float b);
	void draw_noise(int rank, int cpu, int start, int end, float r, float g, float b);
	void draw_seperator(int rank, int cpu, int pos);
	
	void add_osend(int rank, int start, int end, int cpu, float r, float g, float b);
	void add_orecv(int rank, int start, int end, int cpu, float r, float g, float b);
	void add_transmission(int source, int dest, int starttime, int endtime, int size, int G, float r, float g, float b);
	void add_loclop(int rank, int start, int end, int cpu, float r, float g, float b);
	void add_noise(int rank, int start, int end, int cpu, float r, float g, float b);
};


