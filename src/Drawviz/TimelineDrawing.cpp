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
#include <sstream>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <libps/pslib.h>

#include "TimelineDrawing.hpp"


void TimelineDrawing::init_graph(int numranks, int numcpus, int width = 800, int height = 800, std::string filename = "timeline.ps") {

	this->numranks = numranks;
	this->numcpus = numcpus;
	this->width = width;
	this->height = height;

	this->ranksep = height/(numranks+2);
	this->cpusep = (ranksep*0.75) / 5; // this means we assume 4 cpus at max
	this->timesep = width/100;
	this->fontsize = 10;
	this->leftmargin = 50;

	PS_boot();
	this->psdoc = PS_new();
	PS_open_file(this->psdoc, filename.c_str());
	PS_begin_page(this->psdoc, (this->numranks+2)*this->ranksep, (this->numranks+2)*this->ranksep);
	this->psfont = PS_findfont(this->psdoc, "Helvetica", "", 0);
	PS_setfont(psdoc, psfont, this->fontsize);
	
}

void TimelineDrawing::close_graph() {

	PS_end_page(this->psdoc);
	PS_close(this->psdoc);
	PS_delete(this->psdoc);
	PS_shutdown();
}

void TimelineDrawing::draw_everything(int maxtime) {

	this->timesep = ((double) (this->width - (this->leftmargin * 2))) / (double) maxtime;
	
	for (unsigned int i=0; i<this->overheads.size(); i++) {
		if (this->overheads.at(i).type == 1) {
			draw_osend(this->overheads.at(i).rank,
			           this->overheads.at(i).cpu,			
			           this->overheads.at(i).start,
			           this->overheads.at(i).end,
			           this->overheads.at(i).r,
			           this->overheads.at(i).g,
			           this->overheads.at(i).b);
		}
		if (this->overheads.at(i).type == 2) {
			draw_orecv(this->overheads.at(i).rank,
			           this->overheads.at(i).cpu,
			           this->overheads.at(i).start,
			           this->overheads.at(i).end,
			           this->overheads.at(i).r,
			           this->overheads.at(i).g,
			           this->overheads.at(i).b
					  );

		}
		if (this->overheads.at(i).type == 3) {
			draw_loclop(this->overheads.at(i).rank,
			            this->overheads.at(i).cpu,
			            this->overheads.at(i).start,
			            this->overheads.at(i).end,
			            this->overheads.at(i).r,
			            this->overheads.at(i).g,
			            this->overheads.at(i).b
					   );

		}
		if (this->overheads.at(i).type == 4) {
			draw_noise(this->overheads.at(i).rank,
			            this->overheads.at(i).cpu,
			            this->overheads.at(i).start,
			            this->overheads.at(i).end,
			            this->overheads.at(i).r,
			            this->overheads.at(i).g,
			            this->overheads.at(i).b
					   );
		}

	}
	for (unsigned int i=0; i<this->transmissions.size(); i++) {
		draw_transmission(this->transmissions.at(i).source,
		                  this->transmissions.at(i).dest,
						  this->transmissions.at(i).starttime,
						  this->transmissions.at(i).endtime,
						  this->transmissions.at(i).size,
						  this->transmissions.at(i).G,
						  this->transmissions.at(i).r,
						  this->transmissions.at(i).g,
						  this->transmissions.at(i).b
						  );
	}
}

void TimelineDrawing::draw_ranklines() {

	for (int i=0; i<numranks; i++) {
		PS_setlinewidth(psdoc, 0.2);
		PS_moveto(psdoc, this->leftmargin, (i+2)*ranksep);
		PS_lineto(psdoc, this->width - this->leftmargin , (i+2)*ranksep);
		PS_stroke(psdoc);
		char textbuffer[128];
		snprintf(textbuffer, 128, "Rank %i", i);
		PS_setfont(psdoc, this->psfont, this->fontsize);
		PS_show_xy(psdoc, textbuffer, 5, (i+2)*ranksep);
		for (int j=1; j<numcpus; j++) {
			PS_setfont(psdoc, this->psfont, this->fontsize/1.75);
			PS_setlinewidth(psdoc, 0.05);
			PS_moveto(psdoc, this->leftmargin, (i+2)*ranksep - j*cpusep);
			PS_lineto(psdoc, this->width - this->leftmargin , (i+2)*ranksep - j*cpusep );
			PS_stroke(psdoc);
			PS_setlinewidth(psdoc, 0.2);
			snprintf(textbuffer, 128, "CPU %i", j);
			PS_show_xy(psdoc, textbuffer, 7, (i+2)*ranksep - j*cpusep);	
		}
	}

	PS_setfont(psdoc, this->psfont, this->fontsize);
	PS_show_xy(psdoc, "Time", width * 0.5, ranksep*0.3);

}

void TimelineDrawing::draw_seperator(int rank, int cpu, int pos) {

	PS_setlinewidth(psdoc, 0.1);
	PS_moveto(psdoc,
	          this->leftmargin + pos * this->timesep, 
			  (rank+2) * this->ranksep - cpu * this->cpusep - 3 );
	PS_lineto(psdoc,
	          this->leftmargin + pos * this->timesep, 
			  (rank+2) * this->ranksep - cpu * this->cpusep + 3 );
	PS_stroke(psdoc);
}

void TimelineDrawing::draw_osend(int rank, int cpu, int start, int end, float r, float g, float b) {

	PS_setcolor(psdoc, "stroke", "rgb", r, g, b, 0.0);
	PS_setlinewidth(psdoc, args_info.linethickness_arg+1.0);
	PS_moveto(psdoc,
	          this->leftmargin + start * this->timesep, 
			  (rank+2)*this->ranksep - cpu*this->cpusep);
	PS_lineto(psdoc,
	          this->leftmargin + end * this->timesep,
			  (rank+2)*this->ranksep - cpu*this->cpusep);
	PS_stroke(psdoc);

	this->draw_seperator(rank, cpu, start);
	this->draw_seperator(rank, cpu, end);

	if (args_info.descrtext_given) {
		PS_setfont(psdoc, this->psfont, this->fontsize/2);
		int xpos = this->leftmargin + (((end-start)/2 + start) * this->timesep);
		xpos -= (PS_stringwidth(psdoc, "o", this->psfont, this->fontsize/2) / 2);
		PS_show_xy(psdoc, "o", xpos, 
		           (rank+2)*ranksep - cpu*cpusep + ranksep * 0.1 );

		xpos = this->leftmargin + (((end-start)/2 + start) * this->timesep);
		xpos -= (PS_stringwidth(psdoc, "send", this->psfont, this->fontsize/2) / 2);
		PS_show_xy(psdoc, "send", xpos, 
		           (rank+2)*ranksep - cpu*cpusep - ranksep * 0.1 );
	}
}

void TimelineDrawing::draw_orecv(int rank, int cpu, int start, int end, float r, float g, float b) {

	PS_setcolor(psdoc, "stroke", "rgb", r, g, b, 0.0);
	PS_setlinewidth(psdoc, args_info.linethickness_arg+1.0);
	PS_moveto(psdoc,
	          this->leftmargin + start * this->timesep, 
			  (rank+2)*this->ranksep - cpu*this->cpusep);
	PS_lineto(psdoc,
	          this->leftmargin + end * this->timesep,
			  (rank+2)*this->ranksep - cpu*this->cpusep);
	PS_stroke(psdoc);

	this->draw_seperator(rank, cpu, start);
	this->draw_seperator(rank, cpu, end);

	if (args_info.descrtext_given) {
		PS_setfont(psdoc, this->psfont, this->fontsize/2);
		int xpos = this->leftmargin + (((end-start)/2 + start) * this->timesep);
		xpos -= (PS_stringwidth(psdoc, "o", this->psfont, this->fontsize/2) / 2);
		PS_show_xy(psdoc, "o", xpos, 
		           (rank+2)*ranksep - cpu*cpusep + ranksep * 0.1 );

		xpos = this->leftmargin + (((end-start)/2 + start) * this->timesep);
		xpos -= (PS_stringwidth(psdoc, "recv", this->psfont, this->fontsize/2) / 2);
		PS_show_xy(psdoc, "recv", xpos, 
		           (rank+2)*ranksep - cpu*cpusep - ranksep * 0.1 );
	}

}

void TimelineDrawing::draw_loclop(int rank, int cpu, int start, int end, float r, float g, float b) {

	PS_setcolor(psdoc, "stroke", "rgb", r, g, b, 0.0);

	PS_setlinewidth(psdoc, args_info.linethickness_arg+1.0);
	PS_moveto(psdoc,
	          this->leftmargin + start * this->timesep, 
			  (rank+2)*this->ranksep - cpu*this->cpusep);
	PS_lineto(psdoc,
	          this->leftmargin + end * this->timesep,
			  (rank+2)*this->ranksep - cpu*this->cpusep);
	PS_stroke(psdoc);

	this->draw_seperator(rank, cpu, start);
	this->draw_seperator(rank, cpu, end);

	if (args_info.descrtext_given) {
		PS_setfont(psdoc, this->psfont, this->fontsize/2);
		int xpos = this->leftmargin + (((end-start)/2 + start) * this->timesep);
		xpos -= (PS_stringwidth(psdoc, "calc", this->psfont, this->fontsize/2) / 2);
		PS_show_xy(psdoc, "calc", xpos, 
		           (rank+2)*ranksep - cpu*cpusep - ranksep * 0.1 );
	}


}

void TimelineDrawing::draw_noise(int rank, int cpu,  int start, int end, float r, float g, float b) {

	PS_setcolor(psdoc, "stroke", "rgb", r, g, b, 0.0);

	PS_setlinewidth(psdoc, args_info.linethickness_arg+1.0);
	PS_moveto(psdoc,
	          this->leftmargin + start * this->timesep, 
			  (rank+2)*this->ranksep - cpu*this->cpusep);
	PS_lineto(psdoc,
	          this->leftmargin + end * this->timesep,
			  (rank+2)*this->ranksep - cpu*this->cpusep);
	PS_stroke(psdoc);

	this->draw_seperator(rank, cpu, start);
	this->draw_seperator(rank, cpu, end);

	if (args_info.descrtext_given) {
		PS_setfont(psdoc, this->psfont, this->fontsize/2);
		int xpos = this->leftmargin + (((end-start)/2 + start) * this->timesep);
		xpos -= (PS_stringwidth(psdoc, "calc", this->psfont, this->fontsize/2) / 2);
		PS_show_xy(psdoc, "calc", xpos, 
		           (rank+2)*ranksep - cpu*cpusep - ranksep * 0.1 );
	}

}

void TimelineDrawing::draw_transmission(int source, int dest, int starttime, int endtime, int size, int G, float r, float g, float b) {

	PS_setcolor(psdoc, "stroke", "rgb", r, g, b, 0.0);
	PS_setlinewidth(psdoc, args_info.linethickness_arg);

	for (int i = 0; i <= size-1; i++) {
		PS_setdash(psdoc, 2.0, 2.0);
		PS_moveto(psdoc, this->leftmargin + (starttime + i * G) * this->timesep, (source+2)*ranksep);
		
		// store coordinates for drawing the arrowheads
		int sx = this->leftmargin + (starttime + i * G) * this->timesep;
		int sy = (source+2)*ranksep;
		
		// the behaviour of the sim changed! oldsin: transmission "ends" with last byte,
		// newsim: transmission ends with first, so orecv can start earlier 
		int L = endtime - starttime;// - (size-1)*G;
		//assert(L > 0);
		PS_lineto(psdoc, this->leftmargin + ((starttime + i * G) + L) * this->timesep, (dest+2)*ranksep);
		
		// store coordinates for drawing the arrowheads
		int dx = this->leftmargin + ((starttime + i * G) + L) * this->timesep;
		int dy = (dest+2)*ranksep;

		PS_stroke(psdoc);

		if (args_info.arrowheads_given) {
			// draw arrowhead
			int x1, y1, x2, y2;
			calc_arrowhead_coords(sx, sy, dx, dy, &x1, &y1, &x2, &y2);
			PS_setdash(psdoc, 0.0, 0.0);
			PS_moveto(psdoc, dx, dy);
			PS_lineto(psdoc, x1, y1);
			PS_stroke(psdoc);
			PS_moveto(psdoc, dx, dy);
			PS_lineto(psdoc, x2, y2);
			PS_stroke(psdoc);
		}
	}
}

void TimelineDrawing::add_osend(int rank, int start, int end, int cpu, float r, float g, float b) {
	
	overh os;
	os.type = 1;
	os.rank = rank;
	os.cpu = cpu;
	os.start = start;
	os.end = end;
	os.r = r;
	os.g = g;
	os.b = b;
	
	this->overheads.push_back(os);

}

void TimelineDrawing::add_orecv(int rank, int start, int end, int cpu, float r, float g, float b) {
	
	overh orecv;
	orecv.type = 2;
	orecv.rank = rank;
	orecv.cpu = cpu;
	orecv.start = start;
	orecv.end = end;
	orecv.r = r;
	orecv.g = g;
	orecv.b = b;

	this->overheads.push_back(orecv);

}

void TimelineDrawing::add_loclop(int rank, int start, int end, int cpu, float r, float g, float b) {
	
	overh lop;
	lop.type = 3;
	lop.rank = rank;
	lop.cpu = cpu;
	lop.start = start;
	lop.end = end;
	lop.r = r;
	lop.g = g;
	lop.b = b;

	this->overheads.push_back(lop);

}

void TimelineDrawing::add_noise(int rank, int start, int end, int cpu, float r, float g, float b) {
	
	overh noise;
	noise.type = 4;
	noise.rank = rank;
	noise.cpu = cpu;
	noise.start = start;
	noise.end = end;
	noise.r = r;
	noise.g = g;
	noise.b = b;

	this->overheads.push_back(noise);

}

void TimelineDrawing::add_transmission(int source, int dest, int starttime, int endtime, int size, int G, float r, float g, float b) {
	
	trans tm;
	tm.source = source;
	tm.dest = dest;
	tm.starttime = starttime;
	tm.endtime = endtime;
	tm.size = size;
	tm.G = G;
	tm.r = r;
	tm.g = g;
	tm.b = b;

	this->transmissions.push_back(tm);

	std::stringstream os;
	os << "transmission " << source << " " << dest << " " << starttime << " ";
	os << endtime << " " << G << ";\n";
	this->content.append(os.str());
}

void TimelineDrawing::calc_arrowhead_coords(int sx, int sy, int dx, int dy, int *x1, int *y1, int *x2, int *y2) {

	double pi = 3.141592;
	double angle = atan2 (dy - sy, dx - sx) + pi;
	double arrowlength = 6*args_info.linethickness_arg;
	
	*x1 = dx + arrowlength * cos(angle - pi/12);
	*y1 = dy + arrowlength * sin(angle - pi/12);
	*x2 = dx + arrowlength * cos(angle + pi/12);
	*y2 = dy + arrowlength * sin(angle + pi/12);		

}

