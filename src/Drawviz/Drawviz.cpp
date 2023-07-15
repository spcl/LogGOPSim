/*
 * Copyright (c) 2009 The Trustees of Indiana University and Indiana
 *                    University Research and Technology
 *                    Corporation.  All rights reserved.
 *
 * Author(s): Torsten Hoefler <htor@cs.indiana.edu>
 *            Timo Schneider <timoschn@cs.indiana.edu>
 *
 */

#include <stdlib.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <boost/regex.hpp>

#include "TimelineDrawing.hpp"
#include "cmdline.h"


int main(int argc, char **argv) {

	gengetopt_args_info args_info;
        
    if (cmdline_parser(argc, argv, &args_info) != 0) {
    	fprintf(stderr, "Couldn't parse command line arguments!\n");
        exit(EXIT_FAILURE);
	}

	std::string line;
	std::ifstream myfile(args_info.inputfile_arg);

	int rank_num = 0;
	int maxtime = 0;
	int maxcpu = 0;
	bool interval = false;

	if (args_info.endtime_arg > 0) {
		interval = true;
	}

	if (myfile.is_open()) {
		
		TimelineDrawing TLViz(args_info);

		while (!myfile.eof()) {

			boost::cmatch matches;
			
			getline (myfile,line);
			
			boost::regex ranknum("numranks (\\d+);");
			boost::regex osend("osend (\\d+) (\\d+) (\\d+) (\\d+) (\\d+(?:.\\d+)?) (\\d+(?:.\\d+)?) (\\d+(?:.\\d+)?);");
			boost::regex orecv("orecv (\\d+) (\\d+) (\\d+) (\\d+) (\\d+(?:.\\d+)?) (\\d+(?:.\\d+)?) (\\d+(?:.\\d+)?);");
			boost::regex loclop("loclop (\\d+) (\\d+) (\\d+) (\\d+) (\\d+(?:.\\d+)?) (\\d+(?:.\\d+)?) (\\d+(?:.\\d+)?);");
			boost::regex noise("noise (\\d+) (\\d+) (\\d+) (\\d+) (\\d+(?:.\\d+)?) (\\d+(?:.\\d+)?) (\\d+(?:.\\d+)?);");
			boost::regex transmission("transmission (\\d+) (\\d+) (\\d+) (\\d+) (\\d+) (\\d+) (\\d+(?:.\\d+)?) (\\d+(?:.\\d+)?) (\\d+(?:.\\d+)?);");
			boost::regex whitespace("\\w*");

			if (boost::regex_match(line.c_str(), matches, osend)) {
				
				std::string ranks = matches[1];
				std::string cpus = matches[2];
				std::string starts = matches[3];
				std::string ends = matches[4];
				std::string reds = matches[5];
				std::string greens = matches[6];
				std::string blues = matches[7];
				
				if ((interval==false) or ((atoi(starts.c_str()) >= args_info.starttime_arg) && (atoi(ends.c_str()) < args_info.endtime_arg))) {
					TLViz.add_osend(atoi(ranks.c_str()), atoi(starts.c_str())-args_info.starttime_arg, atoi(ends.c_str())-args_info.starttime_arg, atoi(cpus.c_str()),
					atof(reds.c_str()), atof(greens.c_str()), atof(blues.c_str()) );
					if (maxtime < atoi(ends.c_str())) maxtime = atoi(ends.c_str());
					if (maxcpu < atoi(cpus.c_str())) maxcpu = atoi(cpus.c_str());
				}	
			}

			else if (boost::regex_match(line.c_str(), matches, orecv)) {
				
				std::string ranks = matches[1];
				std::string cpus = matches[2];
				std::string starts = matches[3];
				std::string ends = matches[4];
				std::string reds = matches[5];
				std::string greens = matches[6];
				std::string blues = matches[7];
				
				if ((interval==false) or ((atoi(starts.c_str()) >= args_info.starttime_arg) && (atoi(ends.c_str()) < args_info.endtime_arg))) {
					TLViz.add_orecv(atoi(ranks.c_str()), atoi(starts.c_str())-args_info.starttime_arg, atoi(ends.c_str())-args_info.starttime_arg, atoi(cpus.c_str()),
					atof(reds.c_str()), atof(greens.c_str()), atof(blues.c_str()) );
					if (maxtime < atoi(ends.c_str())) maxtime = atoi(ends.c_str());
					if (maxcpu < atoi(cpus.c_str())) maxcpu = atoi(cpus.c_str());
				}
			}

			else if (boost::regex_match(line.c_str(), matches, loclop)) {

				std::string ranks = matches[1];
				std::string cpus = matches[2];
				std::string starts = matches[3];
				std::string ends = matches[4];
				std::string reds = matches[5];
				std::string greens = matches[6];
				std::string blues = matches[7];

				if ((interval==false) or ((atoi(starts.c_str()) >= args_info.starttime_arg) && (atoi(ends.c_str()) < args_info.endtime_arg))) {
					TLViz.add_loclop(atoi(ranks.c_str()), atoi(starts.c_str())-args_info.starttime_arg , atoi(ends.c_str())-args_info.starttime_arg, atoi(cpus.c_str()),
					atof(reds.c_str()), atof(greens.c_str()), atof(blues.c_str()) );
					if (maxtime < atoi(ends.c_str())) maxtime = atoi(ends.c_str());
					if (maxcpu < atoi(cpus.c_str())) maxcpu = atoi(cpus.c_str());
				}
			}

			else if (boost::regex_match(line.c_str(), matches, noise)) {

				std::string ranks = matches[1];
				std::string cpus = matches[2];
				std::string starts = matches[3];
				std::string ends = matches[4];
				std::string reds = matches[5];
				std::string greens = matches[6];
				std::string blues = matches[7];

				if ((interval==false) or ((atoi(starts.c_str()) >= args_info.starttime_arg) && (atoi(ends.c_str()) < args_info.endtime_arg))) {
					TLViz.add_noise(atoi(ranks.c_str()), atoi(starts.c_str())-args_info.starttime_arg , atoi(ends.c_str())-args_info.starttime_arg, atoi(cpus.c_str()),
					atof(reds.c_str()), atof(greens.c_str()), atof(blues.c_str()) );
					if (maxtime < atoi(ends.c_str())) maxtime = atoi(ends.c_str());
					if (maxcpu < atoi(cpus.c_str())) maxcpu = atoi(cpus.c_str());
				}
			}
			
			else if (boost::regex_match(line.c_str(), matches, transmission)) {
				
				std::string src = matches[1];
				std::string dest = matches[2];
				std::string start = matches[3];
				std::string end = matches[4];
				std::string size = matches[5];
				std::string G = matches[6];
				std::string reds = matches[7];
				std::string greens = matches[8];
				std::string blues = matches[9];

				if ((interval==false) or ((atoi(start.c_str()) >= args_info.starttime_arg) && (atoi(end.c_str()) < args_info.endtime_arg))) {
					TLViz.add_transmission(atoi(src.c_str()), atoi(dest.c_str()), atoi(start.c_str()) - args_info.starttime_arg,
					                       atoi(end.c_str()) - args_info.starttime_arg, atoi(size.c_str()), atoi(G.c_str()),
										   atof(reds.c_str()), atof(greens.c_str()), atof(blues.c_str()) );

					int endtime = atoi(end.c_str())+atoi(G.c_str())*atoi(size.c_str());
					if (maxtime < endtime ) maxtime = endtime;
				}
			}
			else if (boost::regex_match(line.c_str(), matches, ranknum)) {
				std::string ranknum = matches[1];
				if (atoi(ranknum.c_str()) > rank_num) rank_num = atoi(ranknum.c_str());
			}
			else if (boost::regex_match(line.c_str(), matches, whitespace)) {
			}
			else {
				std::cout << "Unamtched line: [" << line << "]" << std::endl;	
			} 

		}
		myfile.close();

		TLViz.init_graph(rank_num, maxcpu+1, 800, 800, args_info.outputfile_arg);
		TLViz.draw_ranklines();
		maxtime -= args_info.starttime_arg;
		TLViz.draw_everything(maxtime);
		TLViz.close_graph();

	}
	else {
		fprintf(stderr, "Unable to open file with starttimes (%s)\n", args_info.inputfile_arg);
		exit(EXIT_FAILURE);
	}
	 
	exit(EXIT_SUCCESS);
}


