/* This is a very simple program to generate a Newton fractal */

#include <stdio.h>
#include <fcntl.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <mpi.h>

void hsvTOrgb(double h, double s, double v, double*r, double*g, double *b);


#define WIDTH 1920
#define HEIGHT 1920
#define MASTER 0

const static double xstart = -1.0;
const static double ystart = -1.0;
const static double xend = 1.0;
const static double yend = 1.0;
#define PI 3.14159265




void new_fract(char*local_pic)
{
	int myProcessor, numprocs;
	MPI_Comm_rank(MPI_COMM_WORLD, &myProcessor);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	double xstep = (xend - xstart) / WIDTH;
	double ystep = (yend - ystart) / HEIGHT;

	int len = (WIDTH*HEIGHT) / numprocs; 
	
	int start = myProcessor*len; //start in the 1d map of the 2d pic
	int end = start + len;
	
	int id;
	
	//printf("\nmyProcessor= %d, start= %d, end= %d\n", myProcessor, start, end);

	int index = 0;//updating index in the local array 
	for (id = start; id < end; id++){
		
		int i = id / HEIGHT;
		int j = id - i*HEIGHT;

		double y = xstart + i*ystep;
		double x = ystart + j*xstep;

		double z = x;
		double zi = y;
		int converge = 0;
		int k;
		double color;
		for (k = 0; k < 60; k++) {
			//Use Newton's Method to converge to the roots of (z^3 - 1 = 0)
			double f = (z * z * z) - (3 * z * zi * zi) - 1;
			double fprime = 3 * (z * z - zi * zi);
			double fi = (3 * z * z * zi) - (zi * zi * zi);
			double fprimei = 6 * z * zi;

			//Newton's Method using complex number division
			double newz = z - (f * fprime + fi * fprimei) / (fprime * fprime + fprimei * fprimei);
			double newzi = zi - (fi * fprime - f * fprimei) / (fprime * fprime + fprimei * fprimei);
			
			if (hypot(newz - z, newzi - zi) < 0.001) {
				converge = 1;
				color = k;
				break;
			}
			else {
				z = newz;
				zi = newzi;
			}
		}
		if (!converge) {
			local_pic[index*3 + 0] = 0;
			local_pic[index*3 + 1] = 0;
			local_pic[index*3 + 2] = 0;
		}
		else{
			double r, g, b;
			r = g = b = 0;
			double h = sin((double)color / (double)60 * PI / 2);
			hsvTOrgb(h, 1, 1, &r, &g, &b);
			local_pic[index*3 + 0] = r * 255;
			local_pic[index*3 + 1] = g * 255;
			local_pic[index*3 + 2] = b * 255;
		}	
		index++;
	}
}

double wallTime() // returns time in MS as a double
{
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/*
 *  hsvTOrgb
 *  converts an hsv (hue, saturation, value) color value to rgb
 *  (red, green, blue)
 *  Created by Jon McCormack on Sat Jul 10 2004.
 * hsv and rgb values normalised from 0 - 1 
 */
void hsvTOrgb(double h, double s, double v, double*r, double*g, double *b)
{
	
	int k;
	double aa, bb, cc, f;

	if (s <= 0.0){		
		*r = *g = *b = v;
	}
	else {
		if (h >= 1.0)
			h = 0.0;
		h *= 6.0;
		k = floor(h);
		f = h - k;
		aa = v * (1.0 - s);
		bb = v * (1.0 - (s * f));
		cc = v * (1.0 - (s * (1.0 - f)));
		switch (k) {
		case 0:			
			*r = v;
			*g = cc;
			*b = aa;
			break;
		case 1:			
			*r = bb;
			*g = v;
			*b = aa;
			break;
		case 2:			
			*r = aa;
			*g = v;
			*b = cc;
			break;
		case 3:			
			*r = aa;
			*g = bb;
			*b = v;
			break;
		case 4:			
			*r = cc;
			*g = aa;
			*b = v;
			break;
		case 5:			
			*r = v;
			*g = aa;
			*b = bb;
			break;

		}
	}
}

main(int argc, char*argv[])
{

	int i, myid, numprocs;
	char*pic = NULL;//[3 * HEIGHT * WIDTH];


	//MPI Initialization
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);


	int ele_per_process = (int)(((double)(HEIGHT * WIDTH * 3)) / ((double)(numprocs)));

	//if (myid == MASTER){
	//	printf("\nnumprocs= %d, ele_per_process1= %f, ele_per_process= %d\n", numprocs, (HEIGHT * WIDTH * 3.0) / (numprocs*1.0), ele_per_process);
	//}

	if (myid == MASTER){
		//create the main pic buffer once
		pic = (char *)malloc(HEIGHT * WIDTH * 3 * sizeof(char));
		if (pic == NULL) {
			printf("\nError allocating memory in 'pic'.\n");
		}
	}

	char*local_pic = (char *)malloc(sizeof(char) * (ele_per_process));
	if (local_pic == NULL) {
		printf("\nError allocating memory in 'local_pic'.\n");
	}

	MPI_Barrier(MPI_COMM_WORLD);

	double start;
	if (myid == MASTER){
		start = wallTime();
	}

	//sends chuncks of pics to be written by other processors 	
	MPI_Scatter(pic,//send data
		ele_per_process,//send data count 
		MPI_CHAR, //send data type
		local_pic, //recv data
		ele_per_process, //recv count
		MPI_CHAR, //recv date type
		MASTER, //root
		MPI_COMM_WORLD);//communicator


	//do the computation and update local_pic
	//printf("\nmyid= %d\n", myid);
	new_fract(local_pic);


	//send the local_pic to the master 
	MPI_Gather(local_pic,//send data
		ele_per_process,//send count
		MPI_CHAR,//send data type
		pic,//recv data (only root needs valid recv buffer, other can call NULL)
		ele_per_process,//recv count
		MPI_CHAR,//recv data type
		MASTER, //root
		MPI_COMM_WORLD);//communicator

	free(local_pic);

	if (myid == MASTER){
		double end = wallTime();

		printf("\nRuntime = %f msecs\n", end - start);
		//printf("\n#procs = %d\n", numprocs);
		//printf("\nLength per processor = %d\n", ele_per_process / 3); 


		//writes the data to a BMP file
		//See Wikipedia for a description of the BMP format
		int fd;
		if ((fd = open("newton.bmp", O_RDWR + O_CREAT + O_TRUNC, 00644)) == -1) {
			printf("error opening file\n");
			exit(1);
		}
		char buffer[100];
		buffer[0] = 0x42;
		buffer[1] = 0x4D;
		buffer[2] = ((3 * WIDTH * HEIGHT + 40 + 14) & 0xFF);
		buffer[3] = ((3 * WIDTH * HEIGHT + 40 + 14) & 0xFF00) >> 8;
		buffer[4] = ((3 * WIDTH * HEIGHT + 40 + 14) & 0xFF0000) >> 16;
		buffer[5] = ((3 * WIDTH * HEIGHT + 40 + 14) & 0xFF000000) >> 24;
		buffer[10] = 54;
		buffer[11] = buffer[12] = buffer[13] = 0;

		buffer[14] = 40;
		buffer[15] = buffer[16] = buffer[17] = 0;
		buffer[18] = (WIDTH & 0x00FF);
		buffer[19] = (WIDTH & 0xFF00) >> 8;
		buffer[20] = buffer[21] = 0;
		buffer[22] = (HEIGHT & 0x00FF);
		buffer[23] = (HEIGHT & 0xFF00) >> 8;
		buffer[24] = buffer[25] = 0;
		buffer[26] = 1;
		buffer[27] = 0;
		buffer[28] = 24;
		buffer[29] = buffer[30] = buffer[31] = buffer[32] = buffer[33] = 0;
		buffer[34] = ((3 * WIDTH * HEIGHT) & 0xFF);
		buffer[35] = ((3 * WIDTH * HEIGHT) & 0xFF00) >> 8;
		buffer[36] = ((3 * WIDTH * HEIGHT) & 0xFF0000) >> 16;
		buffer[37] = ((3 * WIDTH * HEIGHT) & 0xFF000000) >> 24;
		buffer[38] = buffer[42] = 0x13;
		buffer[39] = buffer[43] = 0x0B;
		buffer[40] = buffer[41] = buffer[44] = buffer[45] = buffer[46] = buffer[47] = buffer[48] = buffer[49] = buffer[50] = buffer[51] = buffer[52] = buffer[53] = 0;
		write(fd, buffer, 54);
		write(fd, pic, WIDTH * HEIGHT * 3);
		close(fd);
		free(pic);
	}


	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();


}
