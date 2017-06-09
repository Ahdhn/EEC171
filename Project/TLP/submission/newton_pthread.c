/* This is a very simple program to generate a Newton fractal */

#include <stdio.h>
#include <fcntl.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <pthread.h>

void hsvTOrgb(double h, double s, double v, double*r, double*g, double *b);


#define WIDTH 1920
#define HEIGHT 1920

const static int total = WIDTH * HEIGHT;

const static double xstart = -1.0;
const static double ystart = -1.0;
const static double xend = 1.0;
const static double yend = 1.0;
#define PI 3.14159265





#define MAXTHRDS 26240





pthread_t callThd[MAXTHRDS];

//#define debug

char *pic;

void new_fract(void*thrdID)
//void new_fract(int thrdID)
{	
	int myThreadID = (int)thrdID; 

	double xstep = (xend - xstart) / WIDTH;
	double ystep = (yend - ystart) / HEIGHT;

	int len = total / MAXTHRDS;//the 1d length 
	int start = myThreadID*len; //start in the 1d map of the 2d pic
	int end = start + len;
	
	int id;
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
			pic[i * WIDTH * 3 + j * 3 + 0] = 0;
			pic[i * WIDTH * 3 + j * 3 + 1] = 0;
			pic[i * WIDTH * 3 + j * 3 + 2] = 0;
		}
		else{
			double r, g, b;
			r = g = b = 0;
			double h = sin((double)color / (double)60 * PI / 2);
			hsvTOrgb(h, 1, 1, &r, &g, &b);
			pic[i * WIDTH * 3 + j * 3 + 0] = r * 255;
			pic[i * WIDTH * 3 + j * 3 + 1] = g * 255;
			pic[i * WIDTH * 3 + j * 3 + 2] = b * 255;
		}
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
 *
 */
void hsvTOrgb(double h, double s, double v, double*r, double*g, double *b /*double rgb[3]*/)
{
	/* hsv and rgb values normalised from 0 - 1 */
	int k;
	double aa, bb, cc, f/*, h = hsv[0], s = hsv[1], v = hsv[2]*/;

	if (s <= 0.0){
		//rgb[0] = rgb[1] = rgb[2] = v;
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
			/*rgb[0] = v;
			rgb[1] = cc;
			rgb[2] = aa;*/
			*r = v;
			*g = cc;
			*b = aa;
			break;
		case 1:
			/*rgb[0] = bb;
			rgb[1] = v;
			rgb[2] = aa;*/
			*r = bb;
			*g = v;
			*b = aa;
			break;
		case 2:
			/*rgb[0] = aa;
			rgb[1] = v;
			rgb[2] = cc;*/
			*r = aa;
			*g = v;
			*b = cc;
			break;
		case 3:
			/*rgb[0] = aa;
			rgb[1] = bb;
			rgb[2] = v;*/
			*r = aa;
			*g = bb;
			*b = v;
			break;
		case 4:
			/*rgb[0] = cc;
			rgb[1] = aa;
			rgb[2] = v;*/
			*r = cc;
			*g = aa;
			*b = v;
			break;
		case 5:
			/*rgb[0] = v;
			rgb[1] = aa;
			rgb[2] = bb;*/
			*r = v;
			*g = aa;
			*b = bb;
			break;

		}
	}
}

int main()
{
		

	pic = (char *)malloc(HEIGHT * WIDTH * 3 * sizeof(char));
	if (pic == NULL) {
		printf("Error allocating memory.\n");	
	}



	int i, check;
	double start = wallTime();
	//************** Main Loop Start **************	
	for (i = 0; i < MAXTHRDS; i++){		
		check = pthread_create(&callThd[i], NULL, (void*)new_fract, (void*)i); 
#ifdef debug
		if (check) {
			printf("Error in thread create\n");
			printf("check=%d, i=%d\n", check, i);
		}
#endif
	}
	
	for (i = 0; i < MAXTHRDS; i++){
		check = pthread_join(callThd[i], NULL);

#ifdef debug
		if (check) { 
			printf("Error in thread join\n"); 
			printf("check=%d, i=%d\n", check, i); 
		}
#endif

	}
	//************** Main Loop End **************
	double end = wallTime();



	printf("\n\nRuntime = %f msecs\n\n",end - start);
	int len = total / MAXTHRDS;
	//printf("#Threads = %d\n", MAXTHRDS);
	//printf("Length per thread = %d\n", len);
	 

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

	return(0);
}
