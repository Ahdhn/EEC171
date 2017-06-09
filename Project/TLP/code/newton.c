/* This is a very simple program to generate a Newton fractal */

#include <stdio.h>
#include <fcntl.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>

#define PI 3.14159265

const static unsigned int width = 1920;
const static unsigned int height = 1920;

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
void hsvTOrgb(double hsv[3], double rgb[3])
{
	/* hsv and rgb values normalised from 0 - 1 */
	int k;
	double aa, bb, cc, f, h = hsv[0], s = hsv[1], v = hsv[2];

	if (s <= 0.0)
		rgb[0] = rgb[1] = rgb[2] = v;
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
			rgb[0] = v;
			rgb[1] = cc;
			rgb[2] = aa;
			break;
		case 1:
			rgb[0] = bb;
			rgb[1] = v;
			rgb[2] = aa;
			break;
		case 2:
			rgb[0] = aa;
			rgb[1] = v;
			rgb[2] = cc;
			break;
		case 3:
			rgb[0] = aa;
			rgb[1] = bb;
			rgb[2] = v;
			break;
		case 4:
			rgb[0] = cc;
			rgb[1] = aa;
			rgb[2] = v;
			break;
		case 5:
			rgb[0] = v;
			rgb[1] = aa;
			rgb[2] = bb;
			break;

		}
	}
}

int main()
{
	double x, y;
	double xstart, xstep, ystart, ystep;
	double xend, yend;
	double z, zi, newz, newzi;
	double color;
	double hsv[3];
	double f, fprime, fi, fprimei;
	double start, end;
	int iter;
	char *pic;
	int i, j, k;
	int converge;
	int fd;
	char buffer[100];
	pic = (char *)malloc(height * width * 3 * sizeof(char));
	if (pic == NULL) {
		printf("Error allocating memory.\n");
	}

	//Use this set of parameters for (z^3 - 1 = 0)
	xstart = -1.0;
	xend = 1.0;
	ystart = -1.0;
	yend = 1.0;
	iter = 60;

	//Use this set of parameters for (sin(z) = 0)
	/*xstart = PI/2.0 - 0.4;
	xend = PI/2.0 + 0.4;
	ystart = -0.4;
	yend = 0.4;
	iter = 150*/

	//these are used for calculating the complex coordinates corresponding to the pixels */
	xstep = (xend - xstart) / width;
	ystep = (yend - ystart) / height;

	start = wallTime();

	//the main loop
	x = xstart;
	y = ystart;
	for (i = 0; i < height; i++) {
		printf("Now on line: %d\n", i);
		for (j = 0; j < width; j++) {
			z = x;
			zi = y;
			converge = 0;
			for (k = 0; k < iter; k++) {
				//Use Newton's Method to converge to the roots of (z^3 - 1 = 0)
				f = (z * z * z) - (3 * z * zi * zi) - 1;
				fprime = 3 * (z * z - zi * zi);
				fi = (3 * z * z * zi) - (zi * zi * zi);
				fprimei = 6 * z * zi;

				//Use Newton's Method to converge to the roots of (sin(z) = 0)
				/*f = sin(z) * cosh(zi);
				fprime = cos(z) * cosh(zi);
				fi = cos(z) * sinh(zi);
				fprimei = -sin(z) * sinh(zi);*/

				//Newton's Method using complex number division
				newz = z - (f * fprime + fi * fprimei)/(fprime * fprime + fprimei * fprimei);
				newzi = zi - (fi * fprime - f * fprimei)/(fprime * fprime + fprimei * fprimei);

				if (hypot(newz - z,newzi - zi) < 0.001) {
					//The solution has converged
					//Assign the pixel color from the number of iterations
					converge = 1;
					color = k;
					k = iter;
				}
				else {
					//Update the values of z and zi for the next iteration
					z = newz;
					zi = newzi;
				}
			}
			if (!converge) {
				//Color pixel black if it did not converge
				pic[i * width * 3 + j * 3 + 0] = 0;
				pic[i * width * 3 + j * 3 + 1] = 0;
				pic[i * width * 3 + j * 3 + 2] = 0;
			} else {
				//use HSV color space to get colorful results
				//hsv[0] is the hue
				hsv[0] = sin((double)color / (double)iter * PI / 2);
				hsv[1] = 1;
				hsv[2] = 1;
				double rgb[3];
				//convert HSV to RGB
				hsvTOrgb(hsv, rgb);
				pic[i * width * 3 + j * 3 + 0] = rgb[0] * 255;
				pic[i * width * 3 + j * 3 + 1] = rgb[1] * 255;
				pic[i * width * 3 + j * 3 + 2] = rgb[2] * 255;
			}
			x += xstep;
		}
		y += ystep;
		x = xstart;
	}

	end = wallTime();
	printf("Runtime = %f msecs\n",end - start);

	//writes the data to a BMP file
	//See Wikipedia for a description of the BMP format
	if ((fd = open("newton.bmp", O_RDWR + O_CREAT + O_TRUNC, 00644)) == -1) {
		printf("error opening file\n");
		exit(1);
	}
	buffer[0] = 0x42;
	buffer[1] = 0x4D;
	buffer[2] = ((3 * width * height + 40 + 14) & 0xFF);
	buffer[3] = ((3 * width * height + 40 + 14) & 0xFF00) >> 8;
	buffer[4] = ((3 * width * height + 40 + 14) & 0xFF0000) >> 16;
	buffer[5] = ((3 * width * height + 40 + 14) & 0xFF000000) >> 24;
	buffer[10] = 54;
	buffer[11] = buffer[12] = buffer[13] = 0;

	buffer[14] = 40;
	buffer[15] = buffer[16] = buffer[17] = 0;
	buffer[18] = (width & 0x00FF);
	buffer[19] = (width & 0xFF00) >> 8;
	buffer[20] = buffer[21] = 0;
	buffer[22] = (height & 0x00FF);
	buffer[23] = (height & 0xFF00) >> 8;
	buffer[24] = buffer[25] = 0;
	buffer[26] = 1;
	buffer[27] = 0;
	buffer[28] = 24;
	buffer[29] = buffer[30] = buffer[31] = buffer[32] = buffer[33] = 0;
	buffer[34] = ((3 * width * height) & 0xFF);
	buffer[35] = ((3 * width * height) & 0xFF00) >> 8;
	buffer[36] = ((3 * width * height) & 0xFF0000) >> 16;
	buffer[37] = ((3 * width * height) & 0xFF000000) >> 24;
	buffer[38] = buffer[42] = 0x13;
	buffer[39] = buffer[43] = 0x0B;
	buffer[40] = buffer[41] = buffer[44] = buffer[45] = buffer[46] = buffer[47] = buffer[48] = buffer[49] = buffer[50] = buffer[51] = buffer[52] = buffer[53] = 0;
	write(fd, buffer, 54);
	write(fd, pic, width * height * 3);
	close(fd);

	free(pic);

	return(0);
}
