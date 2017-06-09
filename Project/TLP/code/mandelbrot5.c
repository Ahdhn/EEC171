/* This is a very simple program to create the mandelbrot set */

#include <stdio.h>
#include <fcntl.h>
#include <math.h>
#include <stdlib.h>

#define width 1280
#define height 960
#define RSAT 255
#define BSAT 255
#define GSAT 255

/*
 *  hsvTOrgb
 *  converts an hsv (hue, saturation, value) colour value to rgb
 *  (red, green, blue)
 *  Created by Jon McCormack on Sat Jul 10 2004.
 *
 */
void hsvTOrgb(double hsv[3], char rgb[3])
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

main()
{
	double x, y;
	double xstart, xstep, ystart, ystep;
	double xend, yend;
	double z, zi, newz, newzi;
	double colour;
	double hsv[3];
	int iter;
	long col;
	char pic[height][width][3];
	int i, j, k;
	int inset;
	int fd;
	char buffer[100];

	/* Read in the initial data */
	printf("Enter xstart, xend, ystart, yend, iterations: ");
	if (scanf("%lf%lf%lf%lf%d", &xstart, &xend, &ystart, &yend, &iter)
	    != 5) {
		printf("Error!\n");
		exit(1);
	}

	/* these are used for calculating the points corresponding to the
	   pixels */
	xstep = (xend - xstart) / width;
	ystep = (yend - ystart) / height;

	/*the main loop */
	x = xstart;
	y = ystart;
	for (i = 0; i < height; i++) {
		printf("Now on line: %d\n", i);
		for (j = 0; j < width; j++) {
			z = 0;
			zi = 0;
			inset = 1;
			for (k = 0; k < iter; k++) {
				/* z^2 = (a+bi)(a+bi) = a^2 + 2abi - b^2 */
				newz = (z * z) - (zi * zi) + x;
				newzi = 2 * z * zi + y;
				z = newz;
				zi = newzi;
				if (((z * z) + (zi * zi)) > 4) {
					inset = 0;
					colour = k;
					k = iter;
				}
			}
			if (inset) {
				pic[i][j][0] = 0;
				pic[i][j][1] = 0;
				pic[i][j][2] = 0;
			} else {
/*
	pic[i][j][0] = iter % 256;
	pic[i][j][1] = colour / iter * RSAT / 2;
	pic[i][j][2] = colour / iter * GSAT / 4; 
*/
				hsv[0] = colour / iter * 255;
				hsv[1] = colour / iter * 255 / 2;
				hsv[2] = colour / iter * 255 / 4;
				hsvTOrgb(hsv, &pic[i][j][0]);
/*
	pic[i][j][0] = colour / iter * BSAT;
	pic[i][j][1] = colour / iter * RSAT / 2;
	pic[i][j][2] = colour / iter * GSAT / 4; 
*/
/*
	pic[i][j][1] = 0;
	pic[i][j][2] = 0;
*/
			}
			x += xstep;
		}
		y += ystep;
		x = xstart;
	}

	/* writes the data to a TGA file */
	if ((fd = open("mand.tga", O_RDWR + O_CREAT, 00644)) == -1) {
		printf("error opening file\n");
		exit(1);
	}
	buffer[0] = 0;
	buffer[1] = 0;
	buffer[2] = 2;		/* rgb image type */
	buffer[8] = 0;
	buffer[9] = 0;
	buffer[10] = 0;
	buffer[11] = 0;
	buffer[12] = (width & 0x00FF);
	buffer[13] = (width & 0xFF00) >> 8;
	buffer[14] = (height & 0x00FF);
	buffer[15] = (height & 0xFF00) >> 8;
	buffer[16] = 24;
	buffer[17] = 0;
	write(fd, buffer, 18);
	write(fd, pic, width * height * 3);
	close(fd);
}
