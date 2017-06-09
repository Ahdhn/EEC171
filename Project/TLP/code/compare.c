/* Simple program to compare your parallel
 * and serial versions of the fractal images.
 *
 * Usage: compare newton_ser.bmp newton_par.bmp
 */

#include <stdio.h>
int main(int argc, char *argv[]) {
	FILE *fp1, *fp2, *fp3;
	int in1, in2, mismatch = -1;
	if((argc == 3) && (fp1 = fopen(argv[1], "rb")) && (fp2 = fopen(argv[2], "rb")) && (fp3 = fopen("mask.bmp", "wb")))
		for(in1 = fgetc(fp1), in2 = fgetc(fp2), mismatch = 0; !feof(fp1) && !feof(fp2); in1 = fgetc(fp1), in2 = fgetc(fp2))
			if (ftell(fp1) <= 53) fputc(in1, fp3); // copy the header; don't compare
			else if(in1 != in2) fputc(0xff, fp3), mismatch++; // not equal pixel: 0xff (R, G or B)
			else fputc(0x00, fp3); // equal pixel: output 0x00 (black)
	if(mismatch >= 0) printf("%d mismatch(es)\nWriting mismatch(es) to file mask.bmp...\n", mismatch);
	
	return(0);
}
