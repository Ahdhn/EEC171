/*
 * This is an elementary program to illustrate the use of 
 * threads in a program. This program is obtained by modifying a 
 * sequential program that performs a dot product. The main data is 
 * made available to all threads through a globally accessible
 * structure. Each thread works on a different part of the data. 
 * The main thread waits for all the threads to complete their
 * computations, and then it prints the resulting sum.
 * */ 

#include <pthread.h>
#include <stdio.h>
#include <malloc.h>
#include <sys/time.h>



/*
 * The following structure contains the necessary information 
 * to allow the function "dotprod" to access its input data and 
 * place its output into the structure. This structure is 
 * unchanged from the sequential version.
 * */ 

typedef struct
 {
   double      *a;
   double      *b;
   double     sum;
   int     veclen;
 } DOTDATA;

/*Define globally accessible variables and a mutex */ 

#define MAXTHRDS 2
#define VECLEN 1000000
        DOTDATA dotstr;
        pthread_t callThd[MAXTHRDS];
        pthread_mutex_t mutexsum;

double wallTime() // returns time in MS as a double
{
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/*
 * The function dotprod is activated when the thread is created. 
 * As before, all input to this routine is obtained from a structure 
 * of type DOTDATA and all output from this function is written into
 * this structure. The benefit of this approach is apparent for the 
 * multi-threaded program: when a thread is created we pass a single
 * argument to the activated function - typically this argument
 * is a thread number. All the other information required by the 
 * function is accessed from the globally accessible structure.
 * */ 

void* dotprod(void *arg)
{

/*Define and use local variables for convenience */ 

   int i, start, end, offset, len ;
   double mysum, *x, *y;
   offset = (int)arg;

   len = dotstr.veclen;
   /* martyon */
   len = len / MAXTHRDS;
   start = offset*len;
   end   = start + len;
   x = dotstr.a;
   y = dotstr.b;

/*
 * Perform the dot product and assign result
 * to the appropriate variable in the structure.
 * */ 

   mysum = 0;
   for (i=start; i<end ; i++)
    {
      mysum += (x[i] * y[i]);
    }

/*
 * Lock a mutex prior to updating the value in the shared
 * structure, and unlock it upon updating.
 * */ 

   printf ("offset %d mysum %f\n", offset, mysum);
   pthread_mutex_lock (&mutexsum);
   dotstr.sum += mysum;
   pthread_mutex_unlock (&mutexsum);

   pthread_exit((void*)0);
}

/*
 * The main program creates threads which do all the work and then 
 * print out result upon completion. Before creating the threads, 
 * the input data is created. Since all threads update a shared structure, we
 * need a mutex for mutual exclusion. The main thread needs to wait for
 * all threads to complete, it waits for each one of the threads. 
 * */ 

main (int argc, char* argv[])
{
   int i;
   double *a, *b;
   int status;
   int ret;
   double start, end;

/*Assign storage and initialize values */ 

   a = (double*) malloc (VECLEN*sizeof(double));
   b = (double*) malloc (VECLEN*sizeof(double));

   for (i=0; i<VECLEN; i++)
    {
     a[i]=1;
     b[i]=a[i];
    }

   dotstr.veclen = VECLEN;
   dotstr.a = a;
   dotstr.b = b;
   dotstr.sum=0;

   pthread_mutex_init(&mutexsum, NULL);

   /* track exec time */
   start = wallTime();

/*Create threads to perform the dotproduct */


   for (i = 0; i<MAXTHRDS; i++){
	   /*
	   ** Each thread works on a different set of data.
	   ** The offset is specified by 'i'. The size of
	   ** the data for each thread is indicated by VECLEN.
	   */
	   ret = pthread_create(&callThd[i], NULL, dotprod, (void *)i);
	   if (ret) printf("Error in thread create \n");
	   if (ret) printf("ret = %d i = %d \n", ret, i);
   }


/*Wait on the other threads */ 

   for (i = 0; i<MAXTHRDS; i++){
	   /* ret = pthread_join( callThd[i], (void **)&status); */
	   ret = pthread_join(callThd[i], NULL);
	   if (ret) printf("Error in thread join \n");
	   if (ret) printf("ret = %d i = %d \n", ret, i);
   }

   /* track exec time */
   end = wallTime();
   printf ("Runtime = %f msecs \n", end - start);

/*After joining, print out the results and cleanup */


   printf ("Sum =  %f \n", dotstr.sum);
   free (a);
   free (b);
   /* free (dotstr); martyon */
   pthread_mutex_destroy(&mutexsum);
   pthread_exit (0);
}


