/* File: projection.c */
/*
  This file is an extenstion of the Corrfunc package
  Copyright (C) 2020-- Kate Storey-Fisher (kstoreyfisher@gmail.com)
  License: MIT LICENSE. See LICENSE file under the top-level
  directory at https://github.com/kstoreyf/Corrfunc/
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "defs.h"
#include "projection.h" //function proto-type for API
#include "proj_functions_double.h"//actual implementations for double
#include "proj_functions_float.h"//actual implementations for float


int compute_amplitudes(int ncomponents, int nd1, int nd2, int nr1, int nr2,
            void *dd, void *dr, void *rd, void *rr, void *trr, void *amps, size_t element_size)
{
    if( ! (element_size == sizeof(float) || element_size == sizeof(double))){
        fprintf(stderr,"ERROR: In %s> Can only handle doubles or floats. Got an array of size = %zu\n",
                __FUNCTION__, element_size);
        return EXIT_FAILURE;
    }
    if(element_size == sizeof(float)) {
        return compute_amplitudes_float(ncomponents, nd1, nd2, nr1, nr2,
            (float *) dd, (float *) dr, (float *) rd, (float *) rr, (float *) trr, (float *) amps);
    } else {
        return compute_amplitudes_double(ncomponents, nd1, nd2, nr1, nr2,
            (double *) dd, (double *) dr, (double *) rd, (double *) rr, (double *) trr, (double *) amps);
    }
}


int evaluate_xi(int ncomponents, void *amps, int nsvals, void *svals,
                      void *xi, proj_method_t proj_method, size_t element_size, int nsbins, void *sbins, char *projfn, struct extra_options *extra)
{
    if( ! (element_size == sizeof(float) || element_size == sizeof(double))){
        fprintf(stderr,"ERROR: In %s> Can only handle doubles or floats. Got an array of size = %zu\n",
                __FUNCTION__, element_size);
        return EXIT_FAILURE;
    }

    if(element_size == sizeof(float)) {
        return evaluate_xi_float(ncomponents, (float *) amps, nsvals, (float *) svals,
                      (float *) xi, proj_method, nsbins, (float *) sbins, projfn, extra);
    } else {
        return evaluate_xi_double(ncomponents, (double *) amps, nsvals, (double *) svals,
                      (double *) xi, proj_method, nsbins, (double *) sbins, projfn, extra);
    }
}



int trr_analytic(double rmin, double rmax, int nd, double volume, int ncomponents, 
        void *rr, void *trr, proj_method_t proj_method, 
        size_t element_size, int nsbins, void *sbins, char *projfn)
{
    if( ! (element_size == sizeof(float) || element_size == sizeof(double))){
        fprintf(stderr,"ERROR: In %s> Can only handle doubles or floats. Got an array of size = %zu\n",
                __FUNCTION__, element_size);
        return EXIT_FAILURE;
    }

    if(element_size == sizeof(float)) {
        return trr_analytic_float((float) rmin, (float) rmax, nd, (float) volume,
                    ncomponents, (float *) rr, (float *) trr,
                    proj_method, nsbins, (float *) sbins, projfn);
    } else {
        return trr_analytic_double((double) rmin, (double) rmax, nd, (double) volume,
                    ncomponents, (double *) rr, (double *) trr,
                    proj_method, nsbins, (double *) sbins, projfn);
    }
}
