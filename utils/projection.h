/* File: projection.h */
/*
  This file is an extenstion of the Corrfunc package
  Copyright (C) 2020-- Kate Storey-Fisher (kstoreyfisher@gmail.com)
  License: MIT LICENSE. See LICENSE file under the top-level
  directory at https://github.com/kstoreyf/Corrfunc/
*/
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "defs.h"

int compute_amplitudes(int ncomponents, int nd1, int nd2, int nr1, int nr2,
            void *dd, void *dr, void *rd, void *rr, void *trr, void *amps, size_t element_size);

int evaluate_xi(int ncomponents, void *amps, int nsvals, void *svals, void *xi, proj_method_t proj_method, size_t element_size, int nsbins, 
          void *sbins, char *projfn, struct extra_options *extra);

int trr_analytic(double rmin, double rmax, int nd, double volume, int ncomponents, 
        void *rr, void *trr, proj_method_t proj_method, 
        size_t element_size, int nsbins, void *sbins,  char *projfn);