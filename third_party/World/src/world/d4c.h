//-----------------------------------------------------------------------------
// Copyright 2012 Masanori Morise
// Author: mmorise [at] meiji.ac.jp (Masanori Morise)
// Last update: 2021/02/15
//-----------------------------------------------------------------------------
#ifndef WORLD_D4C_H_
#define WORLD_D4C_H_

#include "world/macrodefinitions.h"

WORLD_BEGIN_C_DECLS

//-----------------------------------------------------------------------------
// Struct for D4C
//-----------------------------------------------------------------------------
typedef struct {
  double threshold;
} D4COption;

//-----------------------------------------------------------------------------
// D4C() calculates the aperiodicity estimated by D4C.
//
// Input:
//   x                  : Input signal
//   x_length           : Length of x
//   fs                 : Sampling frequency
//   temporal_positions : Time axis
//   f0                 : F0 contour
//   f0_length          : Length of F0 contour
//   fft_size           : Number of samples of the aperiodicity in one frame.
//                      : It is given by the equation fft_size / 2 + 1.
// Output:
//   aperiodicity  : Aperiodicity estimated by D4C.
//-----------------------------------------------------------------------------
void D4C(const double *x, int x_length, int fs,
    const double *temporal_positions, const double *f0, int f0_length,
    int fft_size, const D4COption *option, double **aperiodicity);

//-----------------------------------------------------------------------------
// GetD4CAperiodicity0() exposes the raw D4C LoveTrain statistic per frame,
// the ratio D4C compares against D4COption::threshold to label a frame
// unvoiced (aperiodicity 1.0). High values are periodic (voiced-like), low
// values aperiodic; frames with f0 == 0 read 0.0. Exposed for #55 so a gate
// can be tuned against the statistic itself instead of the binary output of
// D4C().
//
// Input:
//   x                  : Input signal
//   x_length           : Length of x
//   fs                 : Sampling frequency
//   temporal_positions : Time axis
//   f0                 : F0 contour
//   f0_length          : Length of F0 contour
// Output:
//   aperiodicity0  : Raw LoveTrain ratio per frame.
//-----------------------------------------------------------------------------
void GetD4CAperiodicity0(const double *x, int x_length, int fs,
    const double *temporal_positions, const double *f0, int f0_length,
    double *aperiodicity0);

//-----------------------------------------------------------------------------
// InitializeD4COption allocates the memory to the struct and sets the
// default parameters.
//
// Output:
//   option   : Struct for the optional parameter.
//-----------------------------------------------------------------------------
void InitializeD4COption(D4COption *option);

WORLD_END_C_DECLS

#endif  // WORLD_D4C_H_
