#ifndef LIB_AMGCL_H
#define LIB_AMGCL_H

/*
The MIT License

Copyright (c) 20172-2019 Denis Demidov <dennis.demidov@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifdef WIN32
#  define STDCALL __stdcall
#else
#  define STDCALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* amgclHandle;

// Create profiler.
amgclHandle STDCALL amgcl_profile_create();

// Start measurement.
void STDCALL amgcl_profile_tic(amgclHandle p, const char *name);

// Stop measurement.
void STDCALL amgcl_profile_toc(amgclHandle p, const char *name);

// Show profiler report.
void STDCALL amgcl_profile_report(amgclHandle p);

// Destroy profiler.
void STDCALL amgcl_profile_destroy(amgclHandle p);

// Create parameter list.
amgclHandle STDCALL amgcl_params_create();

// Set integer parameter in a parameter list.
void STDCALL amgcl_params_seti(amgclHandle prm, const char *name, int   value);

// Set floating point parameter in a parameter list.
void STDCALL amgcl_params_setf(amgclHandle prm, const char *name, float value);

// Set string parameter in a parameter list.
void STDCALL amgcl_params_sets(amgclHandle prm, const char *name, const char *value);

// Read parameters from a JSON file.
void STDCALL amgcl_params_read_json(amgclHandle prm, const char *fname);

// Destroy parameter list.
void STDCALL amgcl_params_destroy(amgclHandle prm);

// Convergence info
struct conv_info {
    int    iterations;
    double residual;
};

// Create iterative solver preconditioned by AMG.
// ptr and col arrays are 1-based (as in Fortran).
//
// When dev=-1, solver uses CPU with OpenMP backend.
// Otherwise, dev-th compute device is used with OpenCL backend.
// May throw if OpenCL is not available or the specified device is not found.
amgclHandle STDCALL amgcl_solver_create(
        int           n,
        const int    *ptr,
        const int    *col,
        const double *val,
        int           devnum,
        amgclHandle   parameters
        );

// Solve the problem for the given right-hand side.
conv_info STDCALL amgcl_solver_solve(
        amgclHandle    solver,
        double const * rhs,
        double       * x
        );

// Printout solver structure
void STDCALL amgcl_solver_report(amgclHandle solver);

// Destroy iterative solver.
void STDCALL amgcl_solver_destroy(amgclHandle solver);


amgclHandle STDCALL amgcl_schur_pc_create(
        int           n,
        const int    *ptr,
        const int    *col,
        const double *val,
        int           pressure_vars,
        int           devnum,
        amgclHandle   parameters
        );

void STDCALL amgcl_schur_pc_report(amgclHandle solver);

conv_info STDCALL amgcl_schur_pc_solve(
        amgclHandle    solver,
        double const * rhs,
        double       * x
        );

void STDCALL amgcl_schur_pc_destroy(amgclHandle solver);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
