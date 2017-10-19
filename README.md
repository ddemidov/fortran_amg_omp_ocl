An example of Fortran wrapper for the
[AMGCL](https://github.com/ddemidov/amgcl) library.

The wrapper allows to select compute backend (OpenMP or OpenCL) with a runtime
parameter.

On Windows `OpenCL.dll` is delay-loaded so that the wrapper works even when
OpenCL runtime is not available.
