/*
The MIT License

Copyright (c) 2017 Denis Demidov <dennis.demidov@gmail.com>

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

#include <iostream>
#include <functional>

#include <boost/range/adaptor/transformed.hpp>
#include <boost/variant.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/backend/vexcl.hpp>
#include <amgcl/runtime.hpp>
#include <amgcl/preconditioner/runtime.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

#include "amgcl.h"

#ifdef WIN32
#  include <windows.h>
#  include <delayimp.h>
#endif

//---------------------------------------------------------------------------
typedef boost::property_tree::ptree Params;

//---------------------------------------------------------------------------
amgclHandle STDCALL amgcl_params_create() {
    return static_cast<amgclHandle>( new Params() );
}

//---------------------------------------------------------------------------
void STDCALL amgcl_params_seti(amgclHandle prm, const char *name, int value) {
    static_cast<Params*>(prm)->put(name, value);
}

//---------------------------------------------------------------------------
void STDCALL amgcl_params_setf(amgclHandle prm, const char *name, float value) {
    static_cast<Params*>(prm)->put(name, value);
}

//---------------------------------------------------------------------------
void STDCALL amgcl_params_sets(amgclHandle prm, const char *name, const char *value) {
    static_cast<Params*>(prm)->put(name, value);
}

//---------------------------------------------------------------------------
void STDCALL amgcl_params_read_json(amgclHandle prm, const char *fname) {
    try {
        read_json(fname, *static_cast<Params*>(prm));
    } catch(const std::exception &e) {
        std::cout
            << "Failed to read \"" << fname << "\"" << std::endl
            << "  Reason: " << e.what() << std::endl;
    }
}

//---------------------------------------------------------------------------
void STDCALL amgcl_params_destroy(amgclHandle prm) {
    delete static_cast<Params*>(prm);
}

//---------------------------------------------------------------------------
bool opencl_available() {
#ifdef WIN32
    __try {
        if (FAILED(__HrLoadAllImportsForDll("OpenCL.dll")))
            return false;
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
#endif
    return true;
}

//---------------------------------------------------------------------------
vex::Context& ctx(int devnum = 0) {
    static vex::Context c(vex::Filter::DoublePrecision && vex::Filter::Position(devnum));
    return c;
}

//---------------------------------------------------------------------------
typedef amgcl::backend::builtin<double> openmp;
typedef amgcl::backend::vexcl<double>   opencl;

typedef
    amgcl::make_solver<
        amgcl::runtime::preconditioner<openmp>,
        amgcl::runtime::iterative_solver<openmp>
        >
    openmp_solver;

typedef
    amgcl::make_solver<
        amgcl::runtime::preconditioner<opencl>,
        amgcl::runtime::iterative_solver<opencl>
        >
    opencl_solver;

typedef boost::variant<openmp_solver*, opencl_solver*> solver;

//---------------------------------------------------------------------------
amgclHandle STDCALL amgcl_solver_create(
        int           n,
        const int    *ptr,
        const int    *col,
        const double *val,
        int           device,
        amgclHandle   prm
        )
{
    const int nnz = ptr[n] - 1;

    auto ptr_rng = boost::make_iterator_range(ptr, ptr + n+1);
    auto col_rng = boost::make_iterator_range(col, col + nnz);
    auto val_rng = boost::make_iterator_range(val, val + nnz);

    auto A = boost::make_tuple(n,
            ptr_rng | boost::adaptors::transformed([](int i){ return i - 1; }),
            col_rng | boost::adaptors::transformed([](int i){ return i - 1; }),
            val_rng
            );

    Params p;
    if (prm) p = *static_cast<Params*>(prm);

    if (device < 0) {
        return static_cast<amgclHandle>(new solver(new openmp_solver(A, p)));
    } else {
        amgcl::precondition(opencl_available(), "OpenCL.dll not found!");

        opencl::params bp;
        bp.q = ctx(device);

        return static_cast<amgclHandle>(new solver(new opencl_solver(A, p, bp)));
    }
}

//---------------------------------------------------------------------------
struct solver_report : boost::static_visitor<> {
    template <class S>
    void operator()(const S *s) const {
        std::cout << s->precond() << std::endl;
    }
};

//---------------------------------------------------------------------------
void STDCALL amgcl_solver_report(amgclHandle handle) {
    const solver *s = static_cast<const solver*>(handle);
    boost::apply_visitor(solver_report(), *s);
}

//---------------------------------------------------------------------------
struct solver_destroy : boost::static_visitor<> {
    template <class S>
    void operator()(const S *s) const {
        delete s;
    }
};

//---------------------------------------------------------------------------
void STDCALL amgcl_solver_destroy(amgclHandle handle) {
    const solver *s = static_cast<const solver*>(handle);
    boost::apply_visitor(solver_destroy(), *s);
    delete s;
}

//---------------------------------------------------------------------------
struct solver_solve : boost::static_visitor<> {
    const double *rhs;
    double *x;
    mutable conv_info conv;

    solver_solve(const double *rhs, double *x) : rhs(rhs), x(x) {}

    void operator()(const openmp_solver *s) const {
        size_t n = s->size();

        auto X = boost::make_iterator_range(x, x + n);
        auto F = boost::make_iterator_range(rhs, rhs + n);

        boost::tie(conv.iterations, conv.residual) = (*s)(F, X);
    }

    void operator()(const opencl_solver *s) const {
        size_t n = s->size();

        vex::vector<double> X(ctx(), n, x);
        vex::vector<double> F(ctx(), n, rhs);

        boost::tie(conv.iterations, conv.residual) = (*s)(F, X);

        vex::copy(X.begin(), X.end(), x);
    }
};

//---------------------------------------------------------------------------
conv_info STDCALL amgcl_solver_solve(
        amgclHandle handle,
        const double *rhs,
        double *x
        )
{
    const solver *s = static_cast<const solver*>(handle);
    solver_solve solve(rhs, x);
    boost::apply_visitor(solve, *s);
    return solve.conv;
}
