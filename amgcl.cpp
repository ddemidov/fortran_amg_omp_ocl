/*
The MIT License

Copyright (c) 2017-2019 Denis Demidov <dennis.demidov@gmail.com>

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

#include <boost/core/ignore_unused.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/variant.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/filesystem.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/backend/vexcl.hpp>
#include <amgcl/coarsening/runtime.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/solver/runtime.hpp>
#include <amgcl/preconditioner/runtime.hpp>
#include <amgcl/preconditioner/schur_pressure_correction.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/profiler.hpp>

#ifdef AMGCL_DEBUG
#  include <amgcl/io/mm.hpp>
#endif

#include "amgcl.h"

#ifdef WIN32
#  include <windows.h>
#  include <delayimp.h>
#endif

//---------------------------------------------------------------------------
typedef boost::property_tree::ptree Params;

//---------------------------------------------------------------------------
amgclHandle STDCALL amgcl_profile_create() {
    return static_cast<amgclHandle>( new amgcl::profiler<>() );
}

//---------------------------------------------------------------------------
void STDCALL amgcl_profile_tic(amgclHandle p, const char *name) {
    static_cast<amgcl::profiler<>*>(p)->tic(name);
}

//---------------------------------------------------------------------------
void STDCALL amgcl_profile_toc(amgclHandle p, const char *name) {
    static_cast<amgcl::profiler<>*>(p)->toc(name);
}

//---------------------------------------------------------------------------
void STDCALL amgcl_profile_report(amgclHandle p) {
    std::cout << *static_cast<amgcl::profiler<>*>(p) << std::endl;
}

//---------------------------------------------------------------------------
void STDCALL amgcl_profile_destroy(amgclHandle p) {
    delete static_cast<amgcl::profiler<>*>(p);
}

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

    static bool once = [](){ std::cout << c << std::endl; return true; }();
    boost::ignore_unused(once);

    return c;
}

//---------------------------------------------------------------------------
typedef amgcl::backend::builtin<double> openmp;
typedef amgcl::backend::vexcl<double>   opencl;

template <class Backend>
using amg_solver =
    amgcl::make_solver<
        amgcl::runtime::preconditioner<Backend>,
        amgcl::runtime::solver::wrapper<Backend>
        >;

template <class Backend>
struct solver_type;

template <> struct solver_type<openmp> : amg_solver<openmp>
{
    typedef amg_solver<openmp> Base;
    using Base::Base;
};
    
template <> struct solver_type<opencl> : amg_solver<opencl>
{
    typedef amg_solver<opencl> Base;
    typedef Base::params params;
    typedef opencl::params backend_params;

    mutable vex::vector<double> X, F;

    template <class Matrix>
    solver_type(
            const Matrix &A,
            const params &prm = params(),
            const backend_params &bprm = backend_params()
            )
        : Base(A, prm, bprm),
          X(bprm.q, amgcl::backend::rows(A)),
          F(bprm.q, amgcl::backend::rows(A))
    {}
};
    

typedef boost::variant<
    solver_type<openmp>*,
    solver_type<opencl>*
    > solver;

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

    auto A = std::make_tuple(n,
            ptr_rng | boost::adaptors::transformed([](int i){ return i - 1; }),
            col_rng | boost::adaptors::transformed([](int i){ return i - 1; }),
            val_rng
            );

    Params p;
    if (prm) p = *static_cast<Params*>(prm);

    if (boost::filesystem::exists("libamgcl.json")) {
        read_json("libamgcl.json", p);
    }

#ifdef AMGCL_DEBUG
    if (boost::filesystem::exists("libamgcl-device.txt")) {
        std::ifstream f("libamgcl-device.txt");
        f >> device;
    }

    if (boost::filesystem::exists("libamgcl-debug.json")) {
        Params dbg;
        read_json("libamgcl-debug.json", dbg);

        std::string matrix = "A.mtx", params = "p.json";
        dbg.get("matrix", matrix);
        dbg.get("params", params);

        amgcl::io::mm_write(matrix, A);

        std::ofstream pfile(params);
        write_json(pfile, p);
    }
#endif

    if (device < 0) {
        return static_cast<amgclHandle>(new solver(new solver_type<openmp>(A, p)));
    } else {
        amgcl::precondition(opencl_available(), "OpenCL.dll not found!");

        opencl::params bp;
        bp.q = ctx(device);

        return static_cast<amgclHandle>(new solver(new solver_type<opencl>(A, p, bp)));
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

    template <template <class> class S>
    void operator()(const S<openmp> *s) const {
        size_t n = s->size();

        auto X = boost::make_iterator_range(x, x + n);
        auto F = boost::make_iterator_range(rhs, rhs + n);

        std::tie(conv.iterations, conv.residual) = (*s)(F, X);
    }

    template <template <class> class S>
    void operator()(const S<opencl> *s) const {
        size_t n = s->size();

        vex::copy(x, x + n, s->X.begin());
        vex::copy(rhs, rhs + n, s->F.begin());

        std::tie(conv.iterations, conv.residual) = (*s)(s->F, s->X);

        vex::copy(s->X.begin(), s->X.end(), x);
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

//---------------------------------------------------------------------------
template <class Backend>
using schur_solver =
    amgcl::make_solver<
        amgcl::preconditioner::schur_pressure_correction<
            amgcl::make_solver<
                amgcl::relaxation::as_preconditioner<
                    Backend,
                    amgcl::runtime::relaxation::wrapper
                    >,
                amgcl::runtime::solver::wrapper<Backend>
            >,
            amgcl::make_solver<
                amgcl::amg<
                    Backend,
                    amgcl::runtime::coarsening::wrapper,
                    amgcl::runtime::relaxation::wrapper
                    >,
                amgcl::runtime::solver::wrapper<Backend>
            >
        >,
        amgcl::runtime::solver::wrapper<Backend>
    >;

template <class Backend>
struct spc_solver_type;

template <> struct spc_solver_type<openmp> : schur_solver<openmp>
{
    typedef schur_solver<openmp> Base;
    using Base::Base;
};
    
template <> struct spc_solver_type<opencl> : schur_solver<opencl>
{
    typedef schur_solver<opencl> Base;
    typedef Base::params params;
    typedef opencl::params backend_params;

    mutable vex::vector<double> X, F;

    template <class Matrix>
    spc_solver_type(
            const Matrix &A,
            const params &prm = params(),
            const backend_params &bprm = backend_params()
            )
        : Base(A, prm, bprm),
          X(bprm.q, amgcl::backend::rows(A)),
          F(bprm.q, amgcl::backend::rows(A))
    {}
};
typedef boost::variant<
    spc_solver_type<openmp>*,
    spc_solver_type<opencl>*
    > spc_solver;

//---------------------------------------------------------------------------
amgclHandle STDCALL amgcl_schur_pc_create(
        int           n,
        const int    *ptr,
        const int    *col,
        const double *val,
        int           pvars,
        int           device,
        amgclHandle   prm
        )
{
    const int nnz = ptr[n] - 1;

    auto ptr_rng = boost::make_iterator_range(ptr, ptr + n+1);
    auto col_rng = boost::make_iterator_range(col, col + nnz);
    auto val_rng = boost::make_iterator_range(val, val + nnz);

    auto A = std::make_tuple(n,
            ptr_rng | boost::adaptors::transformed([](int i){ return i - 1; }),
            col_rng | boost::adaptors::transformed([](int i){ return i - 1; }),
            val_rng
            );

    Params p;
    if (prm) p = *static_cast<Params*>(prm);

    if (boost::filesystem::exists("libamgcl.json")) {
        read_json("libamgcl.json", p);
    }

    p.put("precond.pmask_size", n);
    p.put("precond.pmask_pattern", std::string("<") + std::to_string(pvars));

#ifdef AMGCL_DEBUG
    if (boost::filesystem::exists("libamgcl-device.txt")) {
        std::ifstream f("libamgcl-device.txt");
        f >> device;
    }

    if (boost::filesystem::exists("libamgcl-debug.json")) {
        Params dbg;
        read_json("libamgcl-debug.json", dbg);

        std::string matrix = "A.mtx", params = "p.json";
        dbg.get("matrix", matrix);
        dbg.get("params", params);

        amgcl::io::mm_write(matrix, A);

        std::ofstream pfile(params);
        write_json(pfile, p);
    }
#endif

    if (device < 0) {
        return static_cast<amgclHandle>(new spc_solver(new spc_solver_type<openmp>(A, p)));
    } else {
        amgcl::precondition(opencl_available(), "OpenCL.dll not found!");

        opencl::params bp;
        bp.q = ctx(device);

        return static_cast<amgclHandle>(new spc_solver(new spc_solver_type<opencl>(A, p, bp)));
    }
}

//---------------------------------------------------------------------------
void STDCALL amgcl_schur_pc_report(amgclHandle handle) {
    const spc_solver *s = static_cast<const spc_solver*>(handle);
    boost::apply_visitor(solver_report(), *s);
}

//---------------------------------------------------------------------------
conv_info STDCALL amgcl_schur_pc_solve(
        amgclHandle    handle,
        double const * rhs,
        double       * x
        )
{
    const spc_solver *s = static_cast<const spc_solver*>(handle);
    solver_solve solve(rhs, x);
    boost::apply_visitor(solve, *s);
    return solve.conv;
}

//---------------------------------------------------------------------------
void STDCALL amgcl_schur_pc_destroy(amgclHandle handle) {
    const spc_solver *s = static_cast<const spc_solver*>(handle);
    boost::apply_visitor(solver_destroy(), *s);
    delete s;
}
