! The MIT License
!
! Copyright (c) 2017 Denis Demidov <dennis.demidov@gmail.com>
!
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.

module amgcl
    use iso_c_binding
    private
    public c_size_t, c_int, c_double, c_char, conv_info, &
        amgcl_params_create, amgcl_params_seti, amgcl_params_setf, amgcl_params_sets, amgcl_params_destroy, &
        amgcl_solver_create, amgcl_solver_solve, amgcl_solver_report, amgcl_solver_destroy

    type, bind(C) :: conv_info
        integer (c_int)    :: iterations
        real    (c_double) :: residual
    end type

    interface
        ! Create parameter list.
        integer(c_size_t) &
        function amgcl_params_create() bind (C, name="amgcl_params_create")
            use iso_c_binding
        end function

        ! Set integer parameter in a parameter list.
        subroutine amgcl_params_seti_c(prm, name, val) bind (C, name="amgcl_params_seti")
            use iso_c_binding

            integer   (c_size_t), intent(in), value :: prm
            character (c_char),   intent(in)        :: name(*)
            integer   (c_int),    intent(in), value :: val
        end subroutine

        ! Set floating point parameter in a parameter list.
        subroutine amgcl_params_setf_c(prm, name, val) bind (C, name="amgcl_params_setf")
            use iso_c_binding
            integer   (c_size_t), intent(in), value :: prm
            character (c_char),   intent(in)        :: name(*)
            real      (c_float),  intent(in), value :: val
        end subroutine

        ! Set string parameter in a parameter list.
        subroutine amgcl_params_sets_c(prm, name, val) bind (C, name="amgcl_params_sets")
            use iso_c_binding
            integer   (c_size_t), intent(in), value :: prm
            character (c_char),   intent(in)        :: name(*)
            character (c_char),   intent(in)        :: val(*)
        end subroutine

        ! Destroy parameter list.
        subroutine amgcl_params_destroy(prm) bind(C, name="amgcl_params_destroy")
            use iso_c_binding
            integer (c_size_t), intent(in), value :: prm
        end subroutine

        ! Create iterative solver preconditioned by AMG.
        ! ptr and col arrays are 1-based (as in Fortran).
        !
        ! When dev=-1, solver uses CPU with OpenMP backend.
        ! Otherwise, dev-th compute device is used with OpenCL backend.
        ! May throw if OpenCL is not available or the specified device is not found.
        integer(c_size_t) &
        function amgcl_solver_create (n, ptr, col, val, devnum, prm) bind (C, name="amgcl_solver_create")
            use iso_c_binding
            integer (c_int),    intent(in), value :: n
            integer (c_int),    intent(in)        :: ptr(*)
            integer (c_int),    intent(in)        :: col(*)
            real    (c_double), intent(in)        :: val(*)
            integer (c_int),    intent(in), value :: devnum
            integer (c_size_t), intent(in), value :: prm
        end function

        ! Solve the problem for the given right-hand side.
        type(conv_info) &
        function amgcl_solver_solve(solver, rhs, x) bind (C, name="amgcl_solver_solve")
            use iso_c_binding
            integer (c_size_t), intent(in), value :: solver
            real    (c_double), intent(in)        :: rhs(*)
            real    (c_double), intent(inout)     :: x(*)

            type, bind(C) :: conv_info
                integer (c_int)    :: iterations;
                real    (c_double) :: residual
            end type
        end function

        ! Printout solver structure
        subroutine amgcl_solver_report(solver) bind(C, name="amgcl_solver_report")
            use iso_c_binding
            integer (c_size_t), intent(in), value :: solver
        end subroutine

        ! Destroy iterative solver.
        subroutine amgcl_solver_destroy(solver) bind(C, name="amgcl_solver_destroy")
            use iso_c_binding
            integer (c_size_t), intent(in), value :: solver
        end subroutine
    end interface

    contains

    ! Set integer parameter in a parameter list.
    subroutine amgcl_params_seti(prm, name, val)
        use iso_c_binding
        integer   (c_size_t), intent(in), value :: prm
        character (len=*),    intent(in)        :: name
        integer   (c_int),    intent(in), value :: val

        call amgcl_params_seti_c(prm, name // c_null_char, val)
    end subroutine

    ! Set floating point parameter in a parameter list.
    subroutine amgcl_params_setf(prm, name, val)
        use iso_c_binding
        integer   (c_size_t), intent(in), value :: prm
        character (len=*),    intent(in)        :: name
        real      (c_float),  intent(in), value :: val

        call amgcl_params_setf_c(prm, name // c_null_char, val)
    end subroutine

    ! Set string parameter in a parameter list.
    subroutine amgcl_params_sets(prm, name, val)
        use iso_c_binding
        integer   (c_size_t), intent(in), value :: prm
        character (len=*),    intent(in)        :: name
        character (len=*),    intent(in)        :: val

        call amgcl_params_sets_c(prm, name // c_null_char, val // c_null_char)
    end subroutine

end module
