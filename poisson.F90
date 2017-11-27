program poisson
    use, intrinsic :: iso_c_binding
    use amgcl
    implicit none

    integer :: n, n3, idx, nnz, i, j, k, devnum
    character(len=32) arg
    integer(c_int), allocatable :: ptr(:), col(:)
    real(c_double), allocatable :: val(:), rhs(:), x(:)
    integer(c_size_t) :: prof, solver, params
    type(conv_info) :: cnv

    ! Device to use:
    devnum = -1
    if (iargc() > 0) then
        call getarg(1, arg)
        read(arg, "(i4)") devnum
    endif

    ! Create profiler.
    prof = amgcl_profile_create();

    ! Assemble matrix in CRS format for a Poisson problem in n x n square.
    call amgcl_profile_tic(prof, "assemble")
    n  = 64
    n3 = n * n * n

    allocate(ptr(n3 + 1))
    allocate(col(n3 * 7))
    allocate(val(n3 * 7))
    ptr(1) = 1

    idx = 1
    nnz = 0

    do k = 1,n
        do j = 1,n
            do i = 1,n
                if (k > 1) then
                    nnz = nnz + 1
                    col(nnz) = idx - n * n
                    val(nnz) = -1
                end if

                if (j > 1) then
                    nnz = nnz + 1
                    col(nnz) = idx - n
                    val(nnz) = -1
                end if

                if (i > 1) then
                    nnz = nnz + 1
                    col(nnz) = idx - 1
                    val(nnz) = -1
                end if

                nnz = nnz + 1
                col(nnz) = idx
                val(nnz) = 6

                if (i < n) then
                    nnz = nnz + 1
                    col(nnz) = idx + 1
                    val(nnz) = -1
                end if

                if (j < n) then
                    nnz = nnz + 1
                    col(nnz) = idx + n
                    val(nnz) = -1
                end if

                if (k < n) then
                    nnz = nnz + 1
                    col(nnz) = idx + n * n
                    val(nnz) = -1
                end if

                idx = idx + 1
                ptr(idx) = nnz + 1
            end do
        end do
    end do
    call amgcl_profile_toc(prof, "assemble")

    allocate(rhs(n3))
    allocate(x(n3))
    rhs = 1
    x = 0

    ! Create solver parameters.
    params = amgcl_params_create()
    call amgcl_params_sets(params, "solver.type", "bicgstab")
    call amgcl_params_setf(params, "solver.tol", 1e-6)
    call amgcl_params_sets(params, "precond.relax.type", "spai0")

    ! Read parameters from a JSON file.
    ! An example of JSON file with the above parameters:
    ! {
    !   "solver" : {
    !     "type" : "bicgstab",
    !     "tol" : 1e-6
    !   },
    !   "precond" : {
    !     "relax" : {
    !       "type" : "spai0"
    !     }
    !   }
    ! }
    ! call amgcl_params_read_json(params, "params.json");

    ! Create solver, printout its structure.
    call amgcl_profile_tic(prof, "setup")
    solver = amgcl_solver_create(n3, ptr, col, val, devnum, params)
    call amgcl_profile_toc(prof, "setup")
    call amgcl_solver_report(solver)

    ! Solve the problem for the given right-hand-side.
    call amgcl_profile_tic(prof, "solve")
    cnv = amgcl_solver_solve(solver, rhs, x)
    call amgcl_profile_toc(prof, "solve")
    write(*,"('Iterations: ', I3, ', residual: ', E13.6)") cnv%iterations, cnv%residual

    ! Show profiling info
    call amgcl_profile_report(prof)

    ! Destroy solver and parameter pack.
    call amgcl_solver_destroy(solver)
    call amgcl_params_destroy(params)
    call amgcl_profile_destroy(prof)
end program poisson
