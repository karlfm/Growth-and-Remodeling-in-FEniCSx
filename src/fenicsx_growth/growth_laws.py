import dolfinx
import ufl
import basix


def initiate_functions(mesh):
    """Initiate first growth tensor"""
    tensor_space = dolfinx.fem.functionspace(
        mesh,
        basix.ufl.element(
            family="DG", cell=str(mesh.ufl_cell()), degree=1, shape=(3, 3)
        ),
    )
    X = ufl.SpatialCoordinate(mesh)  # get Identity without it being a constant
    Identity = ufl.variable(ufl.grad(X))
    Identity_expression = dolfinx.fem.Expression(
        Identity, tensor_space.element.interpolation_points()
    )

    F_g_tot_function = dolfinx.fem.Function(tensor_space)
    F_g_tot_function.interpolate(Identity_expression)
    E_e_function = dolfinx.fem.Function(
        tensor_space
    )  # E_e_function.interpolate(Identity_expression)

    return F_g_tot_function, E_e_function, tensor_space


def KOM(mesh, F):
    """Initiate first growth tensor"""
    F_g_tot_function, E_e_function, tensor_space = initiate_functions(mesh)

    """Constants from the paper"""
    f_ff_max = 0.31
    f_f = 150
    s_l50 = 0.06
    F_ff50 = 1.35
    f_l_slope = 40
    f_cc_max = 0.1
    c_f = 75
    s_t50 = 0.07
    F_cc50 = 1.28
    c_th_slope = 60

    """Growth Laws"""

    def k_growth(
        F_g_cum: dolfinx.fem.Function, slope: int, F_50: dolfinx.fem.Function
    ) -> dolfinx.fem.Function:
        return 1 / (1 + ufl.exp(slope * (F_g_cum - F_50)))

    def alg_max_princ_strain(E: dolfinx.fem.Function) -> dolfinx.fem.Function:
        return (E[1, 1] + E[2, 2]) / 2 + ufl.sqrt(
            ((E[1, 1] - E[2, 2]) / 2) ** 2 + (E[1, 2] * E[2, 1])
        )

    dt = 0.01

    # Growth in the fiber direction
    F_gff = ufl.conditional(
        ufl.ge(E_e_function[0, 0], 0),
        k_growth(F_g_tot_function[0, 0], f_l_slope, F_ff50)
        * f_ff_max
        * dt
        / (1 + ufl.exp(-f_f * (E_e_function[0, 0] - s_l50)))
        + 1,
        -f_ff_max * dt / (1 + ufl.exp(f_f * (E_e_function[0, 0] + s_l50))) + 1,
    )

    # Growth in the cross-fiber direction
    F_gcc = ufl.conditional(
        ufl.ge(alg_max_princ_strain(E_e_function), 0),
        ufl.sqrt(
            k_growth(F_g_tot_function[1, 1], c_th_slope, F_cc50)
            * f_cc_max
            * dt
            / (1 + ufl.exp(-c_f * (alg_max_princ_strain(E_e_function) - s_t50)))
            + 1
        ),
        ufl.sqrt(
            -f_cc_max
            * dt
            / (1 + ufl.exp(c_f * (alg_max_princ_strain(E_e_function) + s_t50)))
            + 1
        ),
    )

    # Incremental growth tensor
    F_g = ufl.as_tensor(((F_gff, 0, 0), (0, F_gcc, 0), (0, 0, F_gcc)))

    # Elastic deformation tensor
    F_e = ufl.variable(F * ufl.inv(F_g_tot_function))

    F_g_tot_expression = dolfinx.fem.Expression(
        F_g * F_g_tot_function, tensor_space.element.interpolation_points()
    )
    E_e_expression = dolfinx.fem.Expression(
        0.5 * (F_e.T * F_e - ufl.Identity(3)),
        tensor_space.element.interpolation_points(),
    )

    return F_g_tot_function, F_g_tot_expression, E_e_function, E_e_expression, F_e


def KFR(mesh, F):
    """Initiate first growth tensor"""
    F_g_tot_function, E_e_function, tensor_space = initiate_functions(mesh)

    beta = 0.01
    shom = 0.13

    F_g_i = pow(beta * (ufl.sqrt(2 * E_e_function[0, 0] + 1) - 1 - shom) + 1, 1 / 3)

    F_g = ufl.as_tensor(((F_g_i, 0, 0), (0, F_g_i, 0), (0, 0, F_g_i)))

    F_e = ufl.variable(F * ufl.inv(F_g_tot_function))

    F_g_tot_expression = dolfinx.fem.Expression(
        F_g * F_g_tot_function, tensor_space.element.interpolation_points()
    )
    E_e_expression = dolfinx.fem.Expression(
        0.5 * (F_e.T * F_e - ufl.Identity(3)),
        tensor_space.element.interpolation_points(),
    )

    return F_g_tot_function, F_g_tot_expression, E_e_function, E_e_expression, F_e


def KUR(mesh, F):
    """Initiate first growth tensor"""
    F_g_tot_function, E_e_function, tensor_space = initiate_functions(mesh)

    beta = 0.01

    F_g_f = pow(beta * (ufl.sqrt(2 * E_e_function[0, 0] + 1) - 1 - 0.0) + 1, 1 / 3)
    F_g_r = pow(beta * (ufl.sqrt(2 * E_e_function[0, 0] + 1) - 1 - 0.0) + 1, 1 / 3)
    F_g_c = pow(beta * (ufl.sqrt(2 * E_e_function[0, 0] + 1) - 1 - 0.0) + 1, 1 / 3)

    F_g = ufl.as_tensor(((F_g_f, 0, 0), (0, F_g_r, 0), (0, 0, F_g_c)))

    F_e = ufl.variable(F * ufl.inv(F_g))

    F_g_tot_expression = dolfinx.fem.Expression(
        F_g * F_g_tot_function, tensor_space.element.interpolation_points()
    )
    E_e_expression = dolfinx.fem.Expression(
        0.5 * (F_e.T * F_e - ufl.Identity(3)),
        tensor_space.element.interpolation_points(),
    )

    return F_g_tot_function, F_g_tot_expression, E_e_function, E_e_expression, F_e


def GEG(mesh, F):
    F_g_prev_function, F_e_function, tensor_space = initiate_functions(mesh)

    F_gmax = 1.5
    recip_tau = 0.01
    gamma = 2.0
    lambda_crit = 1.01

    F_g_elem = (
        recip_tau
        * pow((F_gmax - F_g_prev_function[0, 0]) / (F_gmax - 1), gamma)
        * (F_e_function[0, 0] - lambda_crit)
        + F_g_prev_function[0, 0]
    )

    F_g = ufl.as_tensor(((F_g_elem, 0, 0), (0, 1, 0), (0, 0, 1)))

    F_e = ufl.variable(F * ufl.inv(F_g))

    F_g_prev_expression = dolfinx.fem.Expression(
        F_g, tensor_space.element.interpolation_points()
    )
    F_e_expression = dolfinx.fem.Expression(
        F_e, tensor_space.element.interpolation_points()
    )

    return F_g_prev_function, F_g_prev_expression, F_e_function, F_e_expression, F_e


def GCG(mesh, F):
    F_g_prev_function, mandel_stress_function, tensor_space = initiate_functions(mesh)

    F_gmax = 1.2
    recip_tau = 0.0001
    gamma = 2.0
    p_crit = 0.12
    F_g_elem = (
        recip_tau
        * pow((F_gmax - F_g_prev_function[1, 1]) / (F_gmax - 1), gamma)
        * (ufl.tr(mandel_stress_function) - p_crit)
        + F_g_prev_function[1, 1]
    )

    F_g = ufl.as_tensor(((1, 0, 0), (0, F_g_elem, 0), (0, 0, 1)))

    F_e = ufl.variable(F * ufl.inv(F_g))

    F_g_prev_expression = dolfinx.fem.Expression(
        F_g, tensor_space.element.interpolation_points()
    )

    return F_g_prev_function, F_g_prev_expression, mandel_stress_function, F_e


def LT2(mesh, F):
    F_g_tot_function, sigma_function, tensor_space = initiate_functions(mesh)

    sigma_seta = 30  # 700
    sigma_setp = 3  # 9
    T_recip = 0.0001

    F_grr = T_recip * (sigma_function[0, 0] - sigma_seta) / sigma_seta + 1
    F_gff = T_recip * (sigma_function[1, 1] - sigma_setp) / sigma_setp + 1

    F_g = ufl.as_tensor(((F_gff, 0, 0), (0, F_grr, 0), (0, 0, 1)))

    F_e = ufl.variable(F * ufl.inv(F_g_tot_function))

    F_g_tot_expression = dolfinx.fem.Expression(
        F_g * F_g_tot_function, tensor_space.element.interpolation_points()
    )

    return F_g_tot_function, F_g_tot_expression, sigma_function, F_e


# THIS ONE ISNT IN USE
def LnT(mesh, F):
    F_g_prev_function, sigma_function, tensor_space = initiate_functions(mesh)
    D = 1
    F_grr = F_g_prev_function[0, 0] / (
        1 - D * F_g_prev_function[0, 0] * sigma_function[0, 0]
    )
    F_gff = F_g_prev_function[1, 1] / (
        1 - D * F_g_prev_function[1, 1] * sigma_function[1, 1]
    )
    F_gcc = F_g_prev_function[2, 2] / (
        1 - D * F_g_prev_function[2, 2] * sigma_function[2, 2]
    )

    F_g = ufl.as_tensor(((F_grr, 0, 0), (0, F_gff, 0), (0, 0, F_gcc)))

    F_e = ufl.variable(F * ufl.inv(F_g))

    F_g_prev_expression = dolfinx.fem.Expression(
        F_g, tensor_space.element.interpolation_points()
    )

    return F_g_prev_function, F_g_prev_expression, sigma_function, F_e, F_g


def ART(mesh, F):
    """Initiate first growth tensor"""
    tensor_space = dolfinx.fem.functionspace(
        mesh,
        basix.ufl.element(
            family="CG", cell=str(mesh.ufl_cell()), degree=1, shape=(3, 3)
        ),
    )
    X = ufl.SpatialCoordinate(mesh)  # get Identity without it being a constant
    Identity = ufl.variable(ufl.grad(X))
    Identity_expression = dolfinx.fem.Expression(
        Identity, tensor_space.element.interpolation_points()
    )

    F_g_tot_function = dolfinx.fem.Function(tensor_space)
    F_g_tot_function.interpolate(Identity_expression)
    F_e_function = dolfinx.fem.Function(
        tensor_space
    )  # E_e_function.interpolate(Identity_expression)
    L_function = dolfinx.fem.Function(tensor_space)
    L_function.interpolate(Identity_expression)

    L_s_max_adapt = 1.0
    L_s_min_adapt = 1.0
    a = 1.0
    b = 1.0

    F_gff = L_s_min_adapt / L_s_max_adapt
    F_grr = ufl.sqrt(
        (1 + a) / 1 + a * ufl.exp(b * L_s_max_adapt - L_function * F_e[1, 1])
    )
    F_gcc = F_gff

    F_g = ufl.as_tensor(((F_grr, 0, 0), (0, F_gff, 0), (0, 0, F_gcc)))

    F_e = ufl.variable(F * ufl.inv(F_g_tot_function))

    L_expression = dolfinx.fem.Expression(
        L_function * F_e, tensor_space.element.interpolation_points()
    )

    F_g_tot_expression = dolfinx.fem.Expression(
        F_g * F_g_tot_function, tensor_space.element.interpolation_points()
    )

    return F_g_tot_function, F_g_tot_expression, L_function, L_expression, F_e
