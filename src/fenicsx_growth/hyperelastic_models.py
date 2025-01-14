import ufl

"""Incompressible part"""


def holzapfel(F, f0):
    """

    Declares the strain energy function for a simplified holzapfel formulation.

    Args:
        F - deformation tensor

    Returns:
        psi(F), scalar function

    """

    # "a_s": 2.481,
    # "b_s": 11.120,
    # "a_fs": 0.216,
    # "b_fs": 11.436,

    a = 0.059
    b = 8.023
    a_f = 18.472
    b_f = 16.026

    J = ufl.det(F)
    F_e_bar = F  # *pow(J, -1/3)
    C = F_e_bar.T * F_e_bar

    # C = pow(J, -float(2) / 3) * F.T * F

    e1 = ufl.as_vector([1.0, 0.0, 0.0])
    IIFx = ufl.tr(C)
    I4e1 = ufl.inner(C * f0, f0)

    cond = lambda a: ufl.conditional(a > 0, a, 0)

    W_hat = a / (2 * b) * (ufl.exp(b * (IIFx - 3)) - 1)
    W_f = a_f / (2 * b_f) * (ufl.exp(b_f * cond(I4e1 - 1) ** 2) - 1)

    return W_hat + W_f


def neohookean(w, C):
    """

    Declares the strain energy function for a neohookean (incompressible) material.

    Args:
        F - deformation tensor

    Returns:
        psi(F), scalar function

    """

    I1 = ufl.tr(C)

    return w * (I1 - 3)


def compressible_neohookean(w, F):
    C = F.T * F
    J = ufl.det(F)

    return 0.5 * w[0] * ufl.ln(J) ** 2 + 0.5 * w[1] * (ufl.tr(C) - 3 - 2 * ufl.ln(J))


def mooney_rivlin(w1, w2, C):
    """
    Args:
        w1, w2 - weights
        C - Ratio tensor (Cauchy tensor)

    Returns:
        psi(C), scalar function

    """

    I1 = ufl.tr(C)
    I2 = ufl.tr(ufl.inv(C)) * ufl.det(C)
    return w1 / 2 * (I1 - 3) + w2 / 2 * (I2 - 3)


def fung_demiray(mu, beta, C):
    """
    Args:
        mu - shear modulus
        beta - controls the strain hardening property
        C - Ratio (Cauchy) tensor

    Returns:
        psi(C), scalar function

    """

    I1 = ufl.tr(C)
    return mu / (2 * beta) * (ufl.exp(beta * (I1 - 3)) - 1)


def gent(mu, beta, C):
    """
    Args:
        mu - shear modulus
        beta - controls the strain hardening property
        C - Ratio (Cauchy) tensor

    Returns:
        psi(C), scalar function

    """

    I1 = ufl.tr(C)
    return -mu / (2 * beta) * (ufl.ln(1 - beta * (I1 - 3)))


def guccione(w1, w2, w3, w4, C):
    """
    Args:
        mu - shear modulus
        beta - controls the strain hardening property
        C - Ratio (Cauchy) tensor

    Returns:
        psi(C), scalar function

    """

    E = 1 / 2 * (C - ufl.Identity(3))
    Q = (
        w1 * E[0, 0] ** 2
        + w2 * (E[1, 1] ** 2 + E[2, 2] ** 2 + E[1, 2] * E[2, 1])
        + w3 * (2 * E[0, 1] * E[1, 0] + 2 * E[0, 2] * E[2, 0])
    )

    return 1 / 2 * w4 * (ufl.exp(Q) - 1)


def kerckhoff(w, C):
    E = ufl.variable(0.5 * (C - ufl.Identity(3)))

    # 0.5*(ufl.exp(3*(ufl.tr(E_bar.T*E_bar))) - 1) + 0.01*(ufl.exp(60*E_bar[0,0]) - 1)

    # E = 1/2*(C - ufl.Identity(3))
    W_m = w[0] * (ufl.exp(w[1] * (ufl.tr(E.T * E))) - 1)
    W_f = w[2] * (ufl.exp(w[3] * E[0, 0]) - 1)

    return W_m + W_f


"""Compressible part"""


def comp1(mu_c, J):
    I3 = pow(J, 2)
    return mu_c * (I3 - 1)


def comp2(mu_c, J):
    return mu_c * pow((J - 1), 2)


def comp3(mu_c, J):
    I3 = pow(J, 2)
    return mu_c * pow((I3 - 1), 2)


def comp4(mu_c, J):
    I3 = pow(J, 2)
    return mu_c * ufl.ln(I3)


def comp5(mu_c, J):
    return mu_c * ufl.ln(J)


def comp6(mu_1, mu_2, J):
    I3 = pow(J, 2)
    return -mu_1 / 2 * (I3 - 1) + mu_2 / 4 * (pow((I3 - 1), 2))


def comp7(w1, J):
    I3 = pow(J, 2)
    return w1 * (J * ufl.ln(J) - J + 1) * ufl.Identity(3)


def comp8(w, J):
    return 0.5 * w * (J - 1) * ufl.ln(J)


def main():
    return


if __name__ == "__main__":
    main()
