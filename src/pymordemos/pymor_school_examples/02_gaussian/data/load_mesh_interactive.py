#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dolfin as df

mesh = df.Mesh("gaussian.xml")
subdomains = df.MeshFunction("size_t", mesh, "gaussian_physical_region.xml")
boundaries = df.MeshFunction("size_t", mesh, "gaussian_facet_region.xml")

V1 = df.FunctionSpace(mesh, "CG", 1)
V2 = df.FunctionSpace(mesh, "CG", 2)

QE1 = df.FiniteElement("Quadrature", mesh.ufl_cell(), degree=1, quad_scheme="default")
QE2 = df.FiniteElement("Quadrature", mesh.ufl_cell(), degree=2, quad_scheme="default")

Q1 = df.FunctionSpace(mesh, QE1)
Q2 = df.FunctionSpace(mesh, QE2)


summary = f"""Mesh data:
    cells in mesh:      {mesh.num_cells()}
    dofs for V1:        {V1.dim()}
    dofs for V2:        {V2.dim()}
    dofs for Q1:        {Q1.dim()}
    dofs for Q2:        {Q2.dim()}\n"""
print(summary)

__import__('IPython').embed()
