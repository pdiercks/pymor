#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate a RVE mesh

there are 4 types of RVEs, see Schroeder p.20

Usage:
    generate_rve_mesh.py [options] TYPE RADIUS LCAR

Arguments:
    TYPE                The type of the RVE.
    RADIUS              The radius of the inclusion.
    LCAR                Characteristic length of the cells in mesh.

Options:
    -h, --help          Show this message.
    --plot              Plot mesh and subdomains.
    --unit-length=UNIT  Unit length of the cell. [default: 1.0]
    --make-test         Compute volume of the mesh and compare to reference value.
    --to-xdmf           Write mesh, subdomains and boundaries to XDMF.
    --path=PATH         Set the path for option '--to-xdmf'. [default: '']
"""

import sys
import pygmsh
import meshio
from docopt import docopt
from math import sqrt, pi, sin, cos
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt

from fenics_modules.io import MeshConverter, MeshReader

def parse_arguments(args):
    args = docopt(__doc__, args)
    args['TYPE'] = int(args['TYPE'])
    args['RADIUS'] = float(args['RADIUS'])
    args['LCAR'] = float(args['LCAR'])
    args['--unit-length'] = float(args['--unit-length'])
    assert args['TYPE'] in (1, 2, 3, 4)
    assert isinstance(eval(args['--path']), str)
    args['--path'] = eval(args['--path'])
    return args

def main(args):
    args = parse_arguments(args)

    geom = pygmsh.built_in.Geometry()

    lcar = args['LCAR']
    R = args['RADIUS']
    a = args['--unit-length']

    # ### define unit vectors of the RVE according to type
    if args['TYPE'] in (1, 2, 3):
        if args['TYPE'] == 2:
            a *= sqrt(2)
            lcar *= sqrt(2)
        a1 = np.array([a, 0.0, 0.0])
        a2 = np.array([0.0, a, 0.0])
    elif args['TYPE'] == 4:
        a1 = np.array([a, 0.0, 0.0])
        a2 = np.array([a, a, 0.0])
    else:
        assert False # this should never happen

    def add_circle_point(self, center, phi, l=lcar):
        assert isinstance(self, pygmsh.built_in.Geometry)
        assert isinstance(center, pygmsh.built_in.point.Point)
        assert isinstance(phi, float)
        return self.add_point(center.x + np.array([cos(phi)*R, sin(phi)*R, 0.0]),
                             lcar=l)

    if args['TYPE'] in (2, 3, 4):
        # create all points
        p0 = geom.add_point(np.array([0.0, 0.0, 0.0]), lcar=lcar)
        p1 = geom.add_point(a1, lcar=lcar)
        p2 = geom.add_point(a1 + a2, lcar=lcar)
        p3 = geom.add_point(a2, lcar=lcar)

        # start and end points for circle arcs
        ps0 = add_circle_point(geom, p0, 0.0)
        pe1 = add_circle_point(geom, p1, pi)
        ps2 = add_circle_point(geom, p2, pi)
        pe3 = add_circle_point(geom, p3, 0.0)
        if args['TYPE'] in (2, 3):
            pe0 = add_circle_point(geom, p0, pi/2)
            ps1 = add_circle_point(geom, p1, pi/2)
            pe2 = add_circle_point(geom, p2, 3*pi/2)
            ps3 = add_circle_point(geom, p3, 3*pi/2)
        elif args['TYPE'] == 4:
            pe0 = add_circle_point(geom, p0, pi/4)
            ps1 = add_circle_point(geom, p1, pi/4)
            pe2 = add_circle_point(geom, p2, 5*pi/4)
            ps3 = add_circle_point(geom, p3, 5*pi/4)
        else:
            assert False

        # create circle arcs, start, center, end
        a0 = geom.add_circle_arc(ps0, p0, pe0)
        a1 = geom.add_circle_arc(ps1, p1, pe1)
        a2 = geom.add_circle_arc(ps2, p2, pe2)
        a3 = geom.add_circle_arc(ps3, p3, pe3)
        arcs = (a0, a1, a2, a3)

        # create tuple of points for matrix lines - clockwise
        mpoints = (pe0, ps3, pe3, ps2, pe2, ps1, pe1, ps0)
        matrix_lines = [geom.add_line(mpoints[2*k], mpoints[2*k + 1]) for k in range(4)]

        # add circle arcs
        for (index, arc) in enumerate(arcs):
            matrix_lines.insert(2*index, arc)

        ipoints = (pe0, p0, ps0, pe3, p3, ps3, pe2, p2, ps2, pe1, p1, ps1)
        inclusion_lines = []
        for k in range(4):
            inclusion_lines.append(geom.add_line(ipoints[3*k], ipoints[3*k + 1]))
            inclusion_lines.append(geom.add_line(ipoints[3*k + 1], ipoints[3*k + 2]))

        # add circle arcs here as well
        arcs = (a0, a3, a2, a1)
        for (index, arc) in enumerate(arcs):
            i = index + 1
            inclusion_lines.insert(3*i- 1, arc)

        # ### line loops
        inclusion_line_loops = [geom.add_line_loop(inclusion_lines[j:(j+3)]) for j in (0, 3, 6, 9)]
        ll = geom.add_line_loop(matrix_lines)

        # ### surfaces
        inclusion_surface = [geom.add_plane_surface(lineloop) for lineloop in inclusion_line_loops]
        holes = []
        if args['TYPE'] == 2:
            circle = geom.add_circle([a/2, a/2, 0.0], R, lcar)
            holes.append(circle.line_loop)
            inclusion_surface += [circle.plane_surface]
        matrix_surface = geom.add_plane_surface(ll, holes)

        all_lines = set().union(*[matrix_lines, inclusion_lines])
        lines_wo_arcs = list(all_lines.difference(set(arcs)))
        ids = [line.id for line in lines_wo_arcs]
        d = {id_: line for (id_, line) in zip(ids, lines_wo_arcs)}
        for line in lines_wo_arcs:
            assert line is d[line.id]

        geom.add_physical(matrix_surface, label="matrix")
        geom.add_physical(inclusion_surface, label="inclusion")

        bottom_ids = ('l9', 'l7', 'l14')
        right_ids = ('l15', 'l6', 'l12')
        top_ids = ('l10', 'l5', 'l13')
        left_ids = ('l8', 'l4', 'l11')

        bottom = [d[b] for b in bottom_ids]
        right = [d[r] for r in right_ids]
        top = [d[t] for t in top_ids]
        left = [d[l] for l in left_ids]

        geom.add_physical(bottom, label="bottom")
        geom.add_physical(right, label="right")
        geom.add_physical(top, label="top")
        geom.add_physical(left, label="left")

        for i in range(3):
            geom.add_raw_code(f'Periodic Curve {{{bottom_ids[i]}}} = {{{top_ids[i]}}};')
            geom.add_raw_code(f'Periodic Curve {{{right_ids[i]}}} = {{{left_ids[i]}}};')

    elif args['TYPE'] == 1:
        # add circle in the center
        circle = geom.add_circle([a/2, a/2, 0.0], R, lcar)
        poly = geom.add_polygon(
           [[0.0, 0.0, 0.0], a1, a1+a2, a2],
           lcar,
           holes=[circle.line_loop],
        )
        #  geom.set_transfinite_lines(poly.lines, size=31)
        geom.add_physical(poly.surface, label="matrix")
        geom.add_physical(circle.plane_surface, label="inclusion")

        boundaries = ("bottom", "right", "top", "left")
        for (i, boundary) in enumerate(boundaries):
            geom.add_physical(poly.lines[i], label=boundary)

        geom.add_raw_code(f'Periodic Curve {{{poly.lines[0].id}}} = {{{poly.lines[2].id}}};')
        geom.add_raw_code(f'Periodic Curve {{{poly.lines[1].id}}} = {{{poly.lines[3].id}}};')
    else:
        assert False

    m = pygmsh.generate_mesh(geom, prune_z_0=True, geo_filename='rve.geo')

    # ### write to XDMF
    if args['--to-xdmf']:
        mc = MeshConverter(m, filename='rve', path=args['--path'])
        mc.write_out(subdomains=True, boundaries=True)

    if args['--make-test']:
        from helpers import compute_volume
        ref = a ** 2
        assert abs(compute_volume(m) - ref) < 1.0e-2 * ref

    if args['--plot']:
        assert args['--to-xdmf'], "Cannot plot mesh without option '--to-xdmf'."
        mr = MeshReader('rve')
        data = mr.read(subdomains=True)
        mesh = data['mesh']
        subdomains = data['subdomains']
        df.plot(mesh)
        p = df.plot(subdomains)
        plt.colorbar(p)
        plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
