# Copyright 2021 SAMURAI TEAM. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib import animation
from matplotlib import rc
import argparse

def read_mesh(filename, ite=None):
    return h5py.File(filename + '.h5', 'r')['mesh']

def scatter_plot(ax, points):
    return ax.scatter(points[:, 0], points[:, 1], marker='+')

def scatter_update(scatter, points):
    scatter.set_offsets(points[:, :2])

def line_plot(ax, x_polynom, y_polynom, x_polynom_reinit, y_polynom_reinit, x_pI_liquid, y_pI_liquid, x_pI_gas, y_pI_gas):
    #Plot results
    plot_polynom        = ax.plot(x_polynom, y_polynom, 'k-', linewidth=1, markersize=4, alpha=1)[0]
    plot_polynom_reinit = ax.plot(x_polynom_reinit, y_polynom_reinit, 'bx', linewidth=1, markersize=6, alpha=1, markevery=32)[0]
    plot_pI_liquid      = ax.plot(x_pI_liquid, y_pI_liquid, 'r-', linewidth=1, markersize=4, alpha=1)[0]
    plot_pI_gas         = ax.plot(x_pI_gas, y_pI_gas, 'go', linewidth=1, markersize=4, alpha=1, markevery=24)[0]

    #Read and plot the analytical results
    if args.analytical is not None:
        if args.column_analytical is None:
            sys.exit("Unknown column to be read for the analytical solution")
        data_analytical = np.genfromtxt(args.analytical)
        plot_analytical = ax.plot(data_analytical[:,0], data_analytical[:,args.column_analytical], 'k-', linewidth=1.5, markersize=3, alpha=0.5)[0]

    #Read and plot the reference results
    if args.reference is not None:
        if args.column_reference is None:
            sys.exit("Unknown column to be read for the reference solution")
        data_ref = np.genfromtxt(args.reference)
        plot_ref = ax.plot(data_ref[:,0], data_ref[:,args.column_reference], 'b-', linewidth=1.5, markersize=3, alpha=0.5)[0]

    #Add legend
    if args.analytical is not None:
        if args.reference is not None:
            ax.legend([plot_polynom, plot_polynom_reinit, plot_pI_liquid, plot_pI_gas, plot_analytical, plot_ref], \
                      ['Polynomial method', 'Polynomial method with reinit step', 'pI liquid', 'pI gas', 'Reference results'], fontsize="20", loc="best")
        else:
            ax.legend([plot_polynom, plot_polynom_reinit, plot_pI_liquid, plot_pI_gas, plot_analytical], \
                      ['Polynomial method', 'Polynomial method with reinit step', 'pI liquid', 'pI gas', 'Analytical (5 equations)'], fontsize="20", loc="best")
    elif args.reference is not None:
        ax.legend([plot_polynom, plot_polynom_reinit, plot_pI_liquid, plot_pI_gas, plot_ref], \
                  ['Polynomial method', 'Polynomial method with reinit step', 'pI liquid', 'pI gas', 'Reference results'], fontsize="20", loc="best")
    else:
        ax.legend([plot_polynom, plot_polynom_reinit, plot_pI_liquid, plot_pI_gas], \
                  ['Polynomial method', 'Polynomial method with reinit step', 'pI liquid', 'pI gas'], fontsize="20", loc="best")

    return plot_polynom, plot_polynom_reinit, plot_pI_liquid, plot_pI_gas

def line_update(lines, x_polynom, y_polynom, x_polynom_reinit, y_polynom_reinit, x_pI_liquid, y_pI_liquid, x_pI_gas, y_pI_gas):
    lines[1].set_data(x_polynom, y_polynom)
    lines[2].set_data(x_polynom_reinit, y_polynom_reinit)
    lines[3].set_data(x_pI_liquid, y_pI_liquid)
    lines[4].set_data(x_pI_gas, y_pI_gas)

class Plot:
    def __init__(self, filename_polynom, filename_polynom_reinit, filename_pI_liquid, filename_pI_gas):
        self.fig = plt.figure()
        self.artists = []
        self.ax = []

        mesh_polynom = read_mesh(filename_polynom)
        if args.field is None:
            ax = plt.subplot(111)
            self.plot(ax, mesh_polynom)
            ax.set_title("Mesh")
            self.ax = [ax]
        else:
            mesh_polynom_reinit = read_mesh(filename_polynom_reinit)
            mesh_pI_liquid      = read_mesh(filename_pI_liquid)
            mesh_pI_gas         = read_mesh(filename_pI_gas)
            for i, f in enumerate(args.field):
                ax = plt.subplot(1, len(args.field), i + 1)
                self.plot(ax, mesh_polynom, mesh_polynom_reinit, mesh_pI_liquid, mesh_pI_gas, f)
                ax.set_title(f)

    def plot(self, ax, mesh_polynom, mesh_polynom_reinit=None, mesh_pI_liquid=None, mesh_pI_gas=None, field=None, init=True):
        points_polynom       = mesh_polynom['points']
        connectivity_polynom = mesh_polynom['connectivity']

        segments_polynom = np.zeros((connectivity_polynom.shape[0], 2, 2))
        segments_polynom[:, :, 0] = points_polynom[:][connectivity_polynom[:]][:, :, 0]

        if field is None:
            segments_polynom[:, :, 1] = 0
            if init:
                self.artists.append(scatter_plot(ax, points))
                self.lc = mc.LineCollection(segments_polynom, colors='b', linewidths=2)
                self.lines = ax.add_collection(self.lc)
            else:
                scatter_update(self.artists[self.index], points)
                self.index += 1
                # self.lc.set_array(segments)
        else:
            data_polynom    = mesh_polynom['fields'][field][:]
            centers_polynom = 0.5*(segments_polynom[:, 0, 0] + segments_polynom[:, 1, 0])
            segments_polynom[:, :, 1] = data_polynom[:, np.newaxis]
            # ax.scatter(centers, data, marker='+')
            index_polynom = np.argsort(centers_polynom)

            points_polynom_reinit       = mesh_polynom_reinit['points']
            connectivity_polynom_reinit = mesh_polynom_reinit['connectivity']
            segments_polynom_reinit     = np.zeros((connectivity_polynom_reinit.shape[0], 2, 2))
            segments_polynom_reinit[:, :, 0] = points_polynom_reinit[:][connectivity_polynom_reinit[:]][:, :, 0]
            data_polynom_reinit    = mesh_polynom_reinit['fields'][field][:]
            centers_polynom_reinit = 0.5*(segments_polynom_reinit[:, 0, 0] + segments_polynom_reinit[:, 1, 0])
            segments_polynom_reinit[:, :, 1] = data_polynom_reinit[:, np.newaxis]
            index_polynom_reinit = np.argsort(centers_polynom_reinit)

            points_pI_liquid       = mesh_pI_liquid['points']
            connectivity_pI_liquid = mesh_pI_liquid['connectivity']
            segments_pI_liquid     = np.zeros((connectivity_pI_liquid.shape[0], 2, 2))
            segments_pI_liquid[:, :, 0] = points_pI_liquid[:][connectivity_pI_liquid[:]][:, :, 0]
            data_pI_liquid     = mesh_pI_liquid['fields'][field][:]
            centers_pI_liquid  = .5*(segments_pI_liquid[:, 0, 0] + segments_pI_liquid[:, 1, 0])
            segments_pI_liquid[:, :, 1] = data_pI_liquid[:, np.newaxis]
            index_pI_liquid = np.argsort(centers_pI_liquid)

            points_pI_gas       = mesh_pI_gas['points']
            connectivity_pI_gas = mesh_pI_gas['connectivity']
            segments_pI_gas     = np.zeros((connectivity_pI_gas.shape[0], 2, 2))
            segments_pI_gas[:, :, 0] = points_pI_gas[:][connectivity_pI_gas[:]][:, :, 0]
            data_pI_gas    = mesh_pI_gas['fields'][field][:]
            centers_pI_gas = .5*(segments_pI_gas[:, 0, 0] + segments_pI_gas[:, 1, 0])
            segments_pI_gas[:, :, 1] = data_pI_gas[:, np.newaxis]
            index_pI_gas = np.argsort(centers_pI_gas)
            if init:
                self.artists.append(line_plot(ax, centers_polynom[index_polynom], data_polynom[index_polynom], \
						                          centers_polynom_reinit[index_polynom_reinit], data_polynom_reinit[index_polynom_reinit], \
						                          centers_pI_liquid[index_pI_liquid], data_pI_liquid[index_pI_liquid], \
                                                  centers_pI_gas[index_pI_gas], data_pI_gas[index_pI_gas]))
            else:
                line_update(self.artists[self.index], centers_polynom[index_polynom], data_polynom[index_polynom], \
                                                      centers_polynom_reinit[index_polynom_reinit], data_polynom_reinit[index_polynom_reinit], \
                                                      centers_pI_liquid[index_pI_liquid], data_pI_liquid[index_pI_liquid], \
                                                      centers_pI_gas[index_pI_gas], data_pI_gas[index_pI_gas])
                self.index += 1

        for aax in self.ax:
            aax.relim()
            aax.autoscale_view()

    def update(self, filename_polynom, filename_polynom_reinit, filename_pI_liquid, filename_pI_gas):
        mesh_polynom = read_mesh(filename_polynom)
        self.index = 0
        if args.field is None:
            self.plot(None, mesh_polynom, init=False)
        else:
            mesh_polynom_reinit = read_mesh(filename_polynom_reinit)
            mesh_pI_liquid      = read_mesh(filename_pI_liquid)
            mesh_pI_gas         = read_mesh(filename_pI_gas)

            for i, f in enumerate(args.field):
                self.plot(None, mesh_polynom, mesh_polynom_reinit, mesh_pI_liquid, mesh_pI_gas, f, init=False)

    def get_artist(self):
        return self.artists

parser = argparse.ArgumentParser(description='Plot 1d mesh and field from samurai simulations.')
parser.add_argument('filename_polynom', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('filename_polynom_reinit', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('filename_pI_liquid', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('filename_pI_gas', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('--field', nargs="+", type=str, required=False, help='list of fields to plot')
parser.add_argument('--start', type=int, required=False, default=0, help='iteration start')
parser.add_argument('--end', type=int, required=False, default=None, help='iteration end')
parser.add_argument('--save', type=str, required=False, help='output file')
parser.add_argument('--wait', type=int, default=200, required=False, help='time between two plot in ms')
parser.add_argument('--analytical', type=str, required=False, help='analytical solution 5 equations model')
parser.add_argument('--column_analytical', type=int, required=False, help='variable of analytical solution 5 equations model')
parser.add_argument('--reference', type=str, required=False, help='reference results')
parser.add_argument('--column_reference', type=int, required=False, help='variable of reference results')
args = parser.parse_args()

if args.end is None:
    Plot(args.filename_polynom, args.filename_polynom_reinit, args.filename_pI_liquid, args.filename_pI_gas)
else:
    p = Plot(f"{args.filename_polynom}{args.start}", f"{args.filename_polynom_reinit}{args.start}", \
             f"{args.filename_pI_liquid}{args.start}", f"{args.filename_pI_gas}{args.start}")
    def animate(i):
        p.fig.suptitle(f"iteration {i + args.start}")
        p.update(f"{args.filename_polynom}{i + args.start}", f"{args.filename_polynom_reinit}{args.start}", \
                 f"{args.filename_pI_liquid}{args.start}", f"{args.filename_pI_gas}{args.start}")
        return p.get_artist()
    ani = animation.FuncAnimation(p.fig, animate, frames=args.end-args.start, interval=args.wait, repeat=True)

if args.save:
    if args.end is None:
        plt.savefig(args.save + '.png', dpi=300)
    else:
        writermp4 = animation.FFMpegWriter(fps=1)
        ani.save(args.save + '.mp4', dpi=300)
else:
    plt.show()
