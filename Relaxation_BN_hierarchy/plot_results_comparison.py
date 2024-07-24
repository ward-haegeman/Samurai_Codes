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

def line_plot(ax, x_Rusanov, y_Rusanov, x_HLLC, y_HLLC, x_HLLC_BR, y_HLLC_BR):
    #Plot results
    plot_Rusanov = ax.plot(x_Rusanov, y_Rusanov, color='orange', linewidth=1, markersize=4, alpha=0.5)[0]
    plot_HLLC    = ax.plot(x_HLLC, y_HLLC, 'r-', linewidth=1, markersize=4, alpha=0.5)[0]
    plot_HLLC_BR = ax.plot(x_HLLC_BR, y_HLLC_BR, 'go', linewidth=1, markersize=4, alpha=0.5, markevery=32)[0]

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
            ax.legend([plot_Rusanov, plot_HLLC, plot_HLLC_BR, plot_analytical, plot_ref], \
                      ['Rusanov + BR', 'HLLC (wave propagation)', 'HLLC + BR', 'Analytical (5 equations)', 'Reference results'], fontsize="20", loc="best")
        else:
            ax.legend([plot_Rusanov, plot_HLLC, plot_HLLC_BR, plot_analytical], \
                      ['Rusanov + BR', 'HLLC (wave propagation)', 'HLLC + BR', 'Analytical (5 equations)'], fontsize="20", loc="best")
    elif args.reference is not None:
        ax.legend([plot_Rusanov, plot_HLLC, plot_HLLC_BR, plot_ref], \
                  ['Rusanov + BR', 'HLLC (wave propagation)', 'HLLC + BR', 'Reference results'], fontsize="20", loc="best")
    else:
        ax.legend([plot_Rusanov, plot_HLLC, plot_HLLC_BR], ['Rusanov + BR', 'HLLC (wave propagation)', 'HLLC + BR'], fontsize="20", loc="best")

    return plot_Rusanov, plot_HLLC, plot_HLLC_BR

def line_update(lines, x_Rusanov, y_Rusanov, x_HLLC, y_HLLC, x_HLLC_BR, y_HLLC_BR):
    lines[1].set_data(x_Rusanov, y_Rusanov)
    lines[2].set_data(x_HLLC, y_HLLC)
    lines[3].set_data(x_HLLC_BR, y_HLLC_BR)

class Plot:
    def __init__(self, filename_Rusanov, filename_HLLC, filename_HLLC_BR):
        self.fig = plt.figure()
        self.artists = []
        self.ax = []

        mesh_Rusanov = read_mesh(filename_Rusanov)
        if args.field is None:
            ax = plt.subplot(111)
            self.plot(ax, mesh_Rusanov)
            ax.set_title("Mesh")
            self.ax = [ax]
        else:
            mesh_HLLC = read_mesh(filename_HLLC)
            mesh_HLLC_BR = read_mesh(filename_HLLC_BR)
            for i, f in enumerate(args.field):
                ax = plt.subplot(1, len(args.field), i + 1)
                self.plot(ax, mesh_Rusanov, mesh_HLLC, mesh_HLLC_BR, f)
                ax.set_title(f)

    def plot(self, ax, mesh_Rusanov, mesh_HLLC=None, mesh_HLLC_BR=None, field=None, init=True):
        points = mesh_Rusanov['points']
        connectivity = mesh_Rusanov['connectivity']

        segments_Rusanov = np.zeros((connectivity.shape[0], 2, 2))
        segments_Rusanov[:, :, 0] = points[:][connectivity[:]][:, :, 0]

        if field is None:
            segments_Rusanov[:, :, 1] = 0
            if init:
                self.artists.append(scatter_plot(ax, points))
                self.lc = mc.LineCollection(segments_Rusanov, colors='b', linewidths=2)
                self.lines = ax.add_collection(self.lc)
            else:
                scatter_update(self.artists[self.index], points)
                self.index += 1
                # self.lc.set_array(segments)
        else:
            data_Rusanov = mesh_Rusanov['fields'][field][:]
            centers_Rusanov = .5*(segments_Rusanov[:, 0, 0] + segments_Rusanov[:, 1, 0])
            segments_Rusanov[:, :, 1] = data_Rusanov[:, np.newaxis]
            # ax.scatter(centers, data, marker='+')
            index_Rusanov = np.argsort(centers_Rusanov)

            points_HLLC = mesh_HLLC['points']
            connectivity_HLLC = mesh_HLLC['connectivity']
            segments_HLLC = np.zeros((connectivity_HLLC.shape[0], 2, 2))
            segments_HLLC[:, :, 0] = points[:][connectivity_HLLC[:]][:, :, 0]
            data_HLLC = mesh_HLLC['fields'][field][:]
            centers_HLLC = .5*(segments_HLLC[:, 0, 0] + segments_HLLC[:, 1, 0])
            segments_HLLC[:, :, 1] = data_HLLC[:, np.newaxis]
            index_HLLC = np.argsort(centers_HLLC)

            points_HLLC_BR = mesh_HLLC_BR['points']
            connectivity_HLLC_BR = mesh_HLLC_BR['connectivity']
            segments_HLLC_BR = np.zeros((connectivity_HLLC_BR.shape[0], 2, 2))
            segments_HLLC_BR[:, :, 0] = points[:][connectivity_HLLC_BR[:]][:, :, 0]
            data_HLLC_BR = mesh_HLLC_BR['fields'][field][:]
            centers_HLLC_BR = .5*(segments_HLLC_BR[:, 0, 0] + segments_HLLC_BR[:, 1, 0])
            segments_HLLC_BR[:, :, 1] = data_HLLC_BR[:, np.newaxis]
            index_HLLC_BR = np.argsort(centers_HLLC_BR)
            if init:
                self.artists.append(line_plot(ax, centers_Rusanov[index_Rusanov], data_Rusanov[index_Rusanov], \
						                          centers_HLLC[index_HLLC], data_HLLC[index_HLLC], \
						                          centers_HLLC_BR[index_HLLC_BR], data_HLLC_BR[index_HLLC_BR]))
            else:
                line_update(self.artists[self.index], centers_Rusanov[index_Rusanov], data_Rusanov[index_Rusanov], \
                                                      centers_HLLC[index_HLLC], data_HLLC[index_HLLC], \
                                                      centers_HLLC_BR[index_HLLC_BR], data_HLLC_BR[index_HLLC_BR])
                self.index += 1

        for aax in self.ax:
            aax.relim()
            aax.autoscale_view()

    def update(self, filename_Rusanov, filename_HLLC, filename_HLLC_BR):
        mesh_Rusanov = read_mesh(filename_HLLC)
        self.index = 0
        if args.field is None:
            self.plot(None, mesh_Rusanov, init=False)
        else:
            mesh_HLLC = read_mesh(filename_HLLC)
            mesh_HLLC_BR = read_mesh(filename_HLLC_BR)

            for i, f in enumerate(args.field):
                self.plot(None, mesh_Rusanov, mesh_HLLC, mesh_HLLC_BR, f, init=False)

    def get_artist(self):
        return self.artists

parser = argparse.ArgumentParser(description='Plot 1d mesh and field from samurai simulations.')
parser.add_argument('filename_Rusanov', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('filename_HLLC', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('filename_HLLC_BR', type=str, help='hdf5 file to plot without .h5 extension')
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
    Plot(args.filename_Rusanov, args.filename_HLLC, args.filename_HLLC_BR)
else:
    p = Plot(f"{args.filename_Rusanov}{args.start}", f"{args.filename_HLLC}{args.start}", f"{args.filename_HLLC_BR}{args.start}")
    def animate(i):
        p.fig.suptitle(f"iteration {i + args.start}")
        p.update(f"{args.filename_Rusanov}{i + args.start}", f"{args.filename_HLLC}{args.start}", f"{args.filename_HLLC_BR}{args.start}")
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
