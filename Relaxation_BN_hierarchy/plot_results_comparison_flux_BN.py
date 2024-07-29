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

def line_plot(ax, x_Rusanov, y_Rusanov, x_Rusanov_BR, y_Rusanov_BR, x_Suliciu, y_Suliciu):
    #Plot results
    plot_Rusanov    = ax.plot(x_Rusanov, y_Rusanov, color='orange', linewidth=1, markersize=4, alpha=1)[0]
    plot_Rusanov_BR = ax.plot(x_Rusanov_BR, y_Rusanov_BR, 'r-', linewidth=1, markersize=4, alpha=1)[0]
    plot_Suliciu    = ax.plot(x_Suliciu, y_Suliciu, 'go', linewidth=1, markersize=4, alpha=1, markevery=32)[0]

    #Read and plot the analytical results
    if args.analytical is not None:
        if args.column_analytical is None:
            sys.exit("Unknown column to be read for the analytical solution")
        data_analytical = np.genfromtxt(args.analytical)
        plot_analytical = ax.plot(data_analytical[:,0], data_analytical[:,args.column_analytical], 'k-', linewidth=1.5, markersize=3, alpha=1)[0]

    #Read and plot the reference results
    if args.reference is not None:
        if args.column_reference is None:
            sys.exit("Unknown column to be read for the reference solution")
        data_ref = np.genfromtxt(args.reference)
        plot_ref = ax.plot(data_ref[:,0], data_ref[:,args.column_reference], 'b-', linewidth=1.5, markersize=3, alpha=1)[0]

    #Add legend
    if args.analytical is not None:
        if args.reference is not None:
            ax.legend([plot_Rusanov, plot_Rusanov_BR, plot_Suliciu, plot_analytical, plot_ref], \
                      ['Rusanov', 'Rusanov + BR', 'Suliciu', 'Analytical (5 equations)', 'Reference results'], fontsize="20", loc="best")
        else:
            ax.legend([plot_Rusanov, plot_Rusanov_BR, plot_Suliciu, plot_analytical], \
                      ['Rusanov', 'Rusanov + BR', 'Suliciu', 'Analytical (5 equations)'], fontsize="20", loc="best")
    elif args.reference is not None:
        ax.legend([plot_Rusanov, plot_Rusanov_BR, plot_Suliciu, plot_ref], \
                  ['Rusanov', 'Rusanov + BR', 'Suliciu', 'Reference results'], fontsize="20", loc="best")
    else:
        ax.legend([plot_Rusanov, plot_Rusanov_BR, plot_Suliciu], ['Rusanov', 'Rusanov + BR', 'Suliciu'], fontsize="20", loc="best")

    return plot_Rusanov, plot_Rusanov_BR, plot_Suliciu

def line_update(lines, x_Rusanov, y_Rusanov, x_Rusanov_BR, y_Rusanov_BR, x_Suliciu, y_Suliciu):
    lines[1].set_data(x_Rusanov, y_Rusanov)
    lines[2].set_data(x_Rusanov_BR, y_Rusanov_BR)
    lines[3].set_data(x_Suliciu, y_Suliciu)

class Plot:
    def __init__(self, filename_Rusanov, filename_Rusanov_BR, filename_Suliciu):
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
            mesh_Rusanov_BR = read_mesh(filename_Rusanov_BR)
            mesh_Suliciu    = read_mesh(filename_Suliciu)
            for i, f in enumerate(args.field):
                ax = plt.subplot(1, len(args.field), i + 1)
                self.plot(ax, mesh_Rusanov, mesh_Rusanov_BR, mesh_Suliciu, f)
                ax.set_title(f)

    def plot(self, ax, mesh_Rusanov, mesh_Rusanov_BR=None, mesh_Suliciu=None, field=None, init=True):
        points_Rusanov       = mesh_Rusanov['points']
        connectivity_Rusanov = mesh_Rusanov['connectivity']

        segments_Rusanov = np.zeros((connectivity_Rusanov.shape[0], 2, 2))
        segments_Rusanov[:, :, 0] = points_Rusanov[:][connectivity_Rusanov[:]][:, :, 0]

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
            data_Rusanov    = mesh_Rusanov['fields'][field][:]
            centers_Rusanov = 0.5*(segments_Rusanov[:, 0, 0] + segments_Rusanov[:, 1, 0])
            segments_Rusanov[:, :, 1] = data_Rusanov[:, np.newaxis]
            # ax.scatter(centers, data, marker='+')
            index_Rusanov = np.argsort(centers_Rusanov)

            points_Rusanov_BR       = mesh_Rusanov_BR['points']
            connectivity_Rusanov_BR = mesh_Rusanov_BR['connectivity']
            segments_Rusanov_BR     = np.zeros((connectivity_Rusanov_BR.shape[0], 2, 2))
            segments_Rusanov_BR[:, :, 0] = points_Rusanov_BR[:][connectivity_Rusanov_BR[:]][:, :, 0]
            data_Rusanov_BR    = mesh_Rusanov_BR['fields'][field][:]
            centers_Rusanov_BR = 0.5*(segments_Rusanov_BR[:, 0, 0] + segments_Rusanov_BR[:, 1, 0])
            segments_Rusanov_BR[:, :, 1] = data_Rusanov_BR[:, np.newaxis]
            index_Rusanov_BR = np.argsort(centers_Rusanov_BR)

            points_Suliciu       = mesh_Suliciu['points']
            connectivity_Suliciu = mesh_Suliciu['connectivity']
            segments_Suliciu     = np.zeros((connectivity_Suliciu.shape[0], 2, 2))
            segments_Suliciu[:, :, 0] = points_Suliciu[:][connectivity_Suliciu[:]][:, :, 0]
            data_Suliciu     = mesh_Suliciu['fields'][field][:]
            centers_Suliciu  = .5*(segments_Suliciu[:, 0, 0] + segments_Suliciu[:, 1, 0])
            segments_Suliciu[:, :, 1] = data_Suliciu[:, np.newaxis]
            index_Suliciu = np.argsort(centers_Suliciu)
            if init:
                self.artists.append(line_plot(ax, centers_Rusanov[index_Rusanov], data_Rusanov[index_Rusanov], \
						                          centers_Rusanov_BR[index_Rusanov_BR], data_Rusanov_BR[index_Rusanov_BR], \
						                          centers_Suliciu[index_Suliciu], data_Suliciu[index_Suliciu]))
            else:
                line_update(self.artists[self.index], centers_Rusanov[index_Rusanov], data_Rusanov[index_Rusanov], \
                                                      centers_Rusanov_BR[index_Rusanov_BR], data_Rusanov_BR[index_Rusanov_BR], \
                                                      centers_Suliciu[index_Suliciu], data_Suliciu[index_Suliciu])
                self.index += 1

        for aax in self.ax:
            aax.relim()
            aax.autoscale_view()

    def update(self, filename_Rusanov, filename_Rusanov_BR, filename_Suliciu):
        mesh_Rusanov = read_mesh(filename_Rusanov)
        self.index = 0
        if args.field is None:
            self.plot(None, mesh_Rusanov, init=False)
        else:
            mesh_Rusanov_BR    = read_mesh(filename_Rusanov_BR)
            mesh_Suliciu = read_mesh(filename_Suliciu)

            for i, f in enumerate(args.field):
                self.plot(None, mesh_Rusanov, mesh_Rusanov_BR, mesh_Suliciu, f, init=False)

    def get_artist(self):
        return self.artists

parser = argparse.ArgumentParser(description='Plot 1d mesh and field from samurai simulations.')
parser.add_argument('filename_Rusanov', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('filename_Rusanov_BR', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('filename_Suliciu', type=str, help='hdf5 file to plot without .h5 extension')
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
    Plot(args.filename_Rusanov, args.filename_Rusanov_BR, args.filename_Suliciu)
else:
    p = Plot(f"{args.filename_Rusanov}{args.start}", f"{args.filename_Rusanov_BR}{args.start}", f"{args.filename_Suliciu}{args.start}")
    def animate(i):
        p.fig.suptitle(f"iteration {i + args.start}")
        p.update(f"{args.filename_Rusanov}{i + args.start}", f"{args.filename_Rusanov_BR}{args.start}", f"{args.filename_Suliciu}{args.start}")
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
