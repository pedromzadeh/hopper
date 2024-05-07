import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size


class RemainderFixed(axes_size.Scaled):
    def __init__(self, xsizes, ysizes, divider):
        self.xsizes = xsizes
        self.ysizes = ysizes
        self.div = divider

    def get_size(self, renderer):
        xrel, xabs = axes_size.AddList(self.xsizes).get_size(renderer)
        yrel, yabs = axes_size.AddList(self.ysizes).get_size(renderer)
        bb = Bbox.from_bounds(*self.div.get_position()).transformed(
            self.div._fig.transFigure
        )
        w = bb.width / self.div._fig.dpi - xabs
        h = bb.height / self.div._fig.dpi - yabs
        return 0, min([w, h])


def make_square_axes_with_colorbar(ax, size=0.1, pad=0.1):
    """Make an axes square, add a colorbar axes next to it,
    Parameters: size: Size of colorbar axes in inches
                pad : Padding between axes and cbar in inches
    Returns: colorbar axes
    """
    divider = make_axes_locatable(ax)
    margin_size = axes_size.Fixed(size)
    pad_size = axes_size.Fixed(pad)
    xsizes = [pad_size, margin_size]
    yhax = divider.append_axes("right", size=margin_size, pad=pad_size)
    divider.set_horizontal([RemainderFixed(xsizes, [], divider)] + xsizes)
    divider.set_vertical([RemainderFixed(xsizes, [], divider)])
    return yhax


class Figure:

    @classmethod
    def view_pol_field(cls, cell, chi, dpi, cbar=True, zoom_in=True, path=None):
        phi = cell.phi
        p_field = cell.p_field
        L_box = cell.simbox.L_box

        p_field_masked = np.ones(p_field.shape) * np.nan
        i, j = np.where(phi >= 0.5)
        p_field_masked[i, j] = p_field[i, j]

        fig, ax = plt.subplots(figsize=(3, 3), dpi=dpi)
        img = plt.imshow(
            p_field_masked, extent=[0, L_box, 0, L_box], origin="lower", cmap="coolwarm"
        )
        cax = make_square_axes_with_colorbar(ax, size=0.15, pad=0.1)

        if cbar:
            colorbar = fig.colorbar(img, cax=cax, format=FormatStrFormatter("%.2f"))
            colorbar.set_label("Polarization field " + r"$\mathbb{P}$")

        ax.contour(
            phi,
            levels=[0.5],
            extent=[0, L_box, 0, L_box],
            origin="lower",
            linewidths=[2],
            colors=["black"],
        )
        ax.contour(
            chi,
            levels=[0.5],
            extent=[0, L_box, 0, L_box],
            origin="lower",
            linewidths=[3],
            colors=["black"],
        )

        if zoom_in:
            ax.set_xlim([15, 35])
            ax.set_ylim([15, 35])
            ax.set_axis("off")
        else:
            ax.set_xlim([0, 50])
            ax.set_ylim([0, 50])

        if path is not None:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

    @classmethod
    def plot_vertical_snapshot(cls, cell, chi, dpi, path=None):
        phi = cell.phi.T
        p_field = cell.p_field.T
        L_box = cell.simbox.L_box
        chi = chi.T

        p_field_masked = np.ones(p_field.shape) * np.nan
        i, j = np.where(phi >= 0.5)
        p_field_masked[i, j] = p_field[i, j]

        plt.figure(figsize=(3, 3), dpi=dpi)
        plt.imshow(
            p_field_masked, extent=[0, L_box, 0, L_box], origin="lower", cmap="coolwarm"
        )
        # cbar = plt.colorbar()
        # cbar = plt.colorbar(
        #     format=FuncFormatter(lambda x, pos: "{:.2f}".format(x)), pad=0.2, shrink=0.8
        # )
        # cbar.set_label(r"$\mathbb{P}\equiv \phi \rho$")
        plt.contour(
            phi,
            levels=[0.5],
            extent=[0, L_box, 0, L_box],
            origin="lower",
            linewidths=[2],
            colors=["black"],
        )
        plt.contour(
            chi,
            levels=[0.5],
            extent=[0, L_box, 0, L_box],
            origin="lower",
            linewidths=[3],
            colors=["black"],
        )

        plt.xlim([10, 40])
        plt.ylim([10, 40])
        plt.axis("off")
        plt.tight_layout()

        if path is not None:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()
