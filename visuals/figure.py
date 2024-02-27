import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


class Figure:
    def __init__(self):
        pass

    @classmethod
    def view_simbox(cls, cell, chi, path):
        L = cell.simbox.L_box
        plt.imshow(
            cell.phi, cmap="Greys", origin="lower", extent=[0, L, 0, L], alpha=0.5
        )

        plt.contour(cell.phi, levels=[0.5], extent=[0, L, 0, L], colors="black")

        xcm, ycm = cell.cm[1]
        px, py = [np.cos(cell.theta), np.sin(cell.theta)]
        vx, vy = cell.v_cm
        rx, ry = cell.r_CR

        plt.quiver(
            xcm,
            ycm,
            px,
            py,
            angles="xy",
            scale_units="xy",
            color="blue",
            label="Polarity",
            alpha=0.7,
        )

        # plt.quiver(
        #     xcm,
        #     ycm,
        #     vx,
        #     vy,
        #     angles="xy",
        #     scale_units="xy",
        #     color="red",
        #     label="CM Velocity",
        #     alpha=0.7,
        # )

        if rx != 0 and ry != 0:
            plt.quiver(
                xcm,
                ycm,
                rx,
                ry,
                angles="xy",
                scale_units="xy",
                color="black",
                label=r"$r_{{CR}}$",
                alpha=0.7,
            )

        plt.contour(chi, levels=[0.5], extent=[0, L, 0, L])
        plt.axis("equal")
        plt.savefig(path)
        plt.close()

    @classmethod
    def view_pol_field(cls, cell, chi, dpi, zoom_in=True, path=None):
        phi = cell.phi
        p_field = cell.p_field
        L_box = cell.simbox.L_box

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

        if zoom_in:
            plt.xlim([15, 35])
            plt.ylim([15, 35])
        # plt.axis("off")

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
