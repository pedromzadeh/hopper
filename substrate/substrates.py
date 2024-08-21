import numpy as np
from scipy.ndimage import rotate, zoom


class Substrate:
    """
    Implements various confinements within the phase-field framework.

    Attributes
    ----------
    self.N_mesh : int
        Number of lattice points.

    self.L_box : float
        Size of box.

    self.xi : float
        Specifies the interfacial thickness of the field.

    Methods
    -------
    __init__(self, N_mesh, L_box, xi)
        Initialize an instance.

    __str__(self)
        Format how to print the class instance.

    get_substrate(self, x, y, type)
        Generate the requested confinement.

    _rectangular(self, x, y)
        Returns a rectangular confinement.

    _line(self, y, yB=5)
        Returns a floor confinement at yB.

    _circular(self, x, y)
        Returns a circular confinement.

    _Y(self, x, y)
        Returns a Y-channel confinement.

    _plus(self, x, y)
        Returns a plus-channel confinement.
    """

    def __init__(self, N_mesh, L_box, xi=0.5):
        """
        Initialize an instance.

        Parameters
        ----------
        self.N_mesh : int
            Number of lattice points.

        self.L_box : float
            Size of box.

        self.xi : float, optional
            Specifies the interfacial thickness of the field, by default 0.5.

        """
        self.N_mesh = N_mesh
        self.L_box = L_box
        self.xi = xi

    def __str__(self):
        return "\t" + " + You are currently using the {} substrate.".format(self.type)

    def infinite(self):
        """A 2D free micropattern"""
        return np.zeros(shape=(self.N_mesh, self.N_mesh))

    def rectangular(self, width=38, length=111):
        # useful variables
        N_mesh = self.N_mesh
        L_box = self.L_box
        xi = self.xi

        # lattice points
        x, y = np.meshgrid(np.linspace(0, L_box, N_mesh), np.linspace(0, L_box, N_mesh))

        delta = length / (2 * 6)  # in PF.L & halved
        L_bridge = width / (2 * 6)  # in PF._bridge & halved

        # build the bridge
        xL = L_box / 2 - delta
        xR = L_box / 2 + delta
        yB, yT = L_box / 2 - L_bridge, L_box / 2 + L_bridge
        chi_y = 0.5 * ((1 - np.tanh((y - yB) / xi)) + (1 + np.tanh((y - yT) / xi)))
        chi_x = 0.5 * ((1 - np.tanh((x - xL) / xi)) + (1 + np.tanh((x - xR) / xi)))
        chi = chi_x + chi_y
        chi = np.where(chi > 1, 1, chi)
        return chi

    def line(self, yB=5):
        """floor confinement"""
        eps = self.xi
        N_mesh = self.N_mesh
        L_box = self.L_box

        # lattice points
        x, y = np.meshgrid(np.linspace(0, L_box, N_mesh), np.linspace(0, L_box, N_mesh))

        return 0.5 * (1 - np.tanh((y - yB) / eps))

    def circular(self, Rl=18, Rs=10):
        """circular confinement"""
        N_mesh = self.N_mesh
        L_box = self.L_box

        # lattice points
        x, y = np.meshgrid(np.linspace(0, L_box, N_mesh), np.linspace(0, L_box, N_mesh))

        x_center, y_center = L_box / 2, L_box / 2
        a, b, c = 1, 1, 1
        x = x - x_center
        y = y - y_center
        chi_sqrd = (x / a) ** 2 + (y / b) ** 2
        chi_sqrd *= c**2
        chi_1 = np.sqrt(chi_sqrd) - Rl
        chi_2 = -(np.sqrt(chi_sqrd) - Rs)
        chi_1 = 1 / (1 + np.exp(-chi_1))
        chi_2 = 1 / (1 + np.exp(-chi_2))
        chi = chi_1 + chi_2
        return chi

    def Y(self):
        """Y substrate"""
        eps = self.xi
        N_mesh = self.N_mesh
        L_box = self.L_box

        # lattice points
        x, y = np.meshgrid(np.linspace(0, L_box, N_mesh), np.linspace(0, L_box, N_mesh))

        width = 3
        eps = 0.5
        x = x - 25
        y = y - 25
        chiL = (
            1
            / 2
            * (
                (np.tanh((y - x + width + 2.5) / eps))
                + (-np.tanh((y - x - width) / eps))
            )
        )
        chiR = (
            1
            / 2
            * (
                (np.tanh((y + x + width + 2.5) / eps))
                + (-np.tanh((y + x - width) / eps))
            )
        )
        chiC = 1 - 1 / 2 * (
            (np.tanh((x + width) / eps)) + (-np.tanh((x - width) / eps))
        )
        chiL_trunc = np.zeros(chiL.shape)
        chiL_trunc[0:50, 0:50] = chiL[0:50, 0:50]
        chiR_trunc = np.zeros(chiR.shape)
        chiR_trunc[0:50, 50:100] = chiR[0:50, 50:100]
        chiC_trunc = np.ones(chiC.shape)
        chiC_trunc[50:100, :] = chiC[50:100, :]
        chi = chiC_trunc - chiR_trunc - chiL_trunc
        chi = np.where(chi < 0, 0, chi)
        return chi

    def plus(self):
        """+ substrate"""
        eps = self.xi
        N_mesh = self.N_mesh
        L_box = self.L_box

        # lattice points
        x, y = np.meshgrid(np.linspace(0, L_box, N_mesh), np.linspace(0, L_box, N_mesh))

        width = 3
        eps = 0.5
        x = x - 25
        y = y - 25
        chiH = 1 / 2 - 1 / 2 * (
            (np.tanh((x + width) / eps)) + (-np.tanh((x - width) / eps))
        )
        chiV = 1 / 2 - 1 / 2 * (
            (np.tanh((y + width) / eps)) + (-np.tanh((y - width) / eps))
        )
        chi = chiV + chiH
        chi = np.where(chi < 0.001, 0, chi)
        return chi

    def two_state_sub(self, square_width=38, bridge_width=10, delta_centers=73):
        """
        Returns a two-state (dumbell-shaped) micropattern, with inside = 0.

        Parameters
        ----------
        square_width : float, optional
            Specifies the dimension of the basins in microns, by default 38
        bridge_width : float, optional
            Specifies the dimension of bridge opening in microns, by default 10
        delta_centers : float, optional
            Specifies the distance between two basin centers in microns, by default 73

        Returns
        -------
        np.ndarray of shape (N_mesh, N_mesh)
            The micropattern.
        """
        # useful variables
        N_mesh = self.N_mesh
        L_box = self.L_box
        xi = self.xi

        # lattice points
        x, y = np.meshgrid(np.linspace(0, L_box, N_mesh), np.linspace(0, L_box, N_mesh))

        # index of the two squares, used for symmetric positioning
        L_sqrd = square_width / (2 * 6)  # in PF.L & halved
        delta = delta_centers / (2 * 6)  # in PF.L & halved
        L_bridge = bridge_width / (2 * 6)  # in PF._bridge & halved

        # build the squares
        squares = None
        for k in range(2):
            if k == 0:
                center = L_box / 2 - delta
            else:
                center = L_box / 2 + delta

            xL = center - L_sqrd
            xR = center + L_sqrd
            yB, yT = L_box / 2 - L_sqrd, L_box / 2 + L_sqrd
            chi_y = 0.5 * ((1 - np.tanh((y - yB) / xi)) + (1 + np.tanh((y - yT) / xi)))
            chi_x = 0.5 * ((1 - np.tanh((x - xL) / xi)) + (1 + np.tanh((x - xR) / xi)))
            chi = chi_x + chi_y
            chi = np.where(chi > 1, 1, chi)
            if squares is None:
                squares = chi
            else:
                squares += chi

        # build the bridge
        xL = L_box / 2 - delta
        xR = L_box / 2 + delta
        yB, yT = L_box / 2 - L_bridge, L_box / 2 + L_bridge
        chi_y = 0.5 * ((1 - np.tanh((y - yB) / xi)) + (1 + np.tanh((y - yT) / xi)))
        chi_x = 0.5 * ((1 - np.tanh((x - xL) / xi)) + (1 + np.tanh((x - xR) / xi)))
        chi = chi_x + chi_y
        chi = np.where(chi > 1, 1, chi)

        # put squares and bridge together
        mp = (squares - 1) + chi
        mp -= 1
        mp = np.where(mp < 0, 0, mp)
        return mp

    def mixed_two_state_sub(self, basin_dims, bridge_dims):
        """
        Returns a two-state (dumbell-shaped) micropattern, with inside = 0.

        Parameters
        ----------
        basin_dims : np.ndarray of shape (2, 2)
            Specifies the (length, width) dimensions of each basin (left: 0, right: 1) in microns
        bridge_dims : list
            Specifies the length and width of the bridge in microns

        Returns
        -------
        np.ndarray of shape (N_mesh, N_mesh)
            The micropattern.
        """
        # useful variables
        N_mesh = self.N_mesh
        L_box = self.L_box
        xi = self.xi
        bridge_length, bridge_width = bridge_dims
        basin_dims = np.array(basin_dims)

        # lattice points
        x, y = np.meshgrid(np.linspace(0, L_box, N_mesh), np.linspace(0, L_box, N_mesh))

        # index of the two squares, used for symmetric positioning
        delta = (bridge_length + basin_dims[:, 0].sum() * 0.5) / (
            2 * 6
        )  # in PF.L & halved
        L_bridge = bridge_width / (2 * 6)  # in PF._bridge & halved

        # build the squares
        squares = None
        for k in range(2):
            if k == 0:
                center = L_box / 2 - delta
            else:
                center = L_box / 2 + delta

            xL = center - basin_dims[k][0] / (2 * 6)  # in PF.L & halved
            xR = center + basin_dims[k][0] / (2 * 6)  # in PF.L & halved
            yB = L_box / 2 - basin_dims[k][1] / (2 * 6)  # in PF.L & halved
            yT = L_box / 2 + basin_dims[k][1] / (2 * 6)  # in PF.L & halved
            chi_y = 0.5 * ((1 - np.tanh((y - yB) / xi)) + (1 + np.tanh((y - yT) / xi)))
            chi_x = 0.5 * ((1 - np.tanh((x - xL) / xi)) + (1 + np.tanh((x - xR) / xi)))
            chi = chi_x + chi_y
            chi = np.where(chi > 1, 1, chi)

            if squares is None:
                squares = chi
            else:
                squares += chi

        # build the bridge
        xL = L_box / 2 - delta
        xR = L_box / 2 + delta
        yB, yT = L_box / 2 - L_bridge, L_box / 2 + L_bridge
        chi_y = 0.5 * ((1 - np.tanh((y - yB) / xi)) + (1 + np.tanh((y - yT) / xi)))
        chi_x = 0.5 * ((1 - np.tanh((x - xL) / xi)) + (1 + np.tanh((x - xR) / xi)))
        chi = chi_x + chi_y
        chi = np.where(chi > 1, 1, chi)

        # put squares and bridge together
        mp = (squares - 1) + chi
        mp -= 1
        mp = np.where(mp < 0, 0, mp)
        return mp

    def sq_tri_two_state_sub(self):
        """
        Build a static, simple square-triangle substrate.

        Returns
        -------
        np.ndarray
            The substrate
        """
        # useful variables
        N_mesh = 200
        L_box = 50
        xi = 0.2
        dims = [38, 38]

        # lattice points
        x, y = np.meshgrid(np.linspace(0, L_box, N_mesh), np.linspace(0, L_box, N_mesh))

        # triangle
        triangle = np.ones(x.shape)
        f = 0.71
        x0 = 23
        side_1 = 0.5 * (1 - np.tanh((f * (x + x0) - y) / (5 * xi)))
        side_2 = 0.5 * (1 - np.tanh((y + (f * (x + x0) - 50)) / (5 * xi)))
        combo = side_1 + 2 * side_2
        triangle[:, N_mesh // 2 - x0 :] = combo[:, N_mesh // 2 - x0 :]
        triangle = np.where(triangle > 1 + 1e-4, 1, triangle)

        # index of the two squares, used for symmetric positioning
        bridge_length = 35
        bridge_width = 17
        delta = (bridge_length + dims[0]) / (2 * 6)  # in PF.L & halved
        L_bridge = bridge_width / (2 * 6)  # in PF._bridge & halved
        center = L_box / 2 - delta

        xL = center - dims[0] / (2 * 6)  # in PF.L & halved
        xR = center + dims[0] / (2 * 6)  # in PF.L & halved
        yB = L_box / 2 - dims[1] / (2 * 6)  # in PF.L & halved
        yT = L_box / 2 + dims[1] / (2 * 6)  # in PF.L & halved
        chi_y = 0.5 * ((1 - np.tanh((y - yB) / xi)) + (1 + np.tanh((y - yT) / xi)))
        chi_x = 0.5 * ((1 - np.tanh((x - xL) / xi)) + (1 + np.tanh((x - xR) / xi)))
        chi = chi_x + chi_y
        square = np.where(chi > 1, 1, chi)

        # build the bridge
        xL = L_box / 2 - delta
        xR = L_box / 2 + delta
        yB, yT = L_box / 2 - L_bridge, L_box / 2 + L_bridge
        chi_y = 0.5 * ((1 - np.tanh((y - yB) / xi)) + (1 + np.tanh((y - yT) / xi)))
        chi_x = 0.5 * ((1 - np.tanh((x - xL) / xi)) + (1 + np.tanh((x - xR) / xi)))
        chi = chi_x + chi_y
        bridge = np.where(chi > 1, 1, chi)

        # put squares and bridge together
        scaled_triangle = zoom(triangle, zoom=0.2, order=0)
        final = np.ones(square.shape)
        q = scaled_triangle.shape[0] // 2
        y0 = N_mesh // 2
        x0 = N_mesh // 2 + 13
        final[y0 - q : y0 + q, x0 - q : x0 + q] = scaled_triangle
        mp = square + bridge + final
        mp -= 2
        mp = np.where(mp < 0, 0, mp)
        return mp

    def sq_rhom_two_state_sub(self):
        """
        Build a static square-rhomboid two-state substrate.

        Returns
        -------
        np.ndarray of shape (N_mesh, N_mesh)
            The micropattern.
        """
        # useful variables
        N_mesh = 200
        L_box = 50
        xi = 0.1
        dims = np.array([[38, 38], [38, 38]])

        # lattice points
        x, y = np.meshgrid(np.linspace(0, L_box, N_mesh), np.linspace(0, L_box, N_mesh))

        # index of the two squares, used for symmetric positioning
        bridge_length = 35
        delta = (bridge_length + dims[:, 0].sum() * 0.5) / (2 * 6)  # in PF.L & halved
        L_bridge = 17 / (2 * 6)  # in PF._bridge & halved

        # build the squares
        squares = None
        for k in range(2):
            if k == 0:
                center = L_box / 2 - delta
            else:
                center = L_box / 2 + delta

            xL = center - dims[k][0] / (2 * 6)  # in PF.L & halved
            xR = center + dims[k][0] / (2 * 6)  # in PF.L & halved
            yB = L_box / 2 - dims[k][1] / (2 * 6)  # in PF.L & halved
            yT = L_box / 2 + dims[k][1] / (2 * 6)  # in PF.L & halved
            chi_y = 0.5 * ((1 - np.tanh((y - yB) / xi)) + (1 + np.tanh((y - yT) / xi)))
            chi_x = 0.5 * ((1 - np.tanh((x - xL) / xi)) + (1 + np.tanh((x - xR) / xi)))
            chi = chi_x + chi_y
            chi = np.where(chi > 1, 1, chi)
            if k == 1:
                chi = rotate(chi, angle=45, reshape=False)
                chi = np.roll(chi, 6, axis=1)
                chi = np.roll(chi, 17, axis=0)
            if squares is None:
                squares = chi
            else:
                squares += chi

        # build the bridge
        xL = L_box / 2 - delta
        xR = L_box / 2 + delta
        yB, yT = L_box / 2 - L_bridge, L_box / 2 + L_bridge
        chi_y = 0.5 * ((1 - np.tanh((y - yB) / xi)) + (1 + np.tanh((y - yT) / xi)))
        chi_x = 0.5 * ((1 - np.tanh((x - xL) / xi)) + (1 + np.tanh((x - xR) / xi)))
        chi = chi_x + chi_y
        chi = np.where(chi > 1, 1, chi)

        # put squares and bridge together
        mp = (squares - 1) + chi
        mp -= 1
        mp = np.where(mp < 0, 0, mp)
        return mp
