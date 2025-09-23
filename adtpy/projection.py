# # Functions


# ## Class 'proj'

import numpy as np
from dataclasses import dataclass, field

@dataclass
class Proj:
    d1_PCs: np.ndarray
    d2_PCs: np.ndarray
    n_loadings: int
    proj_coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    lengths: np.ndarray = field(default_factory=lambda: np.array([]))
    angles: np.ndarray = field(default_factory=lambda: np.array([]))
    similarity: float = 0.0

    def __post_init__(self):
        # --- type checks ---
        if not isinstance(self.d1_PCs, np.ndarray) or not isinstance(self.d2_PCs, np.ndarray):
            raise TypeError("principal components must be numpy arrays (matrices)")
        if not isinstance(self.proj_coords, np.ndarray):
            raise TypeError("proj_coords must be a numpy array")
        if not np.issubdtype(self.lengths.dtype, np.number):
            raise TypeError('"lengths" must be numeric')
        if not np.issubdtype(self.angles.dtype, np.number):
            raise TypeError('"angles" must be numeric')
        if not isinstance(self.similarity, (int, float)):
            raise TypeError('"similarity" must be numeric')

        # --- dimension checks ---
        if self.d1_PCs.shape[1] < 1 or self.d1_PCs.shape[1] > 3:
            raise ValueError("the number of PCs in d1_PCs must be between 1 and 3")
        if self.d2_PCs.shape[1] < 1 or self.d2_PCs.shape[1] > 3:
            raise ValueError("the number of PCs in d2_PCs must be between 1 and 3")
        if self.proj_coords.shape[0] != self.n_loadings:
            raise ValueError('unmatched length of "proj_coords" rows')
        if len(self.lengths) != self.n_loadings:
            raise ValueError('unmatched length of "lengths"')
        if self.n_loadings == 2 and len(self.angles) != 1:
            raise ValueError("for 2 loadings, there must be 1 angle")
        if self.n_loadings == 3 and len(self.angles) != 3:
            raise ValueError("for 3 loadings, there must be 3 angles")
        if not (isinstance(self.similarity, (int, float)) and np.ndim(self.similarity) == 0):
            raise ValueError('"similarity" must be a single number')

        # --- keep only the first n_loadings components ---
        self.d1_PCs = self.d1_PCs[:, :self.n_loadings]
        self.d2_PCs = self.d2_PCs[:, :self.n_loadings]

    # --- Accessor methods (R-style equivalents) ---
    @property
    def d1(self): return self.d1_PCs

    @property
    def d2(self): return self.d2_PCs

    @property
    def n(self): return self.n_loadings

    @property
    def coords(self): return self.proj_coords

    @property
    def lens(self): return self.lengths

    @property
    def angs(self): return self.angles

    @property
    def sim(self): return self.similarity





def proj(d1_PCs, d2_PCs, n_loadings=None,
         proj_coords=None, lengths=None, angles=None, similarity=0.0):
    # determine n_loadings
    if n_loadings is None:
        n_loadings = min(d1_PCs.shape[1], d2_PCs.shape[1])

    # defaults
    if proj_coords is None:
        proj_coords = np.zeros((n_loadings, 1))
    if lengths is None:
        lengths = np.zeros(n_loadings)
    if angles is None:
        angles = np.zeros(3 if n_loadings == 3 else 1)

    return Proj(
        d1_PCs=np.asarray(d1_PCs),
        d2_PCs=np.asarray(d2_PCs),
        n_loadings=n_loadings,
        proj_coords=np.asarray(proj_coords),
        lengths=np.asarray(lengths),
        angles=np.asarray(angles),
        similarity=float(similarity),
    )


# ## Computing Projection




import numpy as np
import pandas as pd

def proj_1d(d1_PC1: np.ndarray, d2_PC1: np.ndarray) -> pd.DataFrame:
    """
    1D projection: cosine of angle between two PC vectors.
    Returns a DataFrame with one column 'x'.
    """
    d1 = np.asarray(d1_PC1, dtype=float).ravel()
    d2 = np.asarray(d2_PC1, dtype=float).ravel()
    d1 /= np.linalg.norm(d1)
    d2 /= np.linalg.norm(d2)
    cos_angle = float(np.dot(d1, d2))
    return pd.DataFrame([[cos_angle]], columns=["x"])





def proj_2d(d1_PCs: np.ndarray, d2_PCs: np.ndarray) -> pd.DataFrame:
    """
    2D projection: project dataset 2's PCs onto dataset 1's 2D plane.
    Returns a DataFrame with columns ['x','y'].
    """
    if not isinstance(d1_PCs, np.ndarray) or not isinstance(d2_PCs, np.ndarray):
        raise TypeError("PCs must be numpy arrays (matrices)")
    if d1_PCs.shape[1] != 2 or d2_PCs.shape[1] != 2:
        raise ValueError("PCs must have exactly 2 dimensions (2 columns)")

    # d2_pc1 projected to d1 plane
    p1 = [proj_1d(d1_PCs[:, 0], d2_PCs[:, 0]).iloc[0, 0],
          proj_1d(d1_PCs[:, 1], d2_PCs[:, 0]).iloc[0, 0]]

    # d2_pc2 projected to d1 plane
    p2 = [proj_1d(d1_PCs[:, 0], d2_PCs[:, 1]).iloc[0, 0],
          proj_1d(d1_PCs[:, 1], d2_PCs[:, 1]).iloc[0, 0]]

    coords = np.vstack([p1, p2])
    return pd.DataFrame(coords, columns=["x", "y"])





def proj_3d(d1_PCs: np.ndarray, d2_PCs: np.ndarray) -> pd.DataFrame:
    """
    3D projection: project dataset 2's PCs onto dataset 1's 3D space.
    Returns a DataFrame with columns ['x','y','z'].
    """
    if not isinstance(d1_PCs, np.ndarray) or not isinstance(d2_PCs, np.ndarray):
        raise TypeError("PCs must be numpy arrays (matrices)")
    if d1_PCs.shape[1] != 3 or d2_PCs.shape[1] != 3:
        raise ValueError("PCs must have exactly 3 dimensions (3 columns)")

    # d2_pc1 projected to d1 plane
    p1 = [proj_1d(d1_PCs[:, 0], d2_PCs[:, 0]).iloc[0, 0],
          proj_1d(d1_PCs[:, 1], d2_PCs[:, 0]).iloc[0, 0],
          proj_1d(d1_PCs[:, 2], d2_PCs[:, 0]).iloc[0, 0]]

    # d2_pc2 projected to d1 plane
    p2 = [proj_1d(d1_PCs[:, 0], d2_PCs[:, 1]).iloc[0, 0],
          proj_1d(d1_PCs[:, 1], d2_PCs[:, 1]).iloc[0, 0],
          proj_1d(d1_PCs[:, 2], d2_PCs[:, 1]).iloc[0, 0]]

    # d2_pc3 projected to d1 plane
    p3 = [proj_1d(d1_PCs[:, 0], d2_PCs[:, 2]).iloc[0, 0],
          proj_1d(d1_PCs[:, 1], d2_PCs[:, 2]).iloc[0, 0],
          proj_1d(d1_PCs[:, 2], d2_PCs[:, 2]).iloc[0, 0]]

    coords = np.vstack([p1, p2, p3])
    return pd.DataFrame(coords, columns=["x", "y", "z"])


# ## Add projection information




import numpy as np
import pandas as pd
from numpy.linalg import det

def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return angle (radians) between two vectors."""
    v1 = np.asarray(v1, dtype=float).ravel()
    v2 = np.asarray(v2, dtype=float).ravel()
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(cos_angle)

def rad2deg(rad: float) -> float:
    """Convert radians to degrees."""
    return np.degrees(rad)

def nthroot(value: float, n: int) -> float:
    """Compute the real n-th root of a value."""
    return np.sign(value) * (abs(value) ** (1.0 / n))

def proj_compute(proj: Proj) -> Proj:
    """
    Compute coordinates, lengths, angles, and similarity for a Proj object.
    Updates the object in place and returns it.
    """
    if not isinstance(proj, Proj):
        raise TypeError("input must be of class 'Proj'")
    if proj.n_loadings not in (1, 2, 3):
        raise ValueError("invalid dimensions: must be 1, 2, or 3")

    if proj.n_loadings == 3:
        # --- 3D case ---
        proj.proj_coords = proj_3d(proj.d1_PCs, proj.d2_PCs)

        p1 = proj.proj_coords.iloc[0].to_numpy()
        p2 = proj.proj_coords.iloc[1].to_numpy()
        p3 = proj.proj_coords.iloc[2].to_numpy()

        proj.lengths = pd.Series({
            "p1_l": np.sqrt(np.sum(p1**2)),
            "p2_l": np.sqrt(np.sum(p2**2)),
            "p3_l": np.sqrt(np.sum(p3**2)),
        })

        proj.angles = pd.Series({
            "p1_p2_deg": rad2deg(angle_between(p1, p2)),
            "p1_p3_deg": rad2deg(angle_between(p1, p3)),
            "p2_p3_deg": rad2deg(angle_between(p2, p3)),
        })

        proj.similarity = abs(nthroot(det(proj.proj_coords.to_numpy().T), 3))

    elif proj.n_loadings == 2:
        # --- 2D case ---
        proj.proj_coords = proj_2d(proj.d1_PCs[:, :2], proj.d2_PCs[:, :2])

        p1 = proj.proj_coords.iloc[0].to_numpy()
        p2 = proj.proj_coords.iloc[1].to_numpy()

        proj.lengths = pd.Series({
            "p1_l": np.sqrt(np.sum(p1**2)),
            "p2_l": np.sqrt(np.sum(p2**2)),
        })

        angle = angle_between(p1, p2)
        proj.angles = pd.Series({"angle_deg": rad2deg(angle)})

        proj.similarity = float(
            np.sqrt(proj.lengths.iloc[0] * proj.lengths.iloc[1] * np.sin(angle))
        )

    else:
        # --- 1D case ---
        proj.proj_coords = proj_1d(proj.d1_PCs, proj.d2_PCs)

        p = float(proj.proj_coords.iloc[0, 0])
        proj.lengths = pd.Series({"p": p})
        proj.similarity = p

    return proj


# ## Defining a “show” Method




@dataclass
class Proj:
    d1_PCs: np.ndarray
    d2_PCs: np.ndarray
    n_loadings: int
    proj_coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    lengths: np.ndarray = field(default_factory=lambda: np.array([]))
    angles: np.ndarray = field(default_factory=lambda: np.array([]))
    similarity: float = 0.0

    def __post_init__(self):
        # (same validation checks as before)
        ...
    
    # --- custom display method (R's show) ---
    def __str__(self):
        if self.similarity != 0:
            return (
                f"Proj(similarity={self.similarity}, "
                f"n_loadings={self.n_loadings}, "
                f"proj_coords=\n{self.proj_coords})"
            )
        else:
            return "Please run function 'proj_compute' to compute projection"

    # Optional: also for interactive printing in Jupyter / REPL
    __repr__ = __str__
