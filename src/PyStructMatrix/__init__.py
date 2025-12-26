import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

@dataclass
class Node:
    """
    Almacena coordenadas, restricciones, desplazamientos,
    reacciones y fuerzas nodales asociadas al sistema global.
    """
    ID: int
    x: float
    z: float
    restrain: Optional[Tuple[bool, bool, bool]] = (False, False, False)
    Ux: Optional[float] = np.nan
    Uz: Optional[float] = np.nan
    Ry: Optional[float] = np.nan
    Fx: Optional[float] = np.nan
    Fz: Optional[float] = np.nan
    My: Optional[float] = np.nan
    index: Optional[int] = None

    def __post_init__(self):
        self.apply_restraints()

    def set_index(self, list_nodes: List):
        self.index = list_nodes.index(self.ID)
    def apply_restraints(self):

        ux_restrained, uz_restrained, ry_restrained = self.restrain

        # Apoyo libre: (False, False, False)
        if not ux_restrained and not uz_restrained and not ry_restrained:
            self.Fx = 0.0
            self.Fz = 0.0
            self.My = 0.0

        # Apoyo empotrado: (True, True, True)
        elif ux_restrained and uz_restrained and ry_restrained:
            self.Ux = 0.0
            self.Uz = 0.0
            self.Ry = 0.0

        # Apoyo fijo: (True, True, False)
        elif ux_restrained and uz_restrained and not ry_restrained:
            self.Ux = 0.0
            self.Uz = 0.0
            self.My = 0.0


        # Apoyo móvil en X y fijo en Z: (False, True, False)
        elif not ux_restrained and uz_restrained and not ry_restrained:
            self.Uz = 0.0
            self.My = 0.0
            self.Fx = 0.0

        # Apoyo móvil en Z y fijo en X: (True, False, False)
        elif ux_restrained and not uz_restrained and not ry_restrained:
            self.Ux = 0.0
            self.My = 0.0
            self.Fz = 0.0

        # Restricción en momento: (False, False, True)
        elif not ux_restrained and not uz_restrained and ry_restrained:
            self.My = 0.0
            self.Ry = 0.0

    def update_restrain(self, new_restrain: Tuple[bool, bool, bool]):
        self.restrain = new_restrain
        self.reset_constrained_values()
        self.apply_restraints()

    def reset_constrained_values(self):
        self.Ux = np.nan
        self.Uz = np.nan
        self.Ry = np.nan
        self.Fx = np.nan
        self.Fz = np.nan
        self.My = np.nan

    def set_displacement(self, ux: Optional[float] = None,
                         uz: Optional[float] = None,
                         ry: Optional[float] = None):

        if ux is not None and self.restrain[0]:
            self.Ux = ux

        if uz is not None and self.restrain[1]:
            self.Uz = uz

        if ry is not None and self.restrain[2]:
            self.Ry = ry

    def set_force(self, fx: Optional[float] = 0,
                  fz: Optional[float] = 0,
                  my: Optional[float] = 0):

        if fx != 0.0 and not self.restrain[0]:
            self.Fx = fx

        if fz != 0.0 and not self.restrain[1]:
            self.Fz = fz

        if my != 0.0 and not self.restrain[2]:
            self.My = my

    @property
    def support_type(self) -> str:
        if self.restrain is None:
            return "Sin restricciones"

        ux_restrained, uz_restrained, ry_restrained = self.restrain

        if not ux_restrained and not uz_restrained and not ry_restrained:
            return "Libre"
        elif ux_restrained and uz_restrained and ry_restrained:
            return "Empotrado"
        elif ux_restrained and uz_restrained and not ry_restrained:
            return "Fijo"
        elif not ux_restrained and uz_restrained and not ry_restrained:
            return "Móvil en X"
        elif ux_restrained and not uz_restrained and not ry_restrained:
            return "Móvil en Z"
        else:
            return f"Apoyo personalizado {self.restrain}"

    def __str__(self) -> str:
        return (f"Node {self.ID}: ({self.x}, {self.z}) - {self.support_type}\n"
                f"  Desplazamientos: Ux={self.Ux:.4e}, Uz={self.Uz:.4e}, Ry={self.Ry:.4e}\n"
                f"  Fuerzas: Fx={self.Fx:.3f}, Fz={self.Fz:.3f}, My={self.My:.3f}\n")

@dataclass
class Load(ABC):
    @abstractmethod
    def get_load_type(self) -> str:
        pass
    @abstractmethod
    def get_description(self) -> str:
        pass

@dataclass
class PointForce(Load):
    L: float
    W: float  # magnitud de la fuerza (positiva hacia abajo)
    distance: float # distancia relativa desde i
    FEF: Optional[np.array] = None

    def __post_init__(self):
        if self.distance <= 1 and self.distance >=0:
            self.l1: float = self.distance*self.L  # distancia desde el nodo i
            self.l2: float = self.L - self.l1  # distancia desde el nodo j (para verificación: l1 + l2 = L)
            self.FEF = self.get_FEF()
        else:
            raise ValueError("La distancia relativa de la ca carga debe ser"
                             "mayor a 0 y menor a 1")

    def get_load_type(self) -> str:
        return "PointForce"

    def get_description(self) -> str:
        return f"Fuerza puntual: P={self.W:.3f} a {self.l1} del nodo i"
    def get_FEF(self):
        FS_b = (self.W * self.l2 ** 2) / (self.L ** 3) * (3 * self.l1 + self.l2)
        FM_b = (self.W * self.l1 * self.l2 ** 2) / (self.L ** 2)
        FS_e = (self.W * self.l1 ** 2) / (self.L ** 3) * (self.l1 + 3 * self.l2)
        FM_e = - (self.W * self.l1 ** 2 * self.l2) / (self.L ** 2)
        return np.array([0, FS_b, FM_b, 0, FS_e, FM_e])

@dataclass
class AxialPointForce(Load):
    L: float
    W: float  # magnitud de la fuerza
    distance: float
    FEF: Optional[np.array] = None

    def __post_init__(self):
        if self.distance <= 1 and self.distance >=0:
            self.l1: float = self.distance*self.L  # distancia desde el nodo i
            self.l2: float = self.L - self.l1  # distancia desde el nodo j (para verificación: l1 + l2 = L)
            self.FEF = self.get_FEF()
        else:
            raise ValueError("La distancia relativa de la ca carga debe ser"
                             "mayor a 0 y menor a 1")

    def get_load_type(self) -> str:
        return "PointForce"

    def get_description(self) -> str:
        return f"Fuerza puntual axial: P={self.W:.3f} a {self.l1} del nodo i"
    def get_FEF(self):
        FA_b = (self.W * self.l2) / self.L
        FA_e = (self.W * self.l1) / self.L
        return np.array([FA_b, 0, 0, FA_e, 0, 0])

@dataclass
class PointMoment(Load):
    L: float
    M: float  # magnitud del momento (positivo horario)
    distance: float  # distancia relativa desde i
    FEF: Optional[np.array] = None

    def __post_init__(self):
        if self.distance <= 1 and self.distance >= 0:
            self.l1: float = self.distance * self.L  # distancia desde el nodo i
            self.l2: float = self.L - self.l1  # distancia desde el nodo j (para verificación: l1 + l2 = L)
            self.FEF = self.get_FEF()
        else:
            raise ValueError("La distancia relativa de la ca carga debe ser"
                             "mayor a 0 y menor a 1")

    def get_load_type(self) -> str:
        return "PointMoment"
    def get_description(self) -> str:
        return f"Momento puntual: M={self.M:.3f} a {self.l1} del nodo i"
    def get_FEF(self):
        FS_b = - (6 * self.M * self.l1 * self.l2) / (self.L ** 3)
        FM_b = (self.M * self.l2) / (self.L ** 2) * (self.l2 - 2 * self.l1)
        FS_c = (6 * self.M * self.l1 * self.l2) / (self.L ** 3)
        FM_e = (self.M * self.l1) / (self.L ** 2) * (self.l1 - 2 * self.l2)
        return np.array([0, FS_b, FM_b, 0, FS_c, FM_e])
@dataclass
class UniformLoad(Load):
    L: float
    w: float  # carga distribuida + hacia abajo
    FEF: Optional[np.array] = None

    def __post_init__(self):
        self.FEF = self.get_FEF()

    def get_load_type(self) -> str:
        return "UniformLoad"

    def get_description(self) -> str:
        return f"Carga uniforme: w={self.w:.3f} en toda la longitud"
    def get_FEF(self):
        return np.array([0, self.w * self.L / 2,
                         self.w * (self.L ** 2) / 12,
                         0, self.w * self.L / 2,
                         -self.w * (self.L ** 2) / 12])

@dataclass
class TrapezoidalLoad(Load):
    L: float
    w1: float  # carga en el nodo 1
    w2: float # carga en el nodo j
    FEF: Optional[np.array] = None

    def __post_init__(self):
        self.FEF = self.get_FEF()

    def get_load_type(self) -> str:
        return "UniformLoad"

    def get_description(self) -> str:
        return f"Carga trapezoidal: wi={self.w1:.3f} -> wj={self.w2:.3f}"
    def get_FEF(self):
        if self.w1<self.w2:
            Mi = (self.w1/20+self.w2/30)*self.L**2
            Mj = (self.w1/30+self.w2/20)*self.L**2
            Ri = (7*self.w1+3*self.w2)*self.L/20
            Rj = (3*self.w1+7*self.w2)*self.L/20
            return np.array([0,Ri,Mi,0,Rj,-Mj])

        else:
            Mj = (self.w2/20+self.w1/30)*self.L**2
            Mi = (self.w2/30+self.w1/20)*self.L**2
            Rj = (7*self.w2+3*self.w1)*self.L/20
            Ri = (3*self.w2+7*self.w1)*self.L/20
            return np.array([0,Ri,Mi,0,Rj,-Mj])

@dataclass
class CenteredTriangularLoad:
    L: float
    w: float  # carga en el centro del elemento
    FEF: Optional[np.array] = None

    def __post_init__(self):
        self.FEF = self.get_FEF()

    def get_load_type(self) -> str:
        return "UniformLoad"

    def get_description(self) -> str:
        return f"Carga Triangular: w={self.w:.3f} al centro del elemento"
    def get_FEF(self):
        Mij = (5/96)*self.w*self.L**2
        Rij = self.w*self.L/4
        return np.array([0,Rij,Mij,0,Rij,-Mij])

@dataclass
class TemperatureLoad(Load):
    tipo: int
    dT: float  # variación de temperatura
    alpha: float
    EI: float
    EA: float
    d: Optional[float] = None
    FEF: Optional[np.array] = None

    def __post_init__(self):
        self.FEF = self.get_FEF()

    def get_load_type(self) -> str:
        return "TemperatureLoad"

    def get_description(self) -> str:
        if self.d is not None:
            return f"Variación térmica: ΔT={self.dT:.3f} | d:{self.d:.3f}"
        else:
            return f"Variación térmica: ΔT={self.dT:.3f}"
    def get_FEF(self):
        if self.d is not None:
            self.alpha * self.dT * np.array([0, 0, self.EI/self.d, 0, 0, -self.EI/self.d])
        else:
            return self.alpha * self.dT * np.array([self.EA, 0, 0, -self.EA, 0, 0])

@dataclass
class AxialUniformLoad(Load):
    L: float
    w: float  # carga axial distribuida
    FEF: Optional[np.array] = None

    def __post_init__(self):
        self.FEF = self.get_FEF()

    def get_load_type(self) -> str:
        return "AxialUniformLoad"

    def get_description(self) -> str:
        return f"Carga axial uniforme: w={self.w:.3f} en toda la longitud"
    def get_FEF(self):
        FA_b = self.w*self.L/2
        FA_e = self.w*self.L/2
        return np.array([FA_b, 0, 0, FA_e, 0, 0])

@dataclass
class Element:
    """
    Representa un elemento lineal entre dos nodos, almacenando
    propiedades mecánicas, cargas, matrices de rigidez y
    resultados internos del análisis estructural
    """
    ID: int
    Ni: int  # ID del nodo i
    Nj: int  # ID del nodo j
    nodi: 'Node'  # referencia al nodo i
    nodj: 'Node'  # referencia al nodo j
    A: float
    E: float
    I: float
    G: Optional[float] = None
    weight: Optional[float] = None

    # Propiedades del elemento
    releases: Optional[Tuple[bool, bool]] = (False, False)  # liberaciones en i y j

    tipo: Optional[int] = 0
    C: Optional[float] = None  # coseno del ángulo
    S: Optional[float] = None  # seno del ángulo
    T: Optional[np.array] = None #

    loads: List[Load] = field(default_factory=list) # Sistema de cargas

    # cuando se haya completado el analisis
    K: Optional[np.array] = None # matriz de rigidez local
    FEF: Optional[np.array] = None # fixed end forces: Fxi, Fzi, Myi, Fxj, Fzj, Myj
    Kgl: Optional[np.array] = None
    Qgl: Optional[np.array] = None
    kgl_sub: Optional[np.array] = None
    qgl_sub: Optional[np.array] = None
    Fint: Optional[np.array] = None # fuerzas internas: Fxi, Fzi, Myi, Fxj, Fzj, Myj
    Uint: Optional[np.array] = None # desplazamientos nodos i y j (ux, uz, ry)

    def __post_init__(self):
        dx = self.nodj.x - self.nodi.x
        dz = self.nodj.z - self.nodi.z
        self.length = np.sqrt(dx ** 2 + dz ** 2)

        if self.C is None:
            self.C = dx / self.length if self.length > 0 else 1.0
        if self.S is None:
            self.S = dz / self.length if self.length > 0 else 0.0

        self.T = np.array([[self.C, self.S, 0, 0, 0, 0],
                           [-self.S, self.C, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, self.C, self.S, 0],
                           [0, 0, 0, -self.S, self.C, 0],
                           [0, 0, 0, 0, 0, 1]])

        self.actualizar_tipo()

        if self.weight is not None:
            w = self.weight
            q = w * self.A
            q_axial = q * self.S  # negativo si S>0 (elemento sube hacia j)
            q_trans = q * self.C  # negativo si C>0 (elemento se orienta hacia +x)

            axial_load = AxialUniformLoad(self.length, q_axial)
            trans_load = UniformLoad(self.length, q_trans)
            self.loads.append(axial_load)
            self.loads.append(trans_load)

    def actualizar_tipo(self):
        rii, rjj = self.releases
        fixi = (self.nodi.restrain == (True, True, False)) #(True, True, False)
        fixj = (self.nodj.restrain == (True, True, False))
        ri = rii or fixi
        rj = rjj or fixj
        if not ri and not rj:
            self.tipo = 0
        if ri and not rj:
            self.tipo = 1
        if not ri and rj:
            self.tipo = 2
        if ri and rj:
            self.tipo = 3

    def computar_rigidez(self):
        L = self.length
        E = self.E
        I = self.I
        A = self.A
        K = np.zeros((6,6))
        if self.tipo == 0:
            K = np.array([[A*L**2/I, 0, 0, -A*L**2/I, 0, 0],
                          [0, 12, 6 * L, 0, -12, 6 * L],
                          [0, 6 * L, 4 * L ** 2, 0, -6 * L, 2 * L ** 2],
                          [-A*L**2/I, 0, 0, A*L**2/I, 0, 0],
                          [0, -12, -6 * L, 0, 12, -6 * L],
                          [0, 6 * L, 2 * L ** 2, 0, -6 * L, 4 * L ** 2]])

        if self.tipo == 1:
            K = np.array([[A*L**2/I, 0, 0, -A*L**2/I, 0, 0],
                          [0, 3, 0, 0, -3, 3 * L],
                          [0, 0, 0, 0, 0, 0],
                          [-A*L**2/I, 0, 0, A*L**2/I, 0, 0],
                          [0, -3, 0, 0, 3, -3 * L],
                          [0, 3 * L, 0, 0, -3 * L, 3 * L * L]])
        if self.tipo == 2:
            K = np.array([[A*L**2/I, 0, 0, -A*L**2/I, 0, 0],
                          [0, 3, 3 * L, 0, -3, 0],
                          [0, 3 * L, 3 * L * L, 0, -3 * L, 0],
                          [-A*L**2/I, 0, 0, A*L**2/I, 0, 0],
                          [0, -3, -3 * L, 0, 3, 0],
                          [0, 0, 0, 0, 0, 0], ])

        self.K = (E * I / L ** 3) * K

        if self.tipo == 3:
            K = np.array([[1, 0, 0, -1, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [-1, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]])
            self.K = (E * A / L) * K

    def computar_FEF(self):
        self.actualizar_tipo()
        fef = np.zeros(6)
        if len(self.loads)>0:
            for load in self.loads:
                if load.FEF is not None:
                    fef += load.FEF

        FAb, FSb, FMb, FAe, FSe, FMe = fef

        L = self.length
        self.u3 = 0
        self.u6 = 0
        if self.tipo == 1:
            FEM = [FAb, FSb - 3*FMb/(2*L), 0,
                   FAe, FSe + 3*FMb/(2*L), FMe - FMb/2]

            self.u3 = -FMb*L/(4*self.E*self.I)

        elif self.tipo == 2:
            FEM = [FAb, FSb - 3*FMe/(2*L), FMb - FMe/2,
                   FAe, FSe + 3*FMe/(2*L), 0]

            self.u6 = -FMe*L/(4*self.E*self.I)

        elif self.tipo == 3:
            FEM = [FAb, FSb - (FMb+FMe)/L, 0,
                   FAe, FSe + (FMb+FMe)/L, 0]

            self.u3 = -L/(6*self.E*self.I)*(2*FMb-FMe)
            self.u6 = -L/(6*self.E*self.I)*(2*FMe-FMb)

        else:
            FEM = [FAb, FSb, FMb, FAe, FSe, FMe]

        self.FEF = np.array(FEM).reshape(-1, 1)

    def set_displacement(self,):
        u1 = self.nodi.Ux
        u2 = self.nodi.Uz
        u3 = self.nodi.Ry
        u4 = self.nodj.Ux
        u5 = self.nodj.Uz
        u6 = self.nodj.Ry
        L = self.length

        if self.tipo == 1:
            u3 = self.u3 + 3*(-u2 + u5)/(2*L) - u6/2
            if self.nodi.restrain == (True, True, False):
                self.nodi.Ry = u3
        if self.tipo == 2:
            u6 = self.u6 + 3*(-u2 + u5)/(2*L) - u3/2
            if self.nodj.restrain == (True, True, False):
                self.nodj.Ry = u6
        if self.tipo == 3:
            u3 = self.u3 + (-u2 + u5) / L
            u6 = self.u3 + (-u2 + u5) / L
            if self.nodi.restrain == (True, True, False):
                self.nodi.Ry = u3
            if self.nodj.restrain == (True, True, False):
                self.nodj.Ry = u6

        Uint = np.array([u1,u2,u3,u4,u5,u6])
        self.Uint = self.T @ Uint

    def computar_Sistema(self, nnodos: int):
        if (self.FEF is not None) and (self.K is not None):
            K = np.zeros((nnodos * 3, nnodos * 3))
            Q = np.zeros((nnodos * 3, 1))
            indices = np.arange(nnodos * 3).reshape(nnodos, 3)
            inicial = self.nodi.index
            final = self.nodj.index

            ind = np.concatenate((indices[inicial], indices[final]), axis=0)
            idx = np.ix_(ind, ind)

            kgl_sub = np.dot(np.transpose(self.T), np.dot(self.K, self.T))
            qgl_sub = np.dot(np.transpose(self.T), self.FEF)

            K[idx] = kgl_sub
            Q[np.ix_(ind)] = qgl_sub

            self.Kgl = K
            self.Qgl = Q
            self.kgl_sub = kgl_sub
            self.qgl_sub = qgl_sub

    # Función auxiliar para formatear matrices
    def format_matrix(self, matrix, name, precision=3, f='f') -> str:
        if matrix is None:
            return f"{name}: No calculada\n"

        lines = [f"{name}:"]
        if name[:6] == 'FUERZA':
            lines.append(self.column_names(op=1, precision=precision))
        if name[:6] == 'DESPLA':
            lines.append(self.column_names(op=2, precision=precision))

        if matrix.ndim == 1:
            # Vector fila
            formatted_values = [f"{val:>{precision + 7}.{precision}{f}} |" for val in matrix]
            lines.append("  |" + " ".join(formatted_values))
        else:
            # Matriz
            for i, row in enumerate(matrix):
                if matrix.ndim == 2 and matrix.shape[1] == 1:
                    # Vector columna
                    formatted_val = f"{row[0]:>{precision + 7}.{precision}{f}}"
                    lines.append(f"  |{formatted_val}|")
                else:
                    # Matriz 2D
                    formatted_values = [f"{val:>{precision + 7}.{precision}{f}} |" for val in row]
                    lines.append("  | " + " ".join(formatted_values))
        return "\n".join(lines) + "\n"

    def column_names(self, op, precision = 3) -> str:
        opciones = {1: ['Fxi', 'Fzi', 'Myi', 'Fxj', 'Fzj', 'Myj'],
                    2: ['Uxi', 'Uzi', 'Ryi', 'Uxj', 'Uzj', 'Ryj']}
        lines = []
        if op in opciones:
            formatted_values = [f"{val:>{precision + 7}} |" for val in opciones[op]]
            lines.append("  |" + " ".join(formatted_values))
        else:
            formatted_values = [f"{val:>{precision + 7}} |" for val in op]
            lines.append("  |" + " ".join(formatted_values))
        return "\n".join(lines)

    def get_results(self):
        result = []
        result.append("-" * 80)
        result.append(f"ELEMENTO {self.ID}: [{self.Ni}, {self.Nj}]  - Tipo {self.tipo}")
        result.append("-" * 80)

        if self.FEF is not None:
            # Convertir FEF a vector fila si es vector columna
            fef_display = self.FEF.flatten() if self.FEF.ndim > 1 else self.FEF
            result.append(self.format_matrix(fef_display, "FUERZAS DE EMPOTRAMIENTO PERFECTO FEF"))

        if self.Fint is not None:
            # Convertir Fint a vector fila si es vector columna
            fint_display = self.Fint.flatten() if self.Fint.ndim > 1 else self.Fint
            result.append(self.format_matrix(fint_display, "FUERZAS INTERNAS"))

        if self.Uint is not None:
            # Convertir Fint a vector fila si es vector columna
            fint_display = self.Uint.flatten() if self.Uint.ndim > 1 else self.Uint
            result.append(self.format_matrix(fint_display, "DESPLAZAMIENTOS INTERNOS",f='e'))

        return "\n".join(result)

    def __str__(self) -> str:

        # Información básica del elemento
        result = []
        result.append("=" * 80)
        result.append(f"ELEMENTO {self.ID}: [{self.Ni}, {self.Nj}]  - Tipo {self.tipo}")
        result.append("=" * 80)
        result.append(f"PROPIEDADES")
        result.append(f"  Longitud: {self.length:.3f}")
        result.append(f"  Área (A): {self.A:.4e}")
        result.append(f"  Módulo elástico (E): {self.E:.4e}")
        result.append(f"  Momento de inercia (I): {self.I:.4e}")
        if self.weight is not None:
            result.append(f"  Peso específico: {self.weight:.3f}")

        release_i = "SÍ" if self.releases[0] else "NO"
        release_j = "SÍ" if self.releases[1] else "NO"
        result.append(f"  Liberaciones:  Nodo i = {release_i} | Nodo j = {release_j}")
        result.append(f"  Orientación: C={self.C:.3f} | S={self.S:.3f}")
        result.append("")

        # Cargas aplicadas
        result.append("CARGAS APLICADAS:")
        if len(self.loads) == 0:
            result.append("  No hay cargas aplicadas")
        else:
            for i, load in enumerate(self.loads):
                if hasattr(load, 'get_description'):
                    result.append(f"  {i + 1}) {load.get_description()}")
                else:
                    result.append(f"  {i + 1}) {type(load).__name__}")
        result.append("")

        # Matriz de transformación T
        result.append(self.format_matrix(self.T, "MATRIZ DE TRANSFORMACIÓN T (6x6)"))

        # Matrices calculadas después del análisis
        if self.K is not None:
            result.append(self.format_matrix(self.K, "MATRIZ DE RIGIDEZ LOCAL K (6x6)"))
            # Matriz de rigidez global T
            result.append(self.format_matrix(self.kgl_sub, "MATRIZ DE RIGIDEZ GLOBAL Tᵀ·K·T (6x6)"))

        if self.FEF is not None:
            # Convertir FEF a vector fila si es vector columna
            fef_display = self.FEF.flatten() if self.FEF.ndim > 1 else self.FEF
            result.append(self.format_matrix(fef_display, "FUERZAS DE EMPOTRAMIENTO PERFECTO FEF"))
            result.append(self.format_matrix(fef_display, "VECTOR DE FUERZAS GLOBALES Tᵀ·FEF"))

        if self.Fint is not None:
            # Convertir Fint a vector fila si es vector columna
            fint_display = self.Fint.flatten() if self.Fint.ndim > 1 else self.Fint
            result.append(self.format_matrix(fint_display, "RESULTADO FUERZAS INTERNAS"))

        if self.Uint is not None:
            # Convertir Fint a vector fila si es vector columna
            fint_display = self.Uint.flatten() if self.Uint.ndim > 1 else self.Uint
            result.append(self.format_matrix(fint_display, "RESULTADO DESPLAZAMIENTOS INTERNOS", f='e', precision=4))

        result.append("=" * 80)
        result.append("")

        return "\n".join(result)

# Clase principal de análisis
class StructuralMatrixAnalysis:
    """
    Motor de análisis matricial para estructuras planas (2D).

    Esta clase implementa el análisis estructural mediante el método
    matricial de la rigidez, permitiendo modelar estructuras compuestas
    por nodos y elementos, ensamblar la matriz global de rigidez,
    aplicar cargas equivalentes, condiciones de borde y resolver el
    sistema estructural.

    El modelo está orientado a análisis lineales en 2D, siendo extensible
    para distintos tipos de elementos estructurales (armaduras, vigas, columnas,
    pórticos, etc.).

    Autor: Anshel Chuquiviguel Zaña
    Contacto: anshel.chuquiviguel@utec.edu.pe

    Fecha de creación: 24-12-2025

    Última actualización: 24-12-2025

    Observaciones:
        - El análisis se basa en el método de rigidez clásica.
        - La clase actúa como contenedor principal de nodos y elementos.
        - Pensado como núcleo de un motor de análisis estructural extensible.
    """

    def __init__(self):
        self.analysis: bool = False
        self.K_GL: Optional[np.ndarray] = None
        self.FEF_GL: Optional[np.ndarray] = None

        self.Nodes: Optional[Dict[int, Node]] = {}
        self.Elements: Optional[Dict[int, Element]] = {}

        self.assignJoint = self.AssignJoint(self)
        self.assignElement = self.AssignElement(self)

    def clear(self):
        self.analysis: bool = False
        self.K_GL: Optional[np.ndarray] = None
        self.FEF_GL: Optional[np.ndarray] = None

        self.Nodes: Optional[Dict[int, Node]] = {}
        self.Elements: Optional[Dict[int, Element]] = {}

    def defineNode(self, ID:int, x: float, z: float):
        if ID not in self.Nodes:
            self.Nodes[ID] = Node(ID=ID, x=x, z=z)
        else:
            raise ValueError(f"El nodo {ID} ya se encuentra definido")

    class AssignJoint:
        def __init__(self, parent):
            self.parent = parent

        def restrictions(self, ID:int, ux:bool, uz:bool, ry:bool):
            if ID in self.parent.Nodes:
                self.parent.Nodes[ID].update_restrain((ux, uz, ry))
            else:
                raise ValueError(f"El nodo {ID} no se encuentra definido")

        def force(self, ID:int, fx:float = 0, fz:float = 0, my:float = 0):
            if ID in self.parent.Nodes:
                self.parent.Nodes[ID].set_force(fx, fz, my)
            else:
                raise ValueError(f"El nodo {ID} no se encuentra definido")

        def displacement(self, ID:int, ux:float = None, uz:float = None, ry:float = None):
            if ID in self.parent.Nodes:
                self.parent.Nodes[ID].set_displacement(ux, uz, ry)
            else:
                raise ValueError(f"El nodo {ID} no se encuentra definido")

    def defineElement(self, ID:int, Ni:int, Nj:int, A:float, E:float, I:float, weight:float = None):
        if Ni not in self.Nodes:
            raise ValueError(f"El nodo {Ni} no se encuentra definido")
        if Nj not in self.Nodes:
            raise ValueError(f"El nodo {Nj} no se encuentra definido")

        if ID not in self.Elements:
            self.Elements[ID] = Element(ID=ID, Ni=Ni, nodi=self.Nodes[Ni],
                                        Nj=Nj, nodj=self.Nodes[Nj],
                                        A=A, E=E, I=I)
            if weight is not None: self.Elements[ID].weight = weight
        else:
            raise ValueError(f"El elemento {ID} ya se encuentra definido")

    class AssignElement:
        def __init__(self, parent):
            self.parent = parent

        def pointForce(self, ID: int, P:float, distance: float = 0.5):
            if ID not in self.parent.Elements:
                raise ValueError(f"El elemento {ID} no se encuentra definido")

            load = PointForce(self.parent.Elements[ID].length, W=P, distance=distance)
            self.parent.Elements[ID].loads.append(load)

        def axialPointForce(self, ID: int, P:float, distance: float = 0.5):
            if ID not in self.parent.Elements:
                raise ValueError(f"El elemento {ID} no se encuentra definido")

            load = PointForce(self.parent.Elements[ID].length, W=P, distance=distance)
            self.parent.Elements[ID].loads.append(load)

        def pointMoment(self, ID: int, M:float, distance:float = 0.5):
            if ID not in self.parent.Elements:
                raise ValueError(f"El elemento {ID} no se encuentra definido")

            load = PointMoment(self.parent.Elements[ID].length, M=M, distance=distance)
            self.parent.Elements[ID].loads.append(load)

        def uniformLoad(self, ID: int, w:float):
            if ID not in self.parent.Elements:
                raise ValueError(f"El elemento {ID} no se encuentra definido")

            load = UniformLoad(self.parent.Elements[ID].length, w=w)
            self.parent.Elements[ID].loads.append(load)

        def temperatureLoad(self, ID: int, dT:float, alpha:float, d:float = None):
            if ID not in self.parent.Elements:
                raise ValueError(f"El elemento {ID} no se encuentra definido")
            elem = self.parent.Elements[ID]
            if d is None:
                load = TemperatureLoad(self.parent.Elements[ID].tipo, dT=dT, alpha=alpha,
                                       EA=elem.E * elem.A, EI=elem.E * elem.I)
            else:
                load = TemperatureLoad(self.parent.Elements[ID].tipo, dT=dT, alpha=alpha,
                                       d=d, EA=elem.E * elem.A, EI=elem.E * elem.I)

            self.parent.Elements[ID].loads.append(load)

        def axialUniformLoad(self, ID: int, w:float):
            if ID not in self.parent.Elements:
                raise ValueError(f"El elemento {ID} no se encuentra definido")

            load = AxialUniformLoad(self.parent.Elements[ID].length, w=w)
            self.parent.Elements[ID].loads.append(load)

        def trapezoidalLoad(self, ID: int, wi:float, wj:float):
            if ID not in self.parent.Elements:
                raise ValueError(f"El elemento {ID} no se encuentra definido")

            load = TrapezoidalLoad(self.parent.Elements[ID].length, w1=wi, w2=wj)
            self.parent.Elements[ID].loads.append(load)

        def triangularLoad(self,ID: int, w:float):
            if ID not in self.parent.Elements:
                raise ValueError(f"El elemento {ID} no se encuentra definido")

            load = CenteredTriangularLoad(self.parent.Elements[ID].length, w=w)
            self.parent.Elements[ID].loads.append(load)

        def gravityUniformLoad(self, ID:int, w:float):
            if ID not in self.parent.Elements:
                raise ValueError(f"El elemento {ID} no se encuentra definido")
            element = self.parent.Elements[ID]
            wx = element.S * w
            wy = element.C * w

            load_axial = AxialUniformLoad(self.parent.Elements[ID].length, w=wx)
            self.parent.Elements[ID].loads.append(load_axial)
            load_normal = UniformLoad(self.parent.Elements[ID].length, w=wy)
            self.parent.Elements[ID].loads.append(load_normal)

        def gravityPointLoad(self, ID:int, P:float, distance:float = 0.5):
            if ID not in self.parent.Elements:
                raise ValueError(f"El elemento {ID} no se encuentra definido")
            element = self.parent.Elements[ID]
            px = element.S * P
            py = element.C * P

            load_axial = AxialPointForce(self.parent.Elements[ID].length, W=px, distance=distance)
            self.parent.Elements[ID].loads.append(load_axial)
            load_normal = PointForce(self.parent.Elements[ID].length, W=py, distance=distance)
            self.parent.Elements[ID].loads.append(load_normal)

        def releases(self, ID:int, start: bool, end: bool):
            if ID in self.parent.Elements:
                self.parent.Elements[ID].releases = (start, end)
                self.parent.Elements[ID].actualizar_tipo()
            else:
                raise ValueError(f"El elemento {ID} no se encuentra definido")

    def getInfoNodes(self, ID: int = None):
        if ID is None or ID not in self.Nodes:
            [print(n) for n in self.Nodes.values()]
        else:
            print(self.Nodes[ID])

    def getInfoElements(self, ID: int = None):
        if ID is None or ID not in self.Elements:
            [print(n) for n in self.Elements.values()]
        else:
            print(self.Elements[ID])

    def getResults(self):
        print("=" * 80)
        print("RESULTADOS DE NODOS\n")
        [print(n) for n in self.Nodes.values()]
        print("=" * 80)
        print("RESULTADOS DE ELEMENTOS\n")
        [print(e.get_results()) for e in self.Elements.values()]


    def RunCompleteAnalysis(self, show_process: bool = True):

        def format_matrix(matrix, name, precision=3, f='f') -> str:
            if matrix is None:
                return f"{name}: No calculada\n"
            lines = [f"{name}"]
            if matrix.ndim == 1:
                # Vector fila
                formatted_values = [f"{val:>{10}.{precision}{f}} |" for val in matrix]
                lines.append("  |" + " ".join(formatted_values))
            else:
                # Matriz
                for i, row in enumerate(matrix):
                    if matrix.ndim == 2 and matrix.shape[1] == 1:
                        # Vector columna
                        formatted_val = f"{row[0]:>{10}.{precision}{f}}"
                        lines.append(f"  |{formatted_val}|")
                    else:
                        # Matriz 2D
                        formatted_values = [f"{val:>{10}.{precision}{f}} |" for val in row]
                        lines.append("  | " + " ".join(formatted_values))
            return "\n".join(lines)

        def print_subsection(title):
            print(f"\n{'-' * 80}")
            print(f" {title}")
            print(f"{'-' * 80}")


        list_nodes = list(self.Nodes.keys())
        nnodes = len(list_nodes)
        indices = np.arange(nnodes * 3).reshape(nnodes, 3)
        self.K_GL = np.zeros((nnodes * 3, nnodes * 3))
        self.Q_GL = np.zeros((nnodes * 3, 1))

        for nod in self.Nodes.values():
            nod.set_index(list_nodes)

        for elem in self.Elements.values():
            elem.computar_rigidez()
            elem.computar_FEF()
            elem.computar_Sistema(nnodes)
            self.K_GL += elem.Kgl
            self.Q_GL += elem.Qgl

        # IDENTIFICACIÓN DE GRADOS DE LIBERTAD
        zero_rows = np.where(np.all(self.K_GL == 0, axis=1))[0]

        # Inicializar los desplazamientos y fuerzas en los nodos
        Desplaz = np.array([[n.Ux, n.Uz, n.Ry] for n in self.Nodes.values()]).ravel()
        Fuerzas = np.array([[n.Fx, n.Fz, n.My] for n in self.Nodes.values()]).ravel()

        # Despreciar del análisis grados de libertad sin información
        for id in zero_rows:
            Desplaz[id] = 0
            Fuerzas[id] = np.nan

        # Identificar grados de libertad restringidos (A) y libres (B)
        A = np.isnan(Desplaz)
        B = np.isnan(Fuerzas)
        idx_A = np.where(A)[0]
        idx_B = np.where(B)[0]

        if show_process:
            print_subsection("1. INFORMACIÓN GENERAL")
            print(f"Número de nodos: {nnodes}")
            print(f"Número de elementos: {len(self.Elements)}")
            print(f"Grados de libertad totales: {nnodes * 3}")

            print(f"Grados de libertad restringidos (desplazamientos conocidos): {len(idx_B)} de {len(B)}")
            print(f"GL restringidos: {idx_B}")
            print(f"Grados de libertad libres (fuerzas conocidas): {len(idx_A)} de {len(A)}")
            print(f"GL libres: {idx_A}")

        # Particionar la matriz de rigidez global en submatrices
        KAA = self.K_GL[np.ix_(A, A)]
        KAB = self.K_GL[np.ix_(A, B)]
        KBA = self.K_GL[np.ix_(B, A)]
        KBB = self.K_GL[np.ix_(B, B)]

        Qk = self.Q_GL[A].reshape(-1, 1)
        Qu = self.Q_GL[B].reshape(-1, 1)

        dk = Desplaz[B].reshape(-1, 1)
        Pk = Fuerzas[A].reshape(-1, 1)

        if show_process:
            print_subsection("2. MATRICES PARTICIONADAS")
            print(format_matrix(idx_A, f"KAA ({KAA.shape[0]}x{KAA.shape[1]}):", precision=0, f='f'))
            print(format_matrix(KAA, "", precision=0, f='f'))
            print(' ')
            if KAB.size > 0:
                print(format_matrix(idx_B, f"KAB ({KAB.shape[0]}x{KAB.shape[1]}):", precision=0, f='f'))
                print(format_matrix(KAB, f" ", precision=0, f='f'))
                print(' ')
            print(format_matrix(Pk, "Pk (fuerzas conocidas):", precision=2))
            print(' ')
            print(format_matrix(dk, "dk (desplazamientos conocidas):", precision=2,f='e'))
            print(' ')
            print(format_matrix(Qk, "Qk (cargas equivalentes - libres):", precision=2))
            print('')
            print(format_matrix(Qu, "Qu (cargas equivalentes - restringidos):", precision=2))
            print('')

            print_subsection("3. RESOLUCIÓN DEL SISTEMA")

        # SOLUCIÓN DEL SISTEMA
        try:
            # Calcular desplazamientos desconocidos
            if KAB.size > 0:
                du = np.linalg.solve(KAA, (Pk - Qk - np.dot(KAB, dk)))
            else:
                du = np.linalg.solve(KAA, (Pk - Qk))

            # Calcular reacciones
            Pu = np.dot(KBA, du) + Qu + np.dot(KBB, dk)

            if show_process:
                print(format_matrix(du, "du (desplazamientos calculados):", precision=3, f='e'))
                print(' ')
                print(format_matrix(Pu, "Pu (reacciones calculadas):", precision=2, f='f'))
                print(' ')

        except np.linalg.LinAlgError as e:
            print(f"ERROR: No se pudo resolver el sistema - {str(e)}")
            return False

        # ASIGNACIÓN DE RESULTADOS A NODOS
        du = du.ravel()
        Pu = Pu.ravel()

        Fuerzas[B] = Pu
        Desplaz[A] = du
        desp = Desplaz.reshape(-1, 3)
        fuer = Fuerzas.reshape(-1, 3)

        for i, n in enumerate(self.Nodes.values()):
            n.Ux, n.Uz, n.Ry = desp[i]
            n.Fx, n.Fz, n.My = fuer[i]

        for i, e in enumerate(self.Elements.values()):
            inicial = e.nodi.index
            final = e.nodj.index

            ind = np.concatenate((indices[inicial], indices[final]), axis=0)
            u = Desplaz[ind].reshape(-1, 1)
            uu = np.dot(e.T, u)
            e.Fint = np.dot(e.K, uu) + e.FEF
            e.set_displacement()

        self.analysis = True

        print(f"ANÁLISIS COMPLETADO EXITOSAMENTE\n")


# Clase para visualización de estructuras en análisis matricial
class StructuralPlotter:
    """
    Clase para la visualización gráfica de estructuras 2D analizadas
    mediante el método matricial.

    Permite representar nodos, elementos, deformadas y resultados
    asociados al sistema de análisis estructural.

    Autor: Anshel Chuquiviguel
    Última actualización: 24-12-2025
    """

    def __init__(self, analysis_system: 'StructuralMatrixAnalysis'):
        self.system = analysis_system
        self.fig = None
        self.ax = None

        # Tamaños de elementos gráficos
        self.sizes = {
            'node': 50,
            'support': 0.25,
            'release': 0.1,
            'line_width': 2.0,
            'font_size': 10
        }

    def _setup_plot(self, figsize=(12, 8)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Z')

    def _get_structure_bounds(self) -> Tuple[float, float, float, float]:

        if not self.system.Nodes:
            return 0, 10, 0, 10

        x_coords = [node.x for node in self.system.Nodes.values()]
        z_coords = [node.z for node in self.system.Nodes.values()]

        x_min, x_max = min(x_coords), max(x_coords)
        z_min, z_max = min(z_coords), max(z_coords)

        # Añadir margen del 10%
        x_range = x_max - x_min if x_max != x_min else 100.0
        z_range = z_max - z_min if z_max != z_min else x_range
        margin_x = x_range * 0.25
        margin_z = z_range * 0.25

        return x_min - margin_x, x_max + margin_x, z_min - margin_z, z_max + margin_z

    def _get_scale_factor(self) -> float:
        x_min, x_max, z_min, z_max = self._get_structure_bounds()
        max_dimension = max(x_max - x_min, z_max - z_min)
        return max_dimension / 17.5

    def _draw_support_fixed(self, node: 'Node', scale: float):
        size = self.sizes['support'] * scale
        square = patches.Rectangle(
            (node.x - size / 2, node.z - size*1.5), size, size,
            facecolor='gray',
            edgecolor='black',
            linewidth=1.5,
        )
        self.ax.add_patch(square)
        self.ax.plot([node.x, node.x],
                     [node.z, node.z - size],
                     color='black', linewidth=1.5, alpha=0.6)

    def _draw_support_pinned(self, node: 'Node', scale: float):
        size = self.sizes['support'] * scale
        x = node.x
        z = node.z
        triangle = patches.Polygon([(x, z - size/4),
                                    (x - size / 2, z - size*5/4),
                                    (x + size / 2, z - size*5/4)],
            facecolor='gray',
            edgecolor='black',
            linewidth=1.5,
            alpha=0.8)
        self.ax.add_patch(triangle)

    def _draw_support_roller(self, node: 'Node', scale: float, direction: str = 'vertical'):
        size = self.sizes['support'] * scale

        # Líneas de rodillos
        if direction == 'vertical':
            circle = patches.Circle(
                (node.x, node.z - size*3/4), size / 2,
                facecolor='gray',
                edgecolor='black',
                linewidth=1.5,
                alpha=0.8
            )
            self.ax.add_patch(circle)

            # Base
            self.ax.plot([node.x - size, node.x + size],
                         [node.z - size * 1.3, node.z - size * 1.3],
                         color='black', linewidth=2)
        else:  # horizontal
            circle = patches.Circle(
                (node.x - size * 3 / 4, node.z), size / 2,
                facecolor='gray',
                edgecolor='black',
                linewidth=1.5,
                alpha=0.8
            )
            self.ax.add_patch(circle)

            # Base
            self.ax.plot([node.x - size * 1.3, node.x - size * 1.3],
                         [node.z - size, node.z + size],
                         color='black', linewidth=1)

    def _draw_supports(self, scale: float):
        if not self.system.Nodes:
            return

        for node in self.system.Nodes.values():
            if node.restrain is None:
                continue

            ux_restrained, uz_restrained, ry_restrained = node.restrain

            # Apoyo empotrado: (True, True, True)
            if ux_restrained and uz_restrained and ry_restrained:
                self._draw_support_fixed(node, scale)

            # Apoyo fijo/articulado: (True, True, False)
            elif ux_restrained and uz_restrained and not ry_restrained:
                self._draw_support_pinned(node, scale)

            # Apoyo móvil en Z: (False, True, False)
            elif not ux_restrained and uz_restrained and not ry_restrained:
                self._draw_support_roller(node, scale, 'vertical')

            # Apoyo móvil en X: (True, False, False)
            elif ux_restrained and not uz_restrained and not ry_restrained:
                self._draw_support_roller(node, scale, 'horizontal')

    def _draw_release(self, node: 'Node', element: 'Element', is_start: bool, scale: float):

        size = self.sizes['release'] * scale

        # Calcular posición de la liberación (ligeramente alejada del nodo)
        if is_start:
            other_node = element.nodj
        else:
            other_node = element.nodi

        # Vector unitario del elemento hacia afuera
        dx = node.x - other_node.x
        dz = node.z - other_node.z
        length = np.sqrt(dx ** 2 + dz ** 2)

        if length > 0:
            dx_unit = dx / length
            dz_unit = dz / length

            # Posición de la liberación
            release_x = node.x - dx_unit * size * 6
            release_z = node.z - dz_unit * size * 6

            self.ax.scatter(release_x, release_z, s=self.sizes['node']*1.25,
                            c='white', zorder=5, edgecolors='blue', linewidths=2)

    def _draw_releases(self, scale: float):
        if not self.system.Elements:
            return

        for element in self.system.Elements.values():
            if element.releases is None:
                continue

            release_i, release_j = element.releases

            if release_i:
                self._draw_release(element.nodi, element, True, scale)
            if release_j:
                self._draw_release(element.nodj, element, False, scale)

    def _draw_elements(self):
        if not self.system.Elements:
            return

        for element in self.system.Elements.values():
            node_i = element.nodi
            node_j = element.nodj
            color = 'blue'
            self.ax.plot([node_i.x, node_j.x], [node_i.z, node_j.z],
                         color=color, linewidth=self.sizes['line_width'],
                         alpha=0.8, solid_capstyle='round')

            # Etiqueta del elemento
            mid_x = (node_i.x + node_j.x) / 2
            mid_z = (node_i.z + node_j.z) / 2
            self.ax.text(mid_x, mid_z, str(element.ID),
                         ha='center', va='center',
                         fontsize=self.sizes['font_size'],
                         color='red',fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    def _draw_nodes(self):
        if not self.system.Nodes:
            return

        for node in self.system.Nodes.values():

            self.ax.scatter(node.x, node.z,
                            c='k', s=self.sizes['node'],
                            zorder=5, edgecolors='yellow', linewidths=1)

            self.ax.text(node.x, node.z + self._get_scale_factor() * 0.3, str(node.ID),
                         ha='center', va='bottom', zorder=10,
                         fontsize=self.sizes['font_size'],
                         color='k',
                         fontweight='bold')

    def showFrame(self):

        if not self.system.Nodes:
            print("Warning: No hay nodos para mostrar")
            return

        figsize=(14, 10)
        self._setup_plot(figsize)
        scale = self._get_scale_factor()

        # Dibujar elementos en orden
        if self.system.Elements:
            self._draw_elements()
            self._draw_releases(scale)

        self._draw_supports(scale)
        self._draw_nodes()

        # Ajustar límites de vista
        x_min, x_max, z_min, z_max = self._get_structure_bounds()
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(z_min, z_max)

        plt.tight_layout()
        plt.show()

    def _draw_node_reactions(self, scale_factor: float = 1.0, show_values: bool = True,
                            x: float = 0.0, z: float = 0.0, Fx: float = 0.0, Fz: float = 0.0, My: float = 0.0):

        # Tamaño base de las flechas (ajustable según la escala)
        arrow_length = scale_factor
        moment_radius = arrow_length * 0.5
        tolerance = 1e-3

        # 1. REACCIÓN HORIZONTAL (Fx)
        if not np.isnan(Fx) and abs(Fx) > tolerance:
            # Determinar dirección (positivo = derecha, negativo = izquierda)
            fx_magnitude = abs(Fx)
            fx_direction = 1 if Fx > 0 else -1

            # Escalar longitud de flecha según magnitud
            fx_arrow_length = arrow_length * min(fx_magnitude / max(abs(Fx), 1), 2.0)

            # Posición de inicio y fin de la flecha
            start_x = x - fx_direction * fx_arrow_length
            end_x = x
            arrow_y = z

            # Dibujar flecha horizontal
            self.ax.annotate('',
                             xy=(end_x, arrow_y),
                             xytext=(start_x, arrow_y),
                             arrowprops=dict(
                                 arrowstyle='->',
                                 color='k',
                                 lw=2,
                                 mutation_scale=20
                             ))

            if show_values:
                label_x = start_x - fx_direction * arrow_length * 0.3
                self.ax.text(label_x, arrow_y + scale_factor * 0.1,
                             f'Fx={Fx:.2f}',
                             ha='center', va='bottom',
                             fontsize=9, color='k',
                             bbox=dict(boxstyle='round,pad=0.2',
                                       facecolor='white', alpha=0.8))

        # 2. REACCIÓN VERTICAL (Fz)
        if not np.isnan(Fz) and abs(Fz) > tolerance:
            # Determinar dirección (positivo = arriba, negativo = abajo)
            fz_magnitude = abs(Fz)
            fz_direction = 1 if Fz > 0 else -1

            # Escalar longitud de flecha según magnitud
            fz_arrow_length = arrow_length * min(fz_magnitude / max(abs(Fz), 1), 2.0)

            # Posición de inicio y fin de la flecha
            start_z = z - fz_direction * fz_arrow_length
            end_z = z
            arrow_x = x

            # Dibujar flecha vertical
            self.ax.annotate('',
                             xy=(arrow_x, end_z),
                             xytext=(arrow_x, start_z),
                             arrowprops=dict(
                                 arrowstyle='->',
                                 color='k',
                                 lw=2,
                                 mutation_scale=20
                             ))

            if show_values:
                label_z = start_z - fz_direction * arrow_length * 0.3
                self.ax.text(arrow_x + scale_factor * 0.1, label_z,
                             f'Fz={Fz:.2f}',
                             ha='left', va='center',
                             fontsize=9, color='k',
                             bbox=dict(boxstyle='round,pad=0.2',
                                       facecolor='white', alpha=0.8))

        # 3. MOMENTO (My)
        if not np.isnan(My) and abs(My) > tolerance:
            # Determinar sentido de rotación (positivo = antihorario)
            my_direction = 1 if My > 0 else -1

            if my_direction > 0:  # Antihorario
                theta_start = 180
                theta_end = 90
            else:  # Horario
                theta_start = 90
                theta_end = 0

            # Crear arco
            arc = patches.Arc((x, z),
                              moment_radius * 2, moment_radius * 2,
                              theta1=theta_start, theta2=theta_end,
                              color='k', linewidth=2)
            self.ax.add_patch(arc)

            arrow_dx = 0.05 * scale_factor

            if my_direction>0:

                self.ax.annotate('',
                                 xy=(x - arrow_dx * 5, z + moment_radius),
                                 xytext=(x + arrow_dx * 0.5, z + moment_radius),
                                 arrowprops=dict(
                                     arrowstyle='->',
                                     color='k',
                                     lw=2,
                                     mutation_scale=20
                                 ))

            else:
                self.ax.annotate('',
                                 xy=(x + arrow_dx * 5, z + moment_radius),
                                 xytext=(x - arrow_dx * 0.5, z + moment_radius),
                                 arrowprops=dict(
                                     arrowstyle='->',
                                     color='k',
                                     lw=2,
                                     mutation_scale=20
                                 ))
            # Etiqueta con valor
            if show_values:
                self.ax.text(x, z + moment_radius + scale_factor * 0.4,
                             f'My={My:.2f}',
                             ha='center', va='top',
                             fontsize=9, color='k',
                             bbox=dict(boxstyle='round,pad=0.2',
                                       facecolor='white', alpha=0.8))

    def _hermite_shape_functions(self, xi):
        N1 = 1 - 3 * xi ** 2 + 2 * xi ** 3  # Para desplazamiento nodo 1
        N2 = xi - 2 * xi ** 2 + xi ** 3  # Para rotación nodo 1 (multiplicar por L)
        N3 = 3 * xi ** 2 - 2 * xi ** 3  # Para desplazamiento nodo 2
        N4 = -xi ** 2 + xi ** 3  # Para rotación nodo 2 (multiplicar por L)
        return np.array([N1, N2, N3, N4])

    def _calculate_auto_scale(self, scale:float):
        if not self.system.Nodes or not self.system.Elements:
            return 1.0, 1.0

        max_displacement = 0.0
        max_rotation = 0.0

        # Revisar desplazamientos nodales
        for node in self.system.Nodes.values():
            if not np.isnan(node.Ux):
                max_displacement = max(max_displacement, abs(node.Ux))
            if not np.isnan(node.Uz):
                max_displacement = max(max_displacement, abs(node.Uz))
            if not np.isnan(node.Ry):
                max_rotation = max(max_rotation, abs(node.Ry))

        if max_displacement == 0:
            scale_u = 1.0
        else:
            scale_u = scale / max_displacement
        if max_rotation == 0:
            scale_r = 1.0
        else:
            scale_r = scale / max_rotation

        return scale_u, scale_r

    def _beam_deformation(self, element: 'Element', scale_factor: Tuple[float, float], n_points=100):
        if not hasattr(element, 'Uint') or element.Uint is None:
            print(f"Warning: Element {element.ID} no tiene desplazamientos calculados")
            x_orig = np.array([element.nodi.x, element.nodj.x])
            z_orig = np.array([element.nodi.z, element.nodj.z])
            return x_orig, z_orig

        scale_u, scale_r = scale_factor
        # Desplazamientos nodales en coordenadas locales del elemento
        h1, v1, theta1, h2, v2, theta2 = element.Uint

        # Posiciones nodales deformadas en coordenadas globales
        # Desplazamientos nodales globales (escalados)
        node_i_ux = element.nodi.Ux if not np.isnan(element.nodi.Ux) else 0.0
        node_i_uz = element.nodi.Uz if not np.isnan(element.nodi.Uz) else 0.0
        node_j_ux = element.nodj.Ux if not np.isnan(element.nodj.Ux) else 0.0
        node_j_uz = element.nodj.Uz if not np.isnan(element.nodj.Uz) else 0.0

        # Posiciones deformadas de los nodos
        x1_def = element.nodi.x + scale_u * node_i_ux
        z1_def = element.nodi.z + scale_u * node_i_uz
        x2_def = element.nodj.x + scale_u * node_j_ux
        z2_def = element.nodj.z + scale_u * node_j_uz

        # PASO 2: Sistema de coordenadas del elemento deformado
        # Calcular la nueva orientación del elemento
        dx_def = x2_def - x1_def
        dz_def = z2_def - z1_def
        deformed_length = np.sqrt(dx_def ** 2 + dz_def ** 2)

        if deformed_length > 0:
            C_def = dx_def / deformed_length  # Coseno del elemento deformado
            S_def = dz_def / deformed_length  # Seno del elemento deformado
        else:
            # Usar orientación original si no hay cambio
            C_def = element.C if hasattr(element, 'C') else 1.0
            S_def = element.S if hasattr(element, 'S') else 0.0

        # Calcular deformación transversal usando funciones de Hermite
        xi_values = np.linspace(0, 1, n_points)  # Coordenada normalizada [0,1]

        # Vector de desplazamientos nodales locales (solo transversales)
        q_transversal = np.array([v1, theta1 * element.length, v2, theta2 * element.length])

        # Deformación transversal local
        v_local = np.zeros(n_points)
        for i, xi in enumerate(xi_values):
            N = self._hermite_shape_functions(xi)
            v_local[i] = np.dot(N, q_transversal) * scale_r

        # Coordenadas a lo largo del eje deformado
        # Posiciones base a lo largo del elemento deformado
        x_axis = x1_def + xi_values * dx_def
        z_axis = z1_def + xi_values * dz_def

        # Aplicar deformación transversal
        # La deformación transversal es perpendicular al eje del elemento
        # Vector perpendicular al elemento (rotado 90° antihorario)
        perp_x = -S_def  # Perpendicular en x
        perp_z = C_def  # Perpendicular en z

        # Coordenadas finales deformadas
        x_deformed = x_axis + v_local * perp_x
        z_deformed = z_axis + v_local * perp_z

        return x_deformed, z_deformed
    def _plot_element_deformation(self, scale: float, escala: float):
        scale_factor = self._calculate_auto_scale(scale*escala)
        tol = 10**-6
        for element in self.system.Elements.values():
            self.ax.plot([element.nodi.x, element.nodj.x], [element.nodi.z, element.nodj.z],
                         'k-', linewidth=1.25,  alpha = 0.50)
            x_d, z_d = self._beam_deformation(element=element,
                                              scale_factor=scale_factor)
            self.ax.plot(x_d, z_d, 'k--', linewidth=1.25, zorder=10)

        for node in self.system.Nodes.values():
            ux = node.Ux
            uz = node.Uz
            ry = node.Ry
            txt_list = []
            if abs(ux)>tol*100:
                txt_list.append(f"ux={ux:.3e}")
            if abs(uz)>tol*100:
                txt_list.append(f"uz={uz:.3e}")
            if abs(ry)>tol:
                txt_list.append(f"ry={ry:.3e}")
            if len(txt_list)>0:
                txt = "\n".join(txt_list)
                self.ax.text(node.x+scale*0.25, node.z+scale*0.3, txt,
                             ha='left', va='center',
                             fontsize=self.sizes['font_size']*0.75,
                             color='r', fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    def showDeformedShape(self, escala=1):
        if not self.system.analysis:
            print("Warning: No se ha realizado análisis. Use RunAnalysis() primero.")
            return
        figsize = (14, 10)
        self._setup_plot(figsize)
        scale = self._get_scale_factor()

        # Dibujar elementos en orden
        self._draw_supports(scale)
        self._draw_nodes()
        self._plot_element_deformation(scale, escala)

        # Ajustar límites de vista
        x_min, x_max, z_min, z_max = self._get_structure_bounds()
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(z_min, z_max)

        plt.tight_layout()
        plt.show()
    def _draw_all_reactions(self, show_values: bool = True):
        if not self.system.Nodes:
            return
        scale_factor = self._get_scale_factor()
        for node in self.system.Nodes.values():
            self._draw_node_reactions(scale_factor=scale_factor, show_values=show_values,
                                     x=node.x, z=node.z, Fx=node.Fx, Fz=node.Fz, My=node.My)

    def showReactions(self):
        if not self.system.analysis:
            print("Warning: No se ha realizado análisis. Use RunAnalysis() primero.")
            return
        figsize=(14, 10)
        self._setup_plot(figsize)
        scale = self._get_scale_factor()

        # Dibujar elementos
        self._draw_elements()
        self._draw_supports(scale)
        self._draw_releases(scale)
        self._draw_nodes()
        self._draw_all_reactions()

        # Ajustar límites de vista
        x_min, x_max, z_min, z_max = self._get_structure_bounds()
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(z_min, z_max)

        plt.tight_layout()
        plt.show()


