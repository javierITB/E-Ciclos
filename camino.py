from typing import List, Optional
from nodo import Nodo

class Camino:
    def __init__(self,
                 id_camino: int,
                 nodo_a: 'Nodo',
                 nodo_b: 'Nodo',
                 ciclovia: bool = False,
                 importancia: int = 1):
        self.id = id_camino
        self.nodos = (nodo_a, nodo_b)
        self.ciclovia = ciclovia
        self.importancia = importancia
        self.vecinos: List['Camino'] = []

        # Enlazar a nodos
        nodo_a.agregar_camino(self)
        nodo_b.agregar_camino(self)

    def agregar_vecino(self, otro: 'Camino'):
        if otro not in self.vecinos and otro is not self:
            self.vecinos.append(otro)

    def obtener_otro_nodo(self, nodo: 'Nodo') -> Optional['Nodo']:
        if nodo == self.nodos[0]:
            return self.nodos[1]
        elif nodo == self.nodos[1]:
            return self.nodos[0]
        return None

    def __repr__(self):
        return (f"Camino({self.id}, nodos=({self.nodos[0].id}, {self.nodos[1].id}), "
                f"ciclovia={self.ciclovia}, imp={self.importancia})")
