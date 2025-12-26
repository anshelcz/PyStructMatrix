# PyStructMatrix
Motor de análisis matricial para estructuras planas (2D)

**PyStructMatrix** es una librería en Python para el **análisis estructural en 2D mediante el método matricial**, orientada a pórticos y estructuras de barras.  
Está diseñada con un enfoque **modular, claro y extensible**, ideal tanto para uso académico como profesional.

La librería permite:
- Definir nodos, elementos y condiciones de borde
- Asignar propiedades geométricas y de material
- Aplicar cargas distribuidas y nodales
- Resolver el sistema estructural mediante el método matricial
- Visualizar la geometría y resultados del modelo estructural

---

### Prerequisites

You need to make sure you have installed the following modules.
* Requests
  ```s
  pip install numpy
  pip install matplotlib
  ```

---

## 📦 Instalación

```python
pip install PyStructMatrix
```

---

<!-- USAGE EXAMPLES -->
## Usage

* Example 1
    ```python
    from PyStructMatrix import StructuralMatrixAnalysis, StructuralPlotter
    
    # Crear el sistema estructural
    system = StructuralMatrixAnalysis()
    ```

_For more examples, please refer to the [Examples packages](https://github.com/anshelcz/PyStructMatrix/edit/main/examples)_

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/avmmodules/AVMWeather/issues) for a list of proposed features (and known issues).

## Características

✔ Análisis matricial de estructuras 2D
✔ Elementos tipo barra / pórtico
✔ Cargas distribuidas y nodales
✔ Liberaciones de momento
✔ Visualización básica del modelo
✔ Código orientado a objetos

## Componentes principales

### StructuralMatrixAnalysis

Clase principal que gestiona:
Definición de nodos y elementos
Ensamblaje de la matriz de rigidez global
Aplicación de cargas y condiciones de borde
Resolución del sistema estructural
Almacenamiento de resultados (desplazamientos, fuerzas internas, reacciones)

### StructuralPlotter

Herramienta de visualización para:
Geometría estructural
Numeración de nodos y elementos
Resultados del análisis (en desarrollo)

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Email: anshel.chuquiviguel@utec.edu.pe
