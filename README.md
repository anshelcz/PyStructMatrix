# PyStructMatrix
Motor de análisis matricial para estructuras planas (2D)

**PyStructMatrix** es una librería en Python para el **análisis estructural en 2D mediante el método matricial**, orientada a pórticos y estructuras de barras.  
Está diseñada con un enfoque **modular, claro y extensible**, ideal tanto para uso académico como profesional.

## Características

✔ Análisis matricial de estructuras 2D  
✔ Elementos tipo barra / pórtico  
✔ Cargas distribuidas y nodales  
✔ Liberaciones de momento  
✔ Visualización básica del modelo  
✔ Código orientado a objetos  

## 📋 Prerrequisitos

Asegúrate de tener instalados los siguientes módulos:

```bash
pip install numpy
pip install matplotlib
```

## 🚀 Instalación

```bash
pip install PyStructMatrix
```

## 📖 Uso

Ejemplo básico:

```python
from PyStructMatrix import StructuralMatrixAnalysis, StructuralPlotter

# Crear el sistema estructural
system = StructuralMatrixAnalysis()

# Visualizar resultados
plotter = StructuralPlotter(system)
```

_Para más ejemplos, consulta el [directorio de ejemplos](https://github.com/anshelcz/PyStructMatrix/edit/main/examples)._

## 🏗️ Componentes principales

### StructuralMatrixAnalysis

Clase principal que gestiona:
- Definición de nodos y elementos
- Ensamblaje de la matriz de rigidez global
- Aplicación de cargas y condiciones de borde
- Resolución del sistema estructural
- Almacenamiento de resultados (desplazamientos, fuerzas internas, reacciones)

### StructuralPlotter

Herramienta de visualización para:
- Geometría estructural
- Numeración de nodos y elementos
- Resultados del análisis (en desarrollo)

## 📊 Funcionalidades

- **Análisis estático lineal** de estructuras planas
- **Soporte para diferentes tipos de elementos**: barras axiales y elementos viga-columna
- **Cargas aplicables**: fuerzas nodales, momentos y cargas distribuidas
- **Condiciones de apoyo**: empotrado, articulado, rodillo
- **Resultados**: desplazamientos, fuerzas internas, reacciones de apoyo
- **Visualización**: gráficos de la estructura antes y después de la deformación

## 📄 Licencia

Distribuido bajo la Licencia MIT. Consulta el archivo `LICENSE` para más información.

## 📧 Contacto

**Autor:** Anshel Chuquiviguel  
**Email:** anshel.chuquiviguel@utec.edu.pe  
