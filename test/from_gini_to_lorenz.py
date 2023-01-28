import numpy as np
# coefficiente di Gini desiderato
gini = 0.5

# genera la distribuzione di Lorenz con il coefficiente di Gini desiderato
dist = lorenz_gen(gini)

# genera una serie di numeri con la distribuzione di Lorenz
numbers = dist.rvs(1000)