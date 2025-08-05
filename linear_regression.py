import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([[30], [50], [60], [80], [100]])
y = np.array([100, 150, 180, 240, 300])

model = LinearRegression()
model.fit(x, y)

predicted_price = model.predict([[70]])
print("Предсказанная цена квартиры 70 кв. метров:", predicted_price)

plt.scatter(x, y, color='blue', label='Данные')
plt.plot(x, model.predict(x), color='red', label='Линейная модель')
plt.xlabel("Площадь (кв. м)")
plt.ylabel("Цена (тыс. евро)")
plt.title("Простая линейная регрессия")
plt.legend()
plt.show()
