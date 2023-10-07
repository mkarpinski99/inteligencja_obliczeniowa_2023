import pandas
import matplotlib.pyplot as plt

miasta = pandas.read_csv('miasta.csv')

print(miasta)
print(miasta.values)
df = pandas.DataFrame(miasta)
print(df)
df = df.append({"Rok": 2010, "Gdansk": 460, "Poznan": 555, "Szczecin": 405}, ignore_index=True)

plt.plot(df.iloc[:, 0], df.iloc[:, 1], color='red', marker='o', label='Gdańsk')
plt.legend(loc="upper left")
plt.xlabel('Lata') 
plt.ylabel('Liczba ludności w tys') 
plt.title('Ludność w miastach Polski') 
plt.show()

plt.plot(df.iloc[:, 0], df.iloc[:, 2], color='blue', marker='o', label='Poznań')
plt.plot(df.iloc[:, 0], df.iloc[:, 3], color='orange', marker='o', label='Szczecin')
plt.plot(df.iloc[:, 0], df.iloc[:, 1], color='red', marker='o', label='Gdańsk')
plt.legend(loc="upper left")
plt.xlabel('Lata')
plt.ylabel('Liczba ludności w tys')
plt.title('Ludność w miastach Polski')
plt.show()
