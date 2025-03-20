import numpy as np
import matplotlib.pyplot as plt

# Generate atomic numbers from 1 to 100
atomic_numbers = np.arange(1, 101)

# Create a synthetic trend:
# For simplicity, assume each period is 10 elements long.
# At the start of each period, the atomic radius "jumps" to a larger value,
# then decreases steadily across the period.
radii = []
for Z in atomic_numbers:
    # Determine which period the element belongs to
    period_index = (Z - 1) // 10  
    # Position within the period (0 to 9)
    period_position = (Z - 1) % 10  
    # Define a starting radius for each period that slightly decreases with higher periods
    start_radius = 2.5 - period_index * 0.2  
    # Assume a linear decrease within a period
    slope = 0.05  
    radius = start_radius - slope * period_position
    radii.append(radius)

plt.figure(figsize=(10, 6))
plt.plot(atomic_numbers, radii, marker='o', linestyle='-', color='green')
plt.xlabel("Atomic Number")
plt.ylabel("Atomic Radius (arbitrary units)")
plt.title("Rough Prediction of Atomic Radii Trend vs. Atomic Number")
plt.grid(True)
plt.show()
