# import numpy as np
# import matplotlib.pyplot as plt
# import math

# def bezier_curve(t, points):
#     """Compute the coordinates of a point on a Bezier curve for a given parameter t."""
#     n = len(points) - 1
#     curve_point = np.zeros(2)
#     for i in range(n + 1):
#         binomial_coeff = math.comb(n, i)
#         curve_point += binomial_coeff * (t ** i) * ((1 - t) ** (n - i)) * points[i]
#     return curve_point

# def plot_bezier(points, num_points=20):
#     """Plot the Bezier curve and control points."""
#     print(f"Control Points: \n{points}")

#     t_values = np.linspace(0, 1, num_points)
#     # Print data type of arrays:    
#     print(f"t_values: {(t_values.dtype)}")
#     print(f"points: {(points.dtype)}")

#     bezier_points = np.array([bezier_curve(t, points) for t in t_values])

#     print(f"Bezier Points: \n{bezier_points}")

#     # Plotting the control points and the lines connecting them
#     plt.plot(points[:, 0], points[:, 1], 'ro--', label="Control Points")

#     # Plotting the Bezier curve
#     plt.plot(bezier_points[:, 0], bezier_points[:, 1], 'b-', label="Bezier Curve")
#     plt.title("Cubic Bezier Curve")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # Simplified control points for debugging
# control_points = np.array([
#     [0.00000000e+00, -5.54406643e-03],
#     [2.40000000e+01,  1.17993355e-03],
#     [4.80000000e+01,  8.31979513e-03],
#     [7.20000000e+01,  1.11682713e-02],
#     [9.60000000e+01,  9.08011198e-03],
#     [1.20000000e+02,  1.11891329e-02],
#     [1.44000000e+02,  8.28635693e-03]
# ])

# plot_bezier(control_points)

# import numpy as np
# import matplotlib.pyplot as plt
# import math

# def bezier_curve_weighted(t, points, weights):
#     """Compute the coordinates of a point on a weighted Bezier curve for a given parameter t."""
#     n = len(points) - 1
#     curve_point = np.zeros(2)
#     for i in range(n + 1):
#         binomial_coeff = math.comb(n, i)
#         weighted_point = binomial_coeff * (t ** i) * ((1 - t) ** (n - i)) * weights[i] * points[i]
#         curve_point += weighted_point
#     # Normalize the curve point by the sum of the weights
#     weight_sum = sum(binomial_coeff * (t ** i) * ((1 - t) ** (n - i)) * weights[i] for i in range(n + 1))
#     return curve_point / weight_sum

# def calculate_weights(points):
#     """Calculate weights based on the y-coordinate displacement between consecutive points."""
#     weights = [1.0]  # The first point always has a weight of 1
#     for i in range(1, len(points)):
#         displacement = abs(points[i][1] - points[i - 1][1])  # Calculate the y-coordinate displacement
#         weights.append(1 + displacement)  # Larger displacements increase the weight
#     return np.array(weights)

# def plot_weighted_bezier(points, num_points=100):
#     """Plot the weighted Bezier curve and control points."""
#     t_values = np.linspace(0, 1, num_points)
#     weights = calculate_weights(points)
#     bezier_points = np.array([bezier_curve_weighted(t, points, weights) for t in t_values])

#     # Plotting the control points and the lines connecting them
#     plt.plot(points[:, 0], points[:, 1], 'ro--', label="Control Points")

#     # Plotting the weighted Bezier curve
#     plt.plot(bezier_points[:, 0], bezier_points[:, 1], 'b-', label="Weighted Bezier Curve")
#     plt.title("Weighted Bezier Curve Based on Y-Displacement")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # Example control points
# control_points = np.array([
#     [0, 0],
#     [1, 2],
#     [3, 3],
#     [4, 0]
# ])

# plot_weighted_bezier(control_points)

import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.widgets import Slider

def bezier_curve(t, points, weights):
    """Compute the coordinates of a point on a weighted Bezier curve for a given parameter t."""
    n = len(points) - 1
    curve_point = np.zeros(2)
    for i in range(n + 1):
        binomial_coeff = math.comb(n, i)
        curve_point += binomial_coeff * (t ** i) * ((1 - t) ** (n - i)) * weights[i] * points[i]
    return curve_point

def plot_bezier_with_sliders(points, num_points=20):
    """Plot the Bezier curve and control points with interactive sliders to adjust weights."""
    weights = np.ones(len(points))  # Initial weights of 1 for all control points
    t_values = np.linspace(0, 1, num_points)

    # Create the plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.3)
    
    # Initial Bezier curve
    bezier_points = np.array([bezier_curve(t, points, weights) for t in t_values])
    line, = ax.plot(bezier_points[:, 0], bezier_points[:, 1], 'b-', label="Bezier Curve")
    ax.plot(points[:, 0], points[:, 1], 'ro--', label="Control Points")
    ax.legend()
    ax.set_title("Cubic Bezier Curve with Adjustable Weights")
    ax.grid(True)

    # Display initial weights
    weight_text = ax.text(0.02, 0.95, f"Weights: {weights}", transform=ax.transAxes, fontsize=12, verticalalignment='top')

    # Slider for selecting the weight of a point
    ax_weight = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor='lightgoldenrodyellow')
    weight_slider = Slider(ax_weight, 'Weight', -2.0, 2.0, valinit=1.0)

    # Slider for selecting which control point to adjust
    ax_point = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor='lightgoldenrodyellow')
    point_slider = Slider(ax_point, 'Control Point', 0, len(points) - 1, valinit=0, valstep=1)

    def update(val):
        # Update the weight of the selected control point
        selected_point = int(point_slider.val)
        weights[selected_point] = weight_slider.val

        # Update the displayed weights
        weight_text.set_text(f"Weights: {weights}")

        # Recompute the Bezier curve with the updated weights
        bezier_points = np.array([bezier_curve(t, points, weights) for t in t_values])

        # Update the plot with the new Bezier curve
        line.set_ydata(bezier_points[:, 1])
        line.set_xdata(bezier_points[:, 0])
        fig.canvas.draw_idle()

    # Update the plot whenever a slider is adjusted
    weight_slider.on_changed(update)
    point_slider.on_changed(update)

    plt.show()

# Simplified control points for debugging
control_points = np.array([
    [0.00000000e+00, -5.54406643e-03],
    [2.40000000e+01,  1.17993355e-03],
    [4.80000000e+01,  8.31979513e-03],
    [7.20000000e+01,  1.11682713e-02],
    [9.60000000e+01,  9.08011198e-03],
    [1.20000000e+02,  1.11891329e-02],
    [1.44000000e+02,  8.28635693e-03]
])

plot_bezier_with_sliders(control_points)
